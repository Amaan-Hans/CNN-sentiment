import os
import json
import time
import string
import numpy as np
from typing import List, Tuple, Dict

# ================================
# Text Preprocessing for Word2Vec
# ================================
class TextPreProc:
    def __init__(self):
        pass

    def retrieve_all_processed_data_from_text(self, text: str):
        """
        Returns:
          processed_text: List[str]
          vocab: List[str]
          word_to_idx: Dict[str, int]
          idx_to_word: Dict[int, str]
        """
        processed_text = self.full_process_text(text)
        vocab = self.get_all_unique_words_in_order(processed_text)
        word_to_idx = self.get_word_to_idx(vocab)
        idx_to_word = self.get_idx_to_word(vocab)
        return processed_text, vocab, word_to_idx, idx_to_word

    def full_process_text(self, text: str) -> List[str]:
        """
        Remove punctuation, lowercase, and split on whitespace.
        """
        text = self._remove_punctuation(text)
        text = self._process_text(text)
        return text

    def _remove_punctuation(self, text: str) -> str:
        """
        Removes punctuation and common unicode quotes/dashes/ellipsis.
        """
        extra_punct = "“”‘’—…"
        all_punct = string.punctuation + extra_punct
        return text.translate(str.maketrans('', '', all_punct))

    def _process_text(self, text: str) -> List[str]:
        """
        Get individual words, remove extra spaces, lowercase all and return array of words.
        """
        return (text.lower()).split()

    def get_all_unique_words_in_order(self, word_list: List[str]) -> List[str]:
        """
        Returns all unique words in the text in the order they appear.
        """
        return list(dict.fromkeys(word_list))

    def get_word_to_idx(self, word_list: List[str]) -> Dict[str, int]:
        """
        Returns a dictionary mapping each word to its index in the list.
        """
        return {word: idx for idx, word in enumerate(word_list)}

    def get_idx_to_word(self, word_list: List[str]) -> Dict[int, str]:
        """
        Returns a dictionary mapping each index to its corresponding word.
        """
        return {idx: word for idx, word in enumerate(word_list)}

    def get_skipgram_pairs(
        self,
        processed_text: List[str],
        word_to_idx_dictionary: Dict[str, int],
        window_size: int = 2
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[int, int]]]:
        """
        Returns (center_word, context_word) pairs as well as (idx_center, idx_context) pairs.
        """
        pairs = []
        idx_pairs = []
        for i in range(window_size, len(processed_text) - window_size):
            center = processed_text[i]
            context = processed_text[i - window_size:i] + processed_text[i + 1:i + window_size + 1]
            ci = word_to_idx_dictionary.get(center, None)
            if ci is None:
                continue
            for cw in context:
                wi = word_to_idx_dictionary.get(cw, None)
                if wi is None:
                    continue
                pairs.append((center, cw))
                idx_pairs.append((ci, wi))
        return pairs, idx_pairs


# =======================================
# Word2Vec Skip-Gram Model Implementation
# =======================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.05)
        nn.init.normal_(self.output.weight, mean=0, std=0.05)
        nn.init.zeros_(self.output.bias)

    def forward(self, center_word_idx: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(center_word_idx)   # [B, D]
        out = self.output(embeds)                   # [B, V]
        return out

    @torch.no_grad()
    def get_embedding(self, word_idx: int | torch.Tensor) -> torch.Tensor:
        if isinstance(word_idx, int):
            word_idx = torch.tensor(word_idx, device=self.embeddings.weight.device)
        else:
            word_idx = word_idx.to(self.embeddings.weight.device)
        return self.embeddings(word_idx)


# =========================
# Dataset and Training Loop
# =========================
class SkipGramDataset(Dataset):
    def __init__(self, data: List[Tuple[int, int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        center, context = self.data[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class TrainModel:
    def __init__(
        self,
        skipgram_idx_pairs: List[Tuple[int, int]],
        vocab: List[str],
        word_to_idx_dictionary: Dict[str, int],
        embedding_dim: int = 100,
        epochs: int = 10,
        learning_rate: float = 0.05,
        batch_size: int = 1024,
        model_override: SkipGramModel | None = None,
        seed: int = 123
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.skipgram_idx_pairs = skipgram_idx_pairs
        self.vocab = vocab
        self.word_to_idx_dictionary = word_to_idx_dictionary

        self.dataset = SkipGramDataset(skipgram_idx_pairs)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=True
        )

        self.model = (model_override or SkipGramModel(len(vocab), self.embedding_dim)).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        print("\nStarting training on combined corpus...")
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0.0
            n_batches = 0

            for centers, contexts in self.dataloader:
                centers = centers.to(self.device, non_blocking=True)
                contexts = contexts.to(self.device, non_blocking=True)

                logits = self.model(centers)
                loss = self.loss_fn(logits, contexts)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg = total_loss / max(1, n_batches)
            print(f"Epoch {epoch+1:02d}/{self.epochs} | Avg Loss: {avg:.4f} | Time: {time.time()-start_time:.2f}s")

        print("Training completed.\n")


# ==========================
# Utility: Load all HP books
# ==========================
def load_and_combine_text(file_list: List[str]) -> str:
    combined_text = ""
    print("\n=== Word Counts for Each File ===")
    for fp in file_list:
        with open(fp, 'r', encoding='utf-8') as f:
            txt = f.read()
            wc = len(txt.split())
            print(f"{fp}: {wc:,} words")
            combined_text += txt + " "
    total = len(combined_text.split())
    print(f"\nTotal combined word count: {total:,} words")
    print("================================\n")
    return combined_text

# =========================
# Page segmentation helpers
# =========================

BOOK_NAMES = [
    "HP1", "HP2", "HP3", "HP4", "HP5", "HP6", "HP7"
]
BOOK_TO_LABEL = {name: i for i, name in enumerate(BOOK_NAMES)}
LABEL_TO_BOOK = {i: name for name, i in BOOK_TO_LABEL.items()}


def segment_into_pages(processed_words, page_length=250):
    """
    Split a list of processed words into equal-length pages of `page_length`.
    The last page may be shorter.
    Returns: List[List[str]]
    """
    pages = []
    for i in range(0, len(processed_words), page_length):
        pages.append(processed_words[i:i+page_length])
    return pages


def pages_to_word_indices(pages, word_to_idx):
    """
    Convert each page (list of words) into a list of word indices (variable length).
    Returns: List[List[int]]
    """
    idx_pages = []
    for page in pages:
        idxs = [word_to_idx[w] for w in page if w in word_to_idx]
        if len(idxs) == 0:
            continue
        idx_pages.append(idxs)
    return idx_pages


def build_labeled_pages_for_books(
    file_paths_per_book,  # List[str] in HP1..HP7 order
    preprocessor,         # TextPreProc()
    word_to_idx,          # combined corpus mapping
    page_length=250
):
    """
    For each book file, preprocess text, segment into 250-word pages,
    convert to word indices using the global vocabulary, and attach labels.

    Returns:
      all_pages_indices: List[List[int]]   (variable-length sequences)
      all_labels:       List[int]          (0..6)
      meta:             List[dict]         (book_name, label, page_id, length)
    """
    all_pages_indices = []
    all_labels = []
    meta = []

    assert len(file_paths_per_book) == 7, "Expect exactly seven books (HP1..HP7)."

    for book_idx, fp in enumerate(file_paths_per_book):
        book_name = BOOK_NAMES[book_idx]
        label = BOOK_TO_LABEL[book_name]

        with open(fp, 'r', encoding='utf-8') as f:
            raw = f.read()

        words = preprocessor.full_process_text(raw)  # reuse same cleaning as training
        pages = segment_into_pages(words, page_length=page_length)
        idx_pages = pages_to_word_indices(pages, word_to_idx)

        start_page_id = len(all_pages_indices)
        all_pages_indices.extend(idx_pages)
        all_labels.extend([label] * len(idx_pages))

        # record metadata for later analysis/debugging
        for local_p, idxs in enumerate(idx_pages):
            meta.append({
                "global_page_id": start_page_id + local_p,
                "book_name": book_name,
                "label": label,
                "local_page_id": local_p,
                "token_count": len(idxs)
            })

        print(f"{book_name}: {len(idx_pages)} pages")

    print(f"Total labeled pages: {len(all_pages_indices)}")
    return all_pages_indices, all_labels, meta


def pad_pages_to_fixed_length(idx_pages, pad_idx, fixed_len=250):
    """
    Optional: pad or truncate pages to a fixed length for CNN input.
    If you prefer to pad later in a DataLoader's collate_fn, you can skip this.

    Returns: torch.LongTensor [N, fixed_len]
    """
    import torch
    N = len(idx_pages)
    out = torch.full((N, fixed_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(idx_pages):
        L = min(len(seq), fixed_len)
        out[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
    return out

# =================
# MAIN: Train once
# =================
if __name__ == "__main__":
    # Adjust these paths to your dataset location
    files_to_load = [
        './harry_potter_books/HP1.txt',
        './harry_potter_books/HP2.txt',
        './harry_potter_books/HP3.txt',
        './harry_potter_books/HP4.txt',
        './harry_potter_books/HP5.txt',
        './harry_potter_books/HP6.txt',
        './harry_potter_books/HP7.txt',
    ]

    # Load and preprocess the entire 7-book corpus once
    corpus_text = load_and_combine_text(files_to_load)
    pre = TextPreProc()
    processed, vocab, w2i, i2w = pre.retrieve_all_processed_data_from_text(corpus_text)

    # Build skip-gram training pairs on the combined corpus
    window_size = 2  # you can vary this for receptive field experiments later
    _, idx_pairs = pre.get_skipgram_pairs(processed, w2i, window_size=window_size)
    print(f"Total skip-gram pairs: {len(idx_pairs):,}")

    # Train the skip-gram model once on the full corpus
    trainer = TrainModel(
        skipgram_idx_pairs=idx_pairs,
        vocab=vocab,
        word_to_idx_dictionary=w2i,
        embedding_dim=100,
        epochs=10,           # increase if you have time
        learning_rate=0.05,
        batch_size=1024,     # raise/lower based on memory
        seed=123
    )
    trainer.train()

    # Save encoder artifacts for Lab 2 CNN
    torch.save(trainer.model.embeddings.weight.data.cpu(), 'word2vec_embeddings.pth')
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f)

    print("Saved artifacts:")
    print(" - vocab.json")
    print(" - word2vec_embeddings.pth")

    # Optional helper: dump a minimal config for later CNN script
    meta = {
        "embedding_dim": trainer.embedding_dim,
        "window_size": window_size,
        "vocab_size": len(vocab)
    }
    with open('encoder_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(" - encoder_meta.json")



        # =========================
    # Build labeled page dataset
    # =========================
    hp_files_in_order = [
        './harry_potter_books/HP1.txt',
        './harry_potter_books/HP2.txt',
        './harry_potter_books/HP3.txt',
        './harry_potter_books/HP4.txt',
        './harry_potter_books/HP5.txt',
        './harry_potter_books/HP6.txt',
        './harry_potter_books/HP7.txt',
    ]

    # If you want a dedicated PAD token for fixed-length CNN tensors:
    # (1) add it to the vocab here (post-training so embeddings do not change),
    # (2) assign it a new index, and (3) expand saved vocab.json.
    # For Lab 2 you can also keep variable lengths and pad in a collate_fn.
    USE_PAD = True
    PAD_TOKEN = "<pad>"
    if USE_PAD and PAD_TOKEN not in w2i:
        # extend vocab and mapping only for downstream dataset usage
        w2i[PAD_TOKEN] = len(w2i)
        vocab.append(PAD_TOKEN)
        with open('vocab.json', 'w', encoding='utf-8') as f:
            json.dump(vocab, f)
        print("Appended <pad> token to vocab for dataset padding.")

    all_pages_indices, all_labels, pages_meta = build_labeled_pages_for_books(
        hp_files_in_order, pre, w2i, page_length=250
    )

    # Optional: pad to fixed length for a CNN that expects [N, 250]
    if USE_PAD:
        pad_idx = w2i[PAD_TOKEN]
        X_indices = pad_pages_to_fixed_length(all_pages_indices, pad_idx, fixed_len=250)  # LongTensor [N, 250]
    else:
        # Keep as a Python list of variable-length sequences (pad later in DataLoader)
        # For saving, we convert to an object array in numpy or save a list with torch.save.
        import torch
        X_indices = all_pages_indices  # list of lists; will torch.save as-is

    # Convert labels to tensor
    import torch
    y_labels = torch.tensor(all_labels, dtype=torch.long)

    # Save dataset artifacts for CNN training
    torch.save(X_indices, 'X_indices.pt')   # either LongTensor [N,250] or list of lists
    torch.save(y_labels, 'y_labels.pt')

    with open('pages_meta.json', 'w', encoding='utf-8') as f:
        json.dump(pages_meta, f, indent=2)

    print("Saved dataset artifacts:")
    print(" - X_indices.pt")
    print(" - y_labels.pt")
    print(" - pages_meta.json")


