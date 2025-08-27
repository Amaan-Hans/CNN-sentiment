import os, json, string, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from collections import defaultdict

# ---------- Preprocessing ----------
class TextPreProc:
    def _rm_punct(self, s):
        # remove punctuation and odd quotes/dashes used in ebooks
        return s.translate(str.maketrans('', '', string.punctuation + "“”‘’—…"))

# ---------- Dataset Builder (each line = one page) ----------
def build_pages(file_paths, pre, w2i, unk_idx):
    pages, labels, meta = [], [], []
    for book_id, fp in enumerate(file_paths):
        with open(fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        count_before = len(pages)
        for line in lines:
            words = pre._rm_punct(line).lower().split()
            idxs = [w2i.get(w, unk_idx) for w in words]  # map OOV -> UNK
            if idxs:
                pages.append(idxs)
                labels.append(book_id)
                meta.append({"book": os.path.basename(fp), "len": len(idxs)})
        print(f"{os.path.basename(fp)} pages: {len(pages)-count_before}")
    return pages, labels, meta

def pad_pages(pages, pad_idx):
    # pad to the longest line (you can clip to a percentile if memory becomes an issue)
    max_len = max(len(p) for p in pages)
    X = torch.full((len(pages), max_len), pad_idx, dtype=torch.long)
    for i, p in enumerate(pages):
        X[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    return X

# ---------- TextCNN ----------
class TextCNN(nn.Module):
    def __init__(self, num_emb, d, emb_weight, pad_idx, n_classes=7, n_f=100, kernels=(3,4,5), drop=0.5, freeze=True):
        super().__init__()
        # Embedding layer initialised from pretrained word2vec (HP1) + our extensions
        self.emb = nn.Embedding(num_emb, d, padding_idx=pad_idx)
        if emb_weight is not None:
            with torch.no_grad():
                sz0, sz1 = min(num_emb, emb_weight.size(0)), min(d, emb_weight.size(1))
                self.emb.weight[:sz0, :sz1].copy_(emb_weight[:sz0, :sz1])
        self.emb.weight.requires_grad = not freeze

        # Convolutional feature extractors (n-gram detectors)
        self.convs = nn.ModuleList([nn.Conv1d(d, n_f, k) for k in kernels])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(n_f * len(kernels), n_classes)  # classifier head

        self.num_filters = n_f
        self.kernel_sizes = kernels

    def forward(self, x):
        # x: [B, L] -> emb: [B, L, D] -> transpose to [B, D, L] for Conv1d
        x = self.emb(x).transpose(1, 2)
        feats = [torch.max(self.act(conv(x)), dim=2).values for conv in self.convs]  # global max pool
        h = self.drop(torch.cat(feats, dim=1))
        return self.fc(h)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def filter_info(model, dataloader, vocab, k = 10, batch_size = 1024, device = "cpu"):

    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    idx2word = {idx: word for word, idx in vocab.items()}

    model.to(device)
    model.eval()

    num_filters = model.num_filters

    all_activations = [defaultdict(list) for _ in model.kernel_sizes]
    correct_counts = [defaultdict(int) for _ in model.kernel_sizes]
    total_counts = [defaultdict(int) for _ in model.kernel_sizes]

    W = model.fc.weight.data.cpu()
    filter_class_identity = []
    filter_offset = 0

    for layer_idx in range(len(model.kernel_sizes)):

        for f in range(num_filters):
            col = filter_offset + f
            class_id = torch.argmax(W[:, col]).item()
            filter_class_identity.append(class_id)
        filter_offset += num_filters

    with torch.no_grad():
        for page, book_number in dataloader:
            page = page.to(device)
            book_number = book_number.to(device)

            logits = model(page)
            loss = loss_function(logits, book_number)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == book_number).sum().item()
            total += book_number.size(0)

            embedded = model.emb(page)
            embedded = embedded.permute(0, 2, 1)

            filter_offset = 0
            for layer_idx, (convolution, kernel_size) in enumerate(zip(model.convs, model.kernel_sizes)):
                convolution_output = F.relu(convolution(embedded))
                values, indices = convolution_output.max(dim=2)
                B, num_f = values.shape

                for b in range(B):
                    for f in range(num_f):
                        start = indices[b, f].item()
                        end = start + kernel_size
                        token_indices = page[b, start:end].tolist()
                        words = [idx2word[t] for t in token_indices if t != 0]
                        ngram_string = " ".join(words)
                        activation = values[b, f].item()
                        class_id = filter_class_identity[filter_offset + f]
                        is_correct = int(class_id == book_number[b].item())

                        all_activations[layer_idx][f].append(
                            (ngram_string, activation, class_id, is_correct)
                        )
                        correct_counts[layer_idx][f] += is_correct
                        total_counts[layer_idx][f] += 1

                filter_offset += num_f

    top_ngrams = [{} for _ in model.kernel_sizes]
    identity_accuracy = [{} for _ in model.kernel_sizes]
    for layer_idx, layer_dict in enumerate(all_activations):
        for f, ngram_list in layer_dict.items():
            ngram_list.sort(key=lambda x: x[1], reverse=True)
            top_ngrams[layer_idx][f] = ngram_list[:k]
            identity_accuracy[layer_idx][f] = (
                correct_counts[layer_idx][f] / total_counts[layer_idx][f]
            )

    return avg_loss, accuracy, top_ngrams, identity_accuracy

def visualize_filters(top_ngrams, identity_accuracy, kernel_sizes):

    rows = []

    for layer_idx, layer_dict in enumerate(top_ngrams):
        for filter_idx, ngrams in layer_dict.items():
            class_id = ngrams[0][2]
            acc = identity_accuracy[layer_idx][filter_idx]
            top_ngrams_str = ", ".join([f"'{ngram[0]}'" for ngram in ngrams])
            
            rows.append({
                "Layer": layer_idx,
                "Kernel Size": kernel_sizes[layer_idx],
                "Filter": filter_idx,
                "Class Identity": class_id,
                "Identity Accuracy": f"{acc:.2f}",
                "Top-k N-grams": top_ngrams_str
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Layer", "Filter"]).reset_index(drop=True)    
    return df

# ---------- Main ----------
if __name__ == "__main__":
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BOOKS = ["HP1","HP2","HP3","HP4","HP5","HP6","HP7"]
    FILES = [f'./harry_potter_books/{b}.txt' for b in BOOKS]

    # 1) Load Lab-1 encoder (HP1) on CPU safely
    print("Loading base encoder (HP1)...")
    emb = torch.load("word2vec_embeddings.pth", map_location=torch.device('cpu')).cpu()  # [V, D]
    with open("./vocab.json","r",encoding="utf-8") as f:
        vocab = json.load(f)
    w2i = {w:i for i,w in enumerate(vocab)}
    d = emb.size(1)  # current embedding dimension D

    # 2) Add PAD and UNK as extra rows
    PAD, UNK = "<pad>", "<unk>"
    if PAD not in w2i:
        w2i[PAD] = len(w2i); vocab.append(PAD)
        emb = torch.cat([emb, torch.zeros(1, d, device=emb.device)], dim=0)
    if UNK not in w2i:
        w2i[UNK] = len(w2i); vocab.append(UNK)
        emb = torch.cat([emb, torch.zeros(1, d, device=emb.device)], dim=0)
    pad_idx = w2i[PAD]; unk_idx = w2i[UNK]

    # 3) 
    emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device)], dim=1)  # [V, D+1]
    emb[unk_idx, -1] = 1.0  # only UNK has '1' in the last coordinate; others (incl. PAD) are 0
    d = emb.size(1)  # update D -> D+1

    # 4) Save extended encoder (after adding PAD/UNK + extra dimension)
    os.makedirs("extended_encoder", exist_ok=True)
    torch.save(emb, "extended_encoder/word2vec_extended.pth")
    with open("extended_encoder/vocab_extended.json","w",encoding="utf-8") as f:
        json.dump(vocab, f)
    print("Extended encoder saved in 'extended_encoder'.")

    # 5) Build dataset from ALL books (each line = page), OOV -> UNK
    pre = TextPreProc()
    pages, labels, meta = build_pages(FILES, pre, w2i, unk_idx)
    X = pad_pages(pages, pad_idx)
    y = torch.tensor(labels, dtype=torch.long)
    torch.save(X, "extended_encoder/X_indices.pt")
    torch.save(y, "extended_encoder/y_labels.pt")
    with open("extended_encoder/pages_meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Dataset saved. Pages: {len(pages)}  Max length: {X.size(1)}")

    # 6) Split into train/val/test
    ds = TensorDataset(X, y)
    n = len(ds); n_train = int(0.8*n); n_val = int(0.1*n); n_test = n - n_train - n_val
    tr, va, te = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(123))
    tr_dl = DataLoader(tr, batch_size=64, shuffle=True)
    va_dl = DataLoader(va, batch_size=128)
    te_dl = DataLoader(te, batch_size=128)

    # 7) Model
    num_filters = 100
    kernel_sizes = [3, 4, 5] # change kernel widths / sizes
    model = TextCNN(len(vocab), d, emb, pad_idx, n_f = num_filters, kernels = kernel_sizes, n_classes=len(BOOKS)).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None
    epochs = 20
    val_accuracies = []

    # 8) Training loop
    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0; c = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); c += 1

        # Validation
        model.eval(); v_acc = 0; vc = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_acc += accuracy(logits, yb); vc += 1
        v_acc /= max(1, vc)
        val_accuracies.append(v_acc)
        print(f"Epoch {ep}/{epochs}  Val Acc = {v_acc:.3f}")
        if v_acc > best_val:
            best_val = v_acc
            best_state = model.state_dict()

    if best_state: model.load_state_dict(best_state)

    # 9) Test

    model.eval(); t_acc = 0; tc = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            t_acc += accuracy(logits, yb); tc += 1
    print(f"Test Accuracy = {t_acc/max(1,tc):.3f}")

    torch.save(model.state_dict(), "extended_encoder/textcnn_hp.pt")
    print("Model saved to 'extended_encoder'.")


    # 10) Evaluation artifacts
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=BOOKS))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=BOOKS, yticklabels=BOOKS)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (TextCNN)")
    plt.savefig("extended_encoder/confusion_matrix.png"); plt.clf()

    plt.plot(range(1, epochs+1), val_accuracies, marker='o', label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Validation Accuracy Over Epochs")
    plt.grid(True); plt.legend()
    plt.savefig("extended_encoder/val_accuracy_plot.png")


    # 11) I
    avg_loss, accuracy, top_ngrams, identity_accuracy = filter_info(
        model, 
        te_dl, 
        w2i,
        k = 10,
        batch_size = 64, 
        device = device
    )

    dataframe = visualize_filters(top_ngrams, identity_accuracy, kernel_sizes = [3, 4, 5])
    dataframe = dataframe.sort_values(by = "Identity Accuracy", ascending = False) ;
    top_effective_filters = dataframe.head(10)
    print(top_effective_filters)

    top_effective_filters = top_effective_filters[["Filter", "Top-k N-grams"]].to_numpy()

    for i in range(5):
        print(f'Filter {top_effective_filters[i][0]} \n {top_effective_filters[i][1]} \n')
