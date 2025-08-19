import os, json, time, string, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# ---------- Preprocessing ----------
class TextPreProc:
    def _rm_punct(self, s):
        return s.translate(str.maketrans('', '', string.punctuation + "“”‘’—…"))
    def process(self, text):
        words = self._rm_punct(text).lower().split()
        vocab = list(dict.fromkeys(words))
        w2i = {w:i for i,w in enumerate(vocab)}
        i2w = {i:w for i,w in enumerate(vocab)}
        return words, vocab, w2i, i2w
    def skipgram_pairs(self, words, w2i, window=2):
        pairs = []
        for i in range(window, len(words)-window):
            c = w2i.get(words[i]); 
            ctx = words[i-window:i] + words[i+1:i+window+1]
            for w in ctx:
                if c is not None and w in w2i: pairs.append((c, w2i[w]))
        return pairs

# ---------- Skip-Gram ----------
class SkipGram(nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.out = nn.Linear(d, vocab_size)
    def forward(self, x):
        return self.out(self.emb(x))

class SGDataset(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): 
        c, t = self.pairs[i]
        return torch.tensor(c), torch.tensor(t)

def train_skipgram(pairs, vocab_size, d=100, lr=0.05, batch=1024, epochs=10, device="cpu"):
    model = SkipGram(vocab_size, d).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(SGDataset(pairs), batch_size=batch, shuffle=True, drop_last=True)
    for e in range(epochs):
        tot = 0
        for c, t in loader:
            c, t = c.to(device), t.to(device)
            loss = loss_fn(model(c), t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[SG] Epoch {e+1:02d} loss={tot/max(1,len(loader)):.4f}")
    return model

# ---------- Page dataset ----------
def build_pages(file_paths, pre, w2i, page_len=250):
    pages, labels, meta = [], [], []
    for book_id, fp in enumerate(file_paths):
        with open(fp, 'r', encoding='utf-8') as f:
            w = pre._rm_punct(f.read()).lower().split()
        count_before = len(pages)
        for i in range(0, len(w), page_len):
            idxs = [w2i[x] for x in w[i:i+page_len] if x in w2i]
            if idxs:
                pages.append(idxs); labels.append(book_id); meta.append({"book": os.path.basename(fp), "len": len(idxs)})
        print(f"{os.path.basename(fp)} pages: {len(pages)-count_before}")
    return pages, labels, meta

def pad_pages(pages, pad_idx, L=250):
    X = torch.full((len(pages), L), pad_idx, dtype=torch.long)
    for i, p in enumerate(pages):
        Lc = min(L, len(p)); 
        if Lc: X[i,:Lc] = torch.tensor(p[:Lc], dtype=torch.long)
    return X

# ---------- TextCNN ----------
class TextCNN(nn.Module):
    def __init__(self, num_emb, d, emb_weight, pad_idx, n_classes=7, n_f=100, kernels=(3,4,5), drop=0.5, freeze=True):
        super().__init__()
        self.emb = nn.Embedding(num_emb, d, padding_idx=pad_idx)
        if emb_weight is not None:
            with torch.no_grad():
                sz0, sz1 = min(num_emb, emb_weight.size(0)), min(d, emb_weight.size(1))
                self.emb.weight[:sz0,:sz1].copy_(emb_weight[:sz0,:sz1])
        self.emb.weight.requires_grad = not freeze
        self.convs = nn.ModuleList([nn.Conv1d(d, n_f, k) for k in kernels])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(n_f * len(kernels), n_classes)
    def forward(self, x):
        x = self.emb(x).transpose(1,2)               # [B,D,L]
        feats = [torch.max(self.act(conv(x)), dim=2).values for conv in self.convs]
        h = self.drop(torch.cat(feats, dim=1))
        return self.fc(h)

def accuracy(logits, y): 
    return (logits.argmax(1) == y).float().mean().item()

# ---------- Main ----------
if __name__ == "__main__":
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BOOKS = ["HP1","HP2","HP3","HP4","HP5","HP6","HP7"]
    FILES = [f'./harry_potter_books/{b}.txt' for b in BOOKS]

    # Load corpus
    print("Loading corpus...")
    corpus = " ".join(open(f, encoding="utf-8").read() for f in FILES)

    # Preprocess
    pre = TextPreProc()
    words, vocab, w2i, i2w = pre.process(corpus)
    print(f"Vocab size: {len(vocab):,}  Tokens: {len(words):,}")

    # Train Skip-Gram on full corpus
    pairs = pre.skipgram_pairs(words, w2i, window=2)
    print(f"Skip-gram pairs: {len(pairs):,}")
    sg_model = train_skipgram(pairs, len(vocab), d=100, lr=0.05, batch=1024, epochs=10, device=device)
    emb = sg_model.emb.weight.data.detach().cpu()   # [V,D]
    d = emb.size(1)

    # Add <pad> and extend embedding
    PAD = "<pad>"
    if PAD not in w2i:
        w2i[PAD] = len(w2i); vocab.append(PAD)
        emb = torch.cat([emb, torch.zeros(1, d)], dim=0)

    # Save encoder artifacts
    torch.save(emb, "word2vec_embeddings.pth")
    with open("vocab.json","w",encoding="utf-8") as f: json.dump(vocab, f)
    print("Saved encoder: vocab.json, word2vec_embeddings.pth")

    # Build page dataset
    pages, labels, meta = build_pages(FILES, pre, w2i, page_len=250)
    X = pad_pages(pages, w2i[PAD], L=250)
    y = torch.tensor(labels, dtype=torch.long)
    torch.save(X, "X_indices.pt"); torch.save(y, "y_labels.pt")
    with open("pages_meta.json","w",encoding="utf-8") as f: json.dump(meta, f)
    print("Saved dataset: X_indices.pt, y_labels.pt, pages_meta.json")

    # Split
    ds = TensorDataset(X, y)
    n = len(ds); n_train = int(0.8*n); n_val = int(0.1*n); n_test = n - n_train - n_val
    tr, va, te = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(123))
    tr_dl = DataLoader(tr, batch_size=128, shuffle=True)
    va_dl = DataLoader(va, batch_size=256)
    te_dl = DataLoader(te, batch_size=256)

    # TextCNN
    num_emb = len(vocab); pad_idx = w2i[PAD]
    model = TextCNN(num_emb, d, emb, pad_idx, n_classes=7, n_f=100, kernels=(3,4,5), drop=0.5, freeze=True).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None
    epochs = 20
    train_losses, val_losses, val_accuracies = [], [], []

    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0; c = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); c += 1
        train_losses.append(tr_loss / max(1, c))

        model.eval(); v_acc = 0; v_loss = 0; vc = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += loss_fn(logits, yb).item()
                v_acc += accuracy(logits, yb); vc += 1
        val_losses.append(v_loss / max(1, vc))
        val_accuracies.append(v_acc / max(1, vc))


    if best_state: model.load_state_dict(best_state)

    # Test
    model.eval(); t_acc = 0; tc = 0
    with torch.no_grad():
        for xb, yb in te_dl:
            xb, yb = xb.to(device), yb.to(device)
            t_acc += accuracy(model(xb), yb); tc += 1
    print(f"[CNN] Test acc={t_acc/max(1,tc):.3f}")

    torch.save(model.state_dict(), "textcnn_hp.pt")
    print("Saved: textcnn_hp.pt")

    import matplotlib.pyplot as plt

    plt.plot(range(1, epochs+1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_accuracy_plot.png")
    plt.clf()

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    all_preds, all_labels = [], []
    
    model.eval()
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=BOOKS, yticklabels=BOOKS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (TextCNN)")
    plt.savefig("confusion_matrix.png")
    plt.clf()

