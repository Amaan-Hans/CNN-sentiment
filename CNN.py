import os, json, string, random, argparse, time, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns



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

    def forward(self, x):
        # x: [B, L] -> emb: [B, L, D] -> transpose to [B, D, L] for Conv1d
        x = self.emb(x).transpose(1, 2)
        feats = [torch.max(self.act(conv(x)), dim=2).values for conv in self.convs]  # global max pool
        h = self.drop(torch.cat(feats, dim=1))
        return self.fc(h)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

# ---------- Main ----------
if __name__ == "__main__":
    # NEW: arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", type=str, default="3,4,5",
                    help="comma-separated kernel sizes, e.g. 3,4,5 or 4,5,7")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--nf", type=int, default=100, help="filters per kernel")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--runs_csv", type=str,
                    default="extended_encoder/runs/sweep_results.csv")
    args = ap.parse_args()

    kernels = tuple(int(k) for k in args.kernels.split(","))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BOOKS = ["HP1","HP2","HP3","HP4","HP5","HP6","HP7"]
    FILES = [f'./harry_potter_books/{b}.txt' for b in BOOKS]

    print("Loading base encoder (HP1)...")
    emb = torch.load("word2vec_embeddings.pth", map_location=torch.device('cpu')).cpu()
    with open("vocab.json","r",encoding="utf-8") as f:
        vocab = json.load(f)
    w2i = {w:i for i,w in enumerate(vocab)}
    d = emb.size(1)

    PAD, UNK = "<pad>", "<unk>"
    if PAD not in w2i:
        w2i[PAD] = len(w2i); vocab.append(PAD)
        emb = torch.cat([emb, torch.zeros(1, d, device=emb.device)], dim=0)
    if UNK not in w2i:
        w2i[UNK] = len(w2i); vocab.append(UNK)
        emb = torch.cat([emb, torch.zeros(1, d, device=emb.device)], dim=0)
    pad_idx = w2i[PAD]; unk_idx = w2i[UNK]

    # extend embedding dim by 1 and mark UNK
    emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device)], dim=1)
    emb[unk_idx, -1] = 1.0
    d = emb.size(1)

    os.makedirs("extended_encoder", exist_ok=True)
    torch.save(emb, "extended_encoder/word2vec_extended.pth")
    with open("extended_encoder/vocab_extended.json","w",encoding="utf-8") as f:
        json.dump(vocab, f)

    pre = TextPreProc()
    pages, labels, meta = build_pages(FILES, pre, w2i, unk_idx)
    X = pad_pages(pages, pad_idx)
    y = torch.tensor(labels, dtype=torch.long)
    torch.save(X, "extended_encoder/X_indices.pt")
    torch.save(y, "extended_encoder/y_labels.pt")
    with open("extended_encoder/pages_meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Dataset saved. Pages: {len(pages)}  Max length: {X.size(1)}")

    ds = TensorDataset(X, y)
    n = len(ds); n_train = int(0.8*n); n_val = int(0.1*n); n_test = n - n_train - n_val
    tr, va, te = random_split(ds, [n_train, n_val, n_test],
                              generator=torch.Generator().manual_seed(args.seed))
    tr_dl = DataLoader(tr, batch_size=64, shuffle=True)
    va_dl = DataLoader(va, batch_size=128)
    te_dl = DataLoader(te, batch_size=128)

    # NEW: use args.nf and args.dropout, freeze emb (per your request)
    model = TextCNN(len(vocab), d, emb, pad_idx,
                    n_classes=len(BOOKS),
                    n_f=args.nf, kernels=kernels,
                    drop=args.dropout, freeze=True).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state, val_accuracies = 0.0, None, []
    epochs = args.epochs
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()

        model.eval(); v_acc = 0.0; vc = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                v_acc += accuracy(model(xb), yb); vc += 1
        v_acc /= max(1, vc)
        val_accuracies.append(v_acc)
        print(f"Epoch {ep}/{epochs}  Val Acc = {v_acc:.3f}")
        if v_acc > best_val:
            best_val = v_acc
            best_state = model.state_dict()

    if best_state: model.load_state_dict(best_state)
    train_time_s = time.time() - t0

    # Test
    model.eval(); t_acc = 0.0; tc = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            t_acc += accuracy(logits, yb); tc += 1
    test_acc = t_acc/max(1,tc)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    torch.save(model.state_dict(), "extended_encoder/textcnn_hp.pt")

    # Artifacts (unchanged)
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

    # NEW: append a CSV row for this run
    os.makedirs(os.path.dirname(args.runs_csv), exist_ok=True)
    header = ["kernels","nf","dropout","seed","best_val_acc","test_acc","macro_f1","train_time_s"]
    row = [list(kernels), args.nf, args.dropout, args.seed,
           round(best_val,4), round(test_acc,4), round(macro_f1,4), round(train_time_s,2)]
    new_file = not os.path.exists(args.runs_csv)
    with open(args.runs_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file: w.writerow(header)
        w.writerow(row)

    # Print test acc for launcher parsing
    print(f"[RUN DONE] kernels={kernels} nf={args.nf} drop={args.dropout} "
          f"seed={args.seed} | test_acc={test_acc:.4f}")