# CNN-sentiment

This repository trains a small TextCNN to classify pages from the seven Harry Potter books by which book they came from. It demonstrates using pretrained word vectors (from `word2vec_embeddings.pth`) as an embedding initializer and a lightweight convolutional architecture for document/page-level text classification.

# Report

[View the report](.\CNN-sentiment\report.pdf)



**Why this matters**
- **Transfer learning:** shows how to initialise embeddings from pretrained vectors and extend them for unknown tokens.
- **Efficiency:** TextCNN is fast to train and effective for short passage classification.
- **Reproducibility:** includes the raw book text and serialized embedding/model artifacts to reproduce experiments.

**Repository contents**
- `nbc.py`: main training script that builds a dataset from `harry_potter_books/`, extends the provided embeddings, trains a `TextCNN`, and writes artifacts to `extended_encoder/`.
- `vocab.json`: vocabulary aligned with `word2vec_embeddings.pth`.
- `word2vec_embeddings.pth`: pretrained embedding matrix (base encoder loaded and extended by the script).
- `harry_potter_books/`: raw text files used to build the page-level dataset (each line is treated as one page).

What the script does (high level)
- Loads pretrained embeddings and vocabulary, adds special tokens (`<pad>`, `<unk>`), and creates an extended embedding matrix.
- Builds a page-level dataset by reading each book file line-by-line, mapping words to indices and padding to a common length.
- Splits data into train/val/test, trains a `TextCNN` classifier, saves the best model and diagnostic artifacts (`confusion_matrix.png`, `val_accuracy_plot.png`, `textcnn_hp.pt`).

Requirements
- Python 3.8+ (Windows tested)
- The script uses the following packages: `torch`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

Quick start (CPU)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install packages (CPU example):

```powershell
pip install numpy scikit-learn matplotlib seaborn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

3. Run training with defaults:

```powershell
python nbc.py
```

Optional arguments
- `--kernels`: comma-separated kernel sizes (default `3,4,5`)
- `--dropout`: dropout rate (default `0.1`)
- `--nf`: filters per kernel (default `100`)
- `--epochs`: number of epochs (default `30`)
- `--seed`: random seed (default `123`)
- `--runs_csv`: CSV file to append run metadata/results (default `extended_encoder/runs/sweep_results.csv`)

Outputs
- `extended_encoder/word2vec_extended.pth` and `extended_encoder/vocab_extended.json` — extended embeddings + vocab
- `extended_encoder/X_indices.pt`, `extended_encoder/y_labels.pt`, `extended_encoder/pages_meta.json` — dataset indices and metadata
- `extended_encoder/textcnn_hp.pt` — saved best model weights
- `extended_encoder/confusion_matrix.png`, `extended_encoder/val_accuracy_plot.png` — diagnostics
- `extended_encoder/runs/sweep_results.csv` — appended run metrics for experiments
