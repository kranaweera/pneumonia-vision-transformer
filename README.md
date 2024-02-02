# Pneumonia Vision Transformer

An efficient Vision Transformer (ViT with Linformer linear attention) for
classifying chest X-rays as **NORMAL** or **PNEUMONIA**, trained on the Kaggle
"Chest X-Ray Images (Pneumonia)" dataset.

Companion code for the results summarized in
[`docs/vision_transformer_results_pneumonia.pdf`](docs/vision_transformer_results_pneumonia.pdf).

## Approach

Standard patch embeddings feed a transformer whose quadratic self-attention is
replaced by [Linformer](https://arxiv.org/abs/2006.04768)'s linear-complexity
attention, making full-resolution (224x224) attention tractable. The model is
trained end-to-end with cross-entropy loss and evaluated with accuracy,
precision, recall and F1.

## Layout

```
pneumonia_vit/
├── config.py   # CLI args + TrainConfig
├── data.py     # ImageFolder-based train/val/test loaders + augmentations
├── model.py    # ViT + Linformer model builder
└── engine.py   # train/eval loops and test-set metrics
scripts/
└── train_pneumonia.py   # training entry point
data/           # dataset index + instructions (images are not committed)
docs/           # results write-up
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # or: pip install -e .
```

## Data

Images are not included — see [`data/README.md`](data/README.md) for the
expected layout and where to download the dataset.

## Usage

```bash
python -m scripts.train_pneumonia \
    --data-dir chest_xray \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 3e-5
```

Run `python -m scripts.train_pneumonia --help` for the full list of
hyper-parameters (patch size, transformer depth/heads, Linformer projection
dimension `k`, etc). The script prints per-epoch train/validation loss and
accuracy, saves the trained weights to `--checkpoint`, then reports test-set
accuracy, precision, recall, F1 and a full classification report.

## License

MIT — see [LICENSE](LICENSE).
