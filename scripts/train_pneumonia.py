#!/usr/bin/env python
"""Train the efficient ViT pneumonia classifier.

Example:

    python -m scripts.train_pneumonia --data-dir chest_xray --epochs 20
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from pneumonia_vit.config import build_arg_parser, config_from_args
from pneumonia_vit.data import build_dataloaders
from pneumonia_vit.engine import evaluate_epoch, evaluate_test, train_one_epoch
from pneumonia_vit.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = build_dataloaders(cfg.data_dir, cfg.image_size, cfg.batch_size, cfg.num_workers)

    model = build_model(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        linformer_k=cfg.linformer_k,
        num_classes=cfg.num_classes,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.lr_gamma)

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{cfg.epochs}: "
              f"loss={train_metrics.loss:.4f} acc={train_metrics.accuracy:.4f} "
              f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}")

    os.makedirs(os.path.dirname(cfg.checkpoint) or ".", exist_ok=True)
    torch.save(model.state_dict(), cfg.checkpoint)
    print(f"Saved model weights to {cfg.checkpoint}")

    print("\nEvaluating on the test set ...")
    results = evaluate_test(model, loaders["test"], device)
    for name, value in results["metrics"].items():
        print(f"Test {name}: {value:.4f}")
    print("\nClassification report:")
    print(results["report"])


if __name__ == "__main__":
    main()
