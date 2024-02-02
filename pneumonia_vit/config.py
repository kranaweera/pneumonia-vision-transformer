"""Configuration and CLI argument parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "chest_xray"
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 3e-5
    lr_gamma: float = 0.7
    seed: int = 42
    num_workers: int = 0
    checkpoint: str = "checkpoints/pneumonia_vit.pt"

    # ViT / Linformer architecture.
    dim: int = 128
    patch_size: int = 32
    depth: int = 12
    heads: int = 8
    linformer_k: int = 64
    num_classes: int = 2


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train an efficient ViT to classify pneumonia in chest X-rays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=defaults.data_dir,
                        help="directory containing train/val/test sub-folders of class-named folders")
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--lr-gamma", type=float, default=defaults.lr_gamma,
                        help="multiplicative LR decay applied every epoch")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--checkpoint", default=defaults.checkpoint,
                        help="path to save the trained model weights to")
    parser.add_argument("--dim", type=int, default=defaults.dim)
    parser.add_argument("--patch-size", type=int, default=defaults.patch_size)
    parser.add_argument("--depth", type=int, default=defaults.depth)
    parser.add_argument("--heads", type=int, default=defaults.heads)
    parser.add_argument("--linformer-k", type=int, default=defaults.linformer_k)
    parser.add_argument("--num-classes", type=int, default=defaults.num_classes)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_gamma=args.lr_gamma,
        seed=args.seed,
        num_workers=args.num_workers,
        checkpoint=args.checkpoint,
        dim=args.dim,
        patch_size=args.patch_size,
        depth=args.depth,
        heads=args.heads,
        linformer_k=args.linformer_k,
        num_classes=args.num_classes,
    )
