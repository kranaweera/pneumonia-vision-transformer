"""Efficient Vision Transformer (ViT + Linformer attention) for binary classification."""

from __future__ import annotations

import torch
from linformer import Linformer
from vit_pytorch.efficient import ViT


def build_model(
    image_size: int = 224,
    patch_size: int = 32,
    dim: int = 128,
    depth: int = 12,
    heads: int = 8,
    linformer_k: int = 64,
    num_classes: int = 2,
    channels: int = 3,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Build a ViT whose self-attention is replaced by Linformer's linear attention."""
    num_patches = (image_size // patch_size) ** 2
    efficient_transformer = Linformer(
        dim=dim,
        seq_len=num_patches + 1,  # patches + 1 cls token
        depth=depth,
        heads=heads,
        k=linformer_k,
    )
    model = ViT(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        transformer=efficient_transformer,
        channels=channels,
    )
    return model.to(device)
