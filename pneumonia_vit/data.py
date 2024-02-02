"""Dataset loading for the Kaggle "Chest X-Ray Images (Pneumonia)" layout.

Expects ``data_dir`` to contain ``train/``, ``val/`` and ``test/``
sub-directories, each with one folder per class (``NORMAL``, ``PNEUMONIA``).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SPLITS = ("train", "val", "test")


def _transforms(image_size: int) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return {
        "train": transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        "val": transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]),
        "test": transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]),
    }


def build_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Build train/val/test :class:`DataLoader`s from an ``ImageFolder`` layout."""
    split_transforms = _transforms(image_size)
    datasets_by_split = {
        split: datasets.ImageFolder(f"{data_dir}/{split}", transform=split_transforms[split])
        for split in SPLITS
    }
    for split, ds in datasets_by_split.items():
        print(f"{split}: {len(ds)} images, classes={ds.classes}")

    return {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers)
        for split, ds in datasets_by_split.items()
    }
