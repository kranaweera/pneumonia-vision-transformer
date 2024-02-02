"""pneumonia_vit: an efficient Vision Transformer for pneumonia detection in chest X-rays.

Classifies chest X-ray images as NORMAL or PNEUMONIA using a ViT with a
Linformer (linear-attention) transformer backbone, trained on the Kaggle
"Chest X-Ray Images (Pneumonia)" dataset.
"""

from .model import build_model
from .data import build_dataloaders

__all__ = ["build_model", "build_dataloaders"]

__version__ = "0.1.0"
