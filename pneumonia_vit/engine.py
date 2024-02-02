"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    for data, label in tqdm(loader, desc="train", leave=False):
        data, label = data.to(device), label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_accuracy += (output.argmax(dim=1) == label).float().mean().item() / len(loader)
        total_loss += loss.item() / len(loader)
    return EpochMetrics(loss=total_loss, accuracy=total_accuracy)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)
        total_accuracy += (output.argmax(dim=1) == label).float().mean().item() / len(loader)
        total_loss += loss.item() / len(loader)
    return EpochMetrics(loss=total_loss, accuracy=total_accuracy)


@torch.no_grad()
def evaluate_test(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run inference on the test set and report standard classification metrics."""
    model.eval()
    predictions, labels = [], []
    for data, label in tqdm(loader, desc="test", leave=False):
        data = data.to(device)
        output = model(data)
        predictions.extend(output.argmax(dim=1).cpu().numpy())
        labels.extend(label.numpy())

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }
    report = classification_report(labels, predictions)
    return {"metrics": metrics, "report": report}
