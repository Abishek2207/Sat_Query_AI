"""
Validation Loop for Remote Sensing Model Training.
Computes validation loss, top-1 accuracy, and collects predictions.
"""

from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int], List[List[float]]]:
    """
    Evaluates model across a validation or test DataLoader.
    
    Args:
        model: PyTorch Remote Sensing model
        data_loader: DataLoader for validation/test data
        criterion: Loss function (e.g. CrossEntropyLoss)
        device: Target compute device (CPU or GPU)
        
    Returns:
        Tuple of (average_loss, accuracy, y_true, y_pred, y_probs)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy, all_targets, all_preds, all_probs
