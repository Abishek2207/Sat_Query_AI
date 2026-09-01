"""
Evaluation Runner for Trained Remote Sensing Models.

Evaluates a saved checkpoint on test imagery, computes detailed metrics,
generates a confusion matrix, and saves evaluation reports.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from models.classifier import RemoteSensingClassifier
from models.backbone import auto_select_device
from data.dataset import create_data_loaders, scan_dataset_directory
from evaluation.metrics import calculate_classification_metrics, print_metrics_summary
from training.validate import validate_epoch
from utils.visualization import plot_confusion_matrix


def evaluate_model_checkpoint(
    checkpoint_path: str = "checkpoints/best_model.pth",
    data_dir: Optional[str] = None,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Loads a checkpoint and evaluates performance on the test dataset split.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        data_dir: Path to test dataset (if None, uses dataset path from checkpoint config)
        output_dir: Folder where evaluation plots and JSON reports will be saved
        
    Returns:
        Dictionary of calculated evaluation metrics
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train a model first!")

    os.makedirs(output_dir, exist_ok=True)
    device = auto_select_device()

    print(f"\n[Evaluation] Loading model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names", [])
    backbone_name = checkpoint.get("backbone", "resnet18")
    in_channels = checkpoint.get("in_channels", 3)
    num_classes = checkpoint.get("num_classes", len(class_names))
    config = checkpoint.get("config", {})

    if data_dir is None:
        data_dir = config.get("data", {}).get("data_dir", "data/raw/EuroSAT_Sample")

    print(f"[Evaluation] Test dataset: {data_dir}")
    print(f"[Evaluation] Classes: {class_names}")

    # Build model
    model = RemoteSensingClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=False,
        in_channels=in_channels,
        class_names=class_names
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test loader
    batch_size = config.get("data", {}).get("batch_size", 16)
    image_size = tuple(config.get("data", {}).get("image_size", [224, 224]))

    _, _, test_loader, _ = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        target_channels=in_channels,
        random_seed=42
    )

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, y_true, y_pred, y_probs = validate_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    metrics = calculate_classification_metrics(y_true, y_pred, class_names=class_names)
    metrics["test_loss"] = round(val_loss, 4)

    # Print summary
    print_metrics_summary(metrics)

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] Evaluation metrics saved to: {os.path.abspath(metrics_path)}")

    # Plot confusion matrix
    cm = metrics.get("confusion_matrix", [])
    if cm:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, save_path=cm_path)
        print(f"[Saved] Confusion matrix plot saved to: {os.path.abspath(cm_path)}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Remote Sensing Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save evaluation reports")
    args = parser.parse_args()

    evaluate_model_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
