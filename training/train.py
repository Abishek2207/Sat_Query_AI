"""
Training Pipeline for Remote Sensing Image Classifier.

Features:
- Configurable hyperparameters via config.yaml or CLI arguments
- Automatic GPU (CUDA/MPS) / CPU hardware selection
- Checkpoint management (best_model.pth, last_model.pth)
- Learning rate schedulers (Cosine, Step, Plateau)
- Early stopping mechanism
- Reproducible random seed control
"""

import os
import sys
import time
import json
import yaml
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from models.classifier import RemoteSensingClassifier
from models.backbone import auto_select_device, get_model_summary
from data.dataset import create_data_loaders, DEFAULT_CLASSES
from training.validate import validate_epoch


def load_config(config_path: str = "training/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        # Fallback default configuration
        return {
            "data": {"data_dir": "data/raw/EuroSAT_Sample", "image_size": [224, 224], "target_channels": 3, "batch_size": 16, "num_workers": 0, "split_ratio": [0.7, 0.15, 0.15], "random_seed": 42},
            "model": {"backbone": "resnet18", "pretrained": True, "num_classes": 10, "dropout_rate": 0.3, "hidden_dim": 256},
            "training": {"epochs": 15, "learning_rate": 0.0005, "weight_decay": 0.0001, "lr_scheduler": "cosine", "early_stopping_patience": 5, "save_dir": "checkpoints", "use_mixed_precision": True}
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "training/config.yaml",
    override_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main training routine.
    
    Args:
        config: Direct dictionary of settings (optional)
        config_path: Path to config.yaml file
        override_args: Optional dict to override specific parameters
        
    Returns:
        Dictionary with training history and best metrics
    """
    if config is None:
        config = load_config(config_path)

    # Apply any CLI overrides
    if override_args:
        for k, v in override_args.items():
            if v is not None:
                if k in ["data_dir", "batch_size", "epochs", "learning_rate", "backbone", "save_dir"]:
                    if k in ["data_dir", "batch_size"]:
                        config["data"][k] = v
                    elif k in ["backbone"]:
                        config["model"][k] = v
                    elif k in ["epochs", "learning_rate", "save_dir"]:
                        config["training"][k] = v

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    # Set seeds for reproducibility
    seed = data_cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Hardware Selection
    device = auto_select_device()

    # 2. Data Loaders
    data_dir = data_cfg.get("data_dir", "data/raw/EuroSAT_Sample")
    print(f"\n[Data] Loading remote sensing dataset from: {data_dir}")
    
    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    target_channels = data_cfg.get("target_channels", 3)
    batch_size = data_cfg.get("batch_size", 16)
    split_ratio = tuple(data_cfg.get("split_ratio", [0.70, 0.15, 0.15]))
    num_workers = data_cfg.get("num_workers", 0)

    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        split_ratio=split_ratio,
        image_size=image_size,
        target_channels=target_channels,
        num_workers=num_workers,
        random_seed=seed
    )

    num_classes = len(class_names)
    print(f"[Data] Found {len(train_loader.dataset)} training, {len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples.")
    print(f"[Data] Classes ({num_classes}): {class_names}")

    # 3. Model Creation
    backbone_name = model_cfg.get("backbone", "resnet18")
    pretrained = model_cfg.get("pretrained", True)
    dropout_rate = model_cfg.get("dropout_rate", 0.3)
    hidden_dim = model_cfg.get("hidden_dim", 256)

    print(f"\n[Model] Building Remote Sensing Classifier with '{backbone_name}' backbone...")
    model = RemoteSensingClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained,
        in_channels=target_channels,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        class_names=class_names
    ).to(device)

    print(f"[Model] {get_model_summary(model)}")

    # 4. Optimization Setup
    lr = float(train_cfg.get("learning_rate", 0.0005))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    epochs = int(train_cfg.get("epochs", 15))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    save_dir = train_cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR Scheduler
    scheduler_type = train_cfg.get("lr_scheduler", "cosine")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.2)
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    else:
        scheduler = None

    # Mixed Precision Scaler
    use_amp = train_cfg.get("use_mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # 5. Training Loop
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [], "epoch_times": []
    }

    best_val_acc = -1.0
    patience_counter = 0

    print(f"\n[Training] Starting {epochs} epochs of training (Early Stopping Patience: {patience})...\n")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'LR':<10} | {'Time':<8}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / max(train_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)

        # Validation
        epoch_val_loss, epoch_val_acc, _, _, _ = validate_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        epoch_time = time.time() - start_time

        # Record History
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)
        history["epoch_times"].append(epoch_time)

        print(
            f"{epoch:<8} | "
            f"{epoch_train_loss:<12.4f} | "
            f"{epoch_train_acc * 100:<9.2f}% | "
            f"{epoch_val_loss:<10.4f} | "
            f"{epoch_val_acc * 100:<9.2f}% | "
            f"{current_lr:<10.6f} | "
            f"{epoch_time:<6.1f}s"
        )

        # Save Checkpoints
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": epoch_val_acc,
            "val_loss": epoch_val_loss,
            "class_names": class_names,
            "config": config,
            "backbone": backbone_name,
            "in_channels": target_channels,
            "num_classes": num_classes
        }

        # Save Last
        torch.save(checkpoint_data, os.path.join(save_dir, "last_model.pth"))

        # Save Best
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(checkpoint_data, os.path.join(save_dir, "best_model.pth"))
            print(f"  --> Saved new BEST model checkpoint (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered after {patience} epochs without improvement.")
                break

    # Save training history to JSON
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Complete] Training finished! Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"[Checkpoints] Saved in: '{os.path.abspath(save_dir)}'")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Remote Sensing Classifier")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone model (resnet18, custom_cnn, etc.)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint output folder")
    args = parser.parse_args()

    overrides = {
        "data_dir": args.data_dir,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "save_dir": args.save_dir
    }

    train_model(config_path=args.config, override_args=overrides)
