"""
backend/adaptation/train.py

Entry point for BigEarthNet-based remote-sensing VQA adaptation.

REQUIREMENTS BEFORE RUNNING:
1. The BigEarthNet parquet file must exist at the configured path.
2. The actual satellite image files must exist and be paired with patch_id.
3. A cloud GPU (e.g., Google Colab A100) is strongly recommended.
4. peft, transformers, and torch must be installed on the training machine.

IMPORTANT:
- This script does NOT run on the local Snapdragon X host.
- Do NOT set remote_sensing_adapted=True without a completed checkpoint.
- The provenance.json written here is the ONLY valid source of adaptation truth.
"""
import os
import json
import time
from pathlib import Path
from .config import AdaptationConfig


def run_adaptation(config: AdaptationConfig) -> bool:
    print(f"[train.py] Starting adaptation pipeline.")
    print(f"  Dataset  : {config.dataset_path}")
    print(f"  Model    : {config.base_model}")
    print(f"  Method   : {config.peft_method}")
    print(f"  Epochs   : {config.epochs}")

    if not Path(config.dataset_path).exists():
        print(f"[train.py] BLOCKED: Dataset not found at {config.dataset_path}")
        print("[train.py] Cannot proceed. real training requires real data.")
        return False

    # Import heavy ML deps only when actually running
    try:
        import torch
        from transformers import Trainer, TrainingArguments
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError as e:
        print(f"[train.py] Missing dependency: {e}")
        print("[train.py] Install: pip install peft transformers torch")
        return False

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (not recommended)"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"  GPU      : {gpu}")
    print(f"  VRAM     : {vram:.1f} GB")

    start = time.time()

    # ----- REAL TRAINING WOULD GO HERE -----
    # trainer = Trainer(...)
    # trainer.train()
    # trainer.save_model(config.output_dir)
    # ----------------------------------------

    print("[train.py] Real training not yet implemented in this phase.")
    print("[train.py] Configure the Trainer and resume.")
    return False


def save_provenance(config: AdaptationConfig, steps: int, loss: float, runtime: float):
    """
    Saves a provenance JSON that the inference layer reads to set
    remote_sensing_adapted=True. Only called after successful training.
    """
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    provenance = {
        "model": config.base_model,
        "model_version": "v1.0-bigearthnet-adapted",
        "adaptation_dataset": config.dataset_name,
        "adaptation_method": config.peft_method,
        "remote_sensing_adapted": True,
        "training_steps": steps,
        "training_loss": loss,
        "training_runtime_seconds": runtime,
    }

    out_path = out / "provenance.json"
    with open(out_path, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"[train.py] Provenance saved to {out_path}")


if __name__ == "__main__":
    cfg = AdaptationConfig()
    run_adaptation(cfg)
