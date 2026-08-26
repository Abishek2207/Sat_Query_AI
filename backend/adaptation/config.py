"""
backend/adaptation/config.py

Configuration for BigEarthNet-based remote-sensing adaptation.

IMPORTANT:
- remote_sensing_adapted must remain False until actual training completes.
- adaptation_status:
    "not_configured" → no dataset/model paths set
    "configured"     → paths exist, training not yet run
    "adapted"        → training completed, checkpoint saved
"""
from pydantic import BaseModel
from typing import List, Optional


class AdaptationConfig(BaseModel):
    # Dataset
    dataset_name: str = "BigEarthNet-VQA"
    dataset_path: str = "datasets/bigearthnet/BigEarthNet.txt.parquet"
    dataset_split_column: str = "split"

    # Base model to adapt
    base_model: str = "Salesforce/blip-vqa-base"
    peft_method: str = "LoRA"

    # Training hyperparameters
    epochs: int = 5
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_samples: int = 10000

    # Modalities to train on
    image_modalities: List[str] = ["optical"]
    text_annotations: bool = True

    # Output
    output_dir: str = "models/rs_adapted_vqa"
    checkpoint_name: str = "best_checkpoint"

    # Status — DO NOT SET TRUE MANUALLY
    adaptation_status: str = "not_configured"   # not_configured | configured | adapted
    remote_sensing_adapted: bool = False         # Only True after training completes
