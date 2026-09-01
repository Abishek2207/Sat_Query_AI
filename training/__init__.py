"""
Remote Sensing Training Module.
Provides training engine, validation loop, learning-rate scheduling, and checkpointing.
"""

from .train import train_model, load_config
from .validate import validate_epoch

__all__ = [
    "train_model",
    "load_config",
    "validate_epoch",
]
