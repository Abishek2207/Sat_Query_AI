"""
Remote Sensing Data Module.
Provides PyTorch Dataset and DataLoader loaders for multi-format satellite datasets.
"""

from .dataset import (
    RemoteSensingDataset,
    create_data_loaders,
    DEFAULT_CLASSES,
    CLASS_FEATURE_DESCRIPTIONS,
)

__all__ = [
    "RemoteSensingDataset",
    "create_data_loaders",
    "DEFAULT_CLASSES",
    "CLASS_FEATURE_DESCRIPTIONS",
]
