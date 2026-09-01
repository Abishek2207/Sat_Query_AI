"""
Remote Sensing Preprocessing Module.
Provides image loading, multi-modal data handling (Optical, Multispectral, SAR),
spectral index calculations (NDVI, NDWI, NDBI), and geospatial normalization.
"""

from .preprocess import (
    ModalityType,
    load_remote_sensing_image,
    compute_spectral_indices,
    process_sar_image,
    RemoteSensingPreprocessor,
)
from .augmentation import get_training_augmentations, get_validation_augmentations

__all__ = [
    "ModalityType",
    "load_remote_sensing_image",
    "compute_spectral_indices",
    "process_sar_image",
    "RemoteSensingPreprocessor",
    "get_training_augmentations",
    "get_validation_augmentations",
]
