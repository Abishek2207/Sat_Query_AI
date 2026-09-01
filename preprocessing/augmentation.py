"""
Remote Sensing Specific Data Augmentations.

Satellite and aerial imagery have rotational invariance (no natural 'up' or 'down' orientation).
This module provides spatial augmentations (D4 symmetry group: flips + 90-degree rotations)
and radiometric jitter suited for remote sensing and Earth observation tasks.
"""

import random
import torch
import numpy as np


class RandomDihedralRotation:
    """
    Randomly applies one of the 8 D4 dihedral group symmetries:
    - 4 orthogonal rotations (0, 90, 180, 270 degrees)
    - Combined with optional horizontal reflection
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is shape (C, H, W)
        k = random.randint(0, 3)  # 0, 90, 180, 270 deg
        if k > 0:
            tensor = torch.rot90(tensor, k, [1, 2])
        if random.random() > 0.5:
            tensor = torch.flip(tensor, [2])  # horizontal flip
        if random.random() > 0.5:
            tensor = torch.flip(tensor, [1])  # vertical flip
        return tensor


class RandomRadiometricJitter:
    """
    Randomly modifies brightness and contrast across satellite channels
    to simulate varying atmospheric conditions and sun-angle illumination.
    """
    def __init__(self, brightness: float = 0.15, contrast: float = 0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Brightness shift
        if self.brightness > 0 and random.random() > 0.5:
            alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
            tensor = tensor * alpha
            
        # Contrast adjustment
        if self.contrast > 0 and random.random() > 0.5:
            mean = tensor.mean(dim=[-2, -1], keepdim=True)
            beta = 1.0 + random.uniform(-self.contrast, self.contrast)
            tensor = (tensor - mean) * beta + mean
            
        return tensor


class ComposeTransforms:
    """Composes several transforms together sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            tensor = t(tensor)
        return tensor


def get_training_augmentations(apply_radiometric: bool = True) -> ComposeTransforms:
    """
    Returns standard remote sensing augmentation pipeline for model training.
    """
    transforms = [RandomDihedralRotation()]
    if apply_radiometric:
        transforms.append(RandomRadiometricJitter(brightness=0.1, contrast=0.1))
    return ComposeTransforms(transforms)


def get_validation_augmentations() -> ComposeTransforms:
    """
    Validation / test transform pipeline (no random augmentations).
    """
    return ComposeTransforms([])
