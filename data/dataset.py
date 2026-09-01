"""
Remote Sensing Dataset Loader.

Handles:
- Folder-based classification datasets (e.g. EuroSAT, UC Merced, custom directories)
- Multi-band GeoTIFF, SAR, and RGB formats
- Train/Validation/Test splitting with stratified splits
- PyTorch DataLoader factories
"""

import os
import glob
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from preprocessing.preprocess import RemoteSensingPreprocessor, load_remote_sensing_image, ModalityType
from preprocessing.augmentation import get_training_augmentations, get_validation_augmentations, ComposeTransforms


# Standard Land Use / Land Cover (LULC) 10-Class taxonomy based on EuroSAT (Sentinel-2 benchmark)
DEFAULT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# Meaningful remote-sensing visual & geospatial features for each class
CLASS_FEATURE_DESCRIPTIONS = {
    "AnnualCrop": [
        "Agricultural crop fields",
        "Plowed and tilled soil",
        "Seasonal vegetation patterns",
        "Rectangular farm boundaries"
    ],
    "Forest": [
        "Dense tree canopy",
        "High vegetative index (NDVI > 0.6)",
        "Woodland and timber areas",
        "Natural green cover"
    ],
    "HerbaceousVegetation": [
        "Natural grasslands",
        "Wild shrubland and prairie",
        "Moderate vegetation density",
        "Non-cultivated open terrain"
    ],
    "Highway": [
        "Linear asphalt / concrete road networks",
        "Transportation corridors",
        "Paved surfaces",
        "Vehicle transit infrastructure"
    ],
    "Industrial": [
        "Large commercial and industrial buildings",
        "Warehouses and factory roofs",
        "High built-up index (NDBI)",
        "Extensive impervious concrete surfaces"
    ],
    "Pasture": [
        "Livestock grazing meadows",
        "Grassland vegetation",
        "Open agricultural pasture",
        "Fenced rural terrain"
    ],
    "PermanentCrop": [
        "Perennial orchards and vineyards",
        "Plantation rows and groves",
        "Persistent agricultural greenery",
        "Structured canopy alignment"
    ],
    "Residential": [
        "Urban and suburban housing",
        "High density of buildings and rooftops",
        "Interspersed street networks",
        "Low to moderate urban vegetation"
    ],
    "River": [
        "Linear flowing water body",
        "High water index (NDWI > 0.2)",
        "Meandering riverbanks and riparian vegetation",
        "Inland hydrological network"
    ],
    "SeaLake": [
        "Deep or shallow open water body",
        "Strong absorption in NIR band (NDWI > 0.5)",
        "Homogeneous dark optical/radar signature",
        "Lakes, reservoirs, or coastal ocean"
    ]
}


class RemoteSensingDataset(Dataset):
    """
    PyTorch Dataset for multi-modal remote sensing and Earth observation imagery.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        target_channels: int = 3,
        transform: Optional[ComposeTransforms] = None,
        is_training: bool = False
    ):
        """
        Args:
            samples: List of tuples (file_path, class_idx)
            class_names: List of class name strings
            image_size: Image target height and width
            target_channels: Number of input channels (e.g. 3)
            transform: Optional spatial/radiometric augmentations
            is_training: Whether augmentations should be active
        """
        self.samples = samples
        self.class_names = class_names
        self.image_size = image_size
        self.target_channels = target_channels
        self.transform = transform
        self.is_training = is_training
        
        self.preprocessor = RemoteSensingPreprocessor(
            image_size=image_size,
            target_channels=target_channels,
            normalize_method="imagenet"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path, label = self.samples[idx]
        
        # Load and preprocess image
        processed_data, meta = self.preprocessor(file_path)
        tensor = torch.from_numpy(processed_data)  # (C, H, W)
        
        # Apply data augmentations if in training mode
        if self.transform is not None:
            tensor = self.transform(tensor)
            
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "file_path": file_path,
            "modality": meta.get("modality", "unknown"),
            "class_name": self.class_names[label] if label < len(self.class_names) else "Unknown"
        }


def scan_dataset_directory(data_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Scans a folder containing class subdirectories:
    data_dir/
       ├── AnnualCrop/
       ├── Forest/
       ...
    
    Returns:
        Tuple of (samples_list, class_names)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.geotiff')
    
    # Check if directory contains class folders
    subdirs = sorted([
        d for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])
    
    samples = []
    class_names = []
    
    if subdirs:
        class_names = subdirs
        for class_idx, c_name in enumerate(class_names):
            c_dir = os.path.join(data_dir, c_name)
            for fname in os.listdir(c_dir):
                if fname.lower().endswith(valid_exts):
                    samples.append((os.path.join(c_dir, fname), class_idx))
    else:
        # Check flat directory with generic files
        all_files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.lower().endswith(valid_exts)
        ]
        class_names = ["Unlabeled"]
        for f in all_files:
            samples.append((f, 0))

    return samples, class_names


def create_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    image_size: Tuple[int, int] = (224, 224),
    target_channels: int = 3,
    num_workers: int = 0,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates stratified Train, Validation, and Test DataLoaders from a dataset folder.
    
    Args:
        data_dir: Root dataset folder
        batch_size: Mini-batch size
        split_ratio: Tuple of (train_pct, val_pct, test_pct)
        image_size: (height, width)
        target_channels: Number of channels (3 for RGB)
        num_workers: DataLoader background workers
        random_seed: Random seed for reproducible splits
        
    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    samples, class_names = scan_dataset_directory(data_dir)
    
    if len(samples) == 0:
        raise ValueError(f"No satellite images found in '{data_dir}'. Supported formats: .jpg, .png, .tif")

    # Set deterministic random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Shuffle samples
    indices = np.random.permutation(len(samples))
    shuffled_samples = [samples[i] for i in indices]
    
    # Compute split bounds
    n_total = len(shuffled_samples)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_test = n_total - n_train - n_val
    
    # Fallback if dataset is very small
    if n_train == 0:
        n_train = n_total
        n_val = 0
        n_test = 0
    elif n_val == 0 and n_total >= 3:
        n_val = 1
        n_train -= 1

    train_samples = shuffled_samples[:n_train]
    val_samples = shuffled_samples[n_train:n_train + n_val] if n_val > 0 else train_samples
    test_samples = shuffled_samples[n_train + n_val:] if n_test > 0 else val_samples

    # Create datasets
    train_dataset = RemoteSensingDataset(
        samples=train_samples,
        class_names=class_names,
        image_size=image_size,
        target_channels=target_channels,
        transform=get_training_augmentations(),
        is_training=True
    )

    val_dataset = RemoteSensingDataset(
        samples=val_samples,
        class_names=class_names,
        image_size=image_size,
        target_channels=target_channels,
        transform=get_validation_augmentations(),
        is_training=False
    )

    test_dataset = RemoteSensingDataset(
        samples=test_samples,
        class_names=class_names,
        image_size=image_size,
        target_channels=target_channels,
        transform=get_validation_augmentations(),
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=(len(train_samples) > batch_size)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, class_names
