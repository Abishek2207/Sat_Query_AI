"""
Data Preprocessing Pipeline for Remote Sensing and Satellite Imagery.

Supports:
- Optical RGB (JPG, PNG, TIFF)
- Multispectral Imagery (e.g. Sentinel-2 / Landsat with NIR, SWIR bands)
- SAR Imagery (Synthetic Aperture Radar / Sentinel-1 with VV/VH backscatter)
- Spectral Index Calculations (NDVI, NDWI, NDBI)
- Invalid/NaN pixel handling and percentile clipping
- Geospatial metadata preservation
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import scipy.ndimage

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class ModalityType(str, Enum):
    """Supported Remote Sensing Modality Types."""
    OPTICAL_RGB = "optical_rgb"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


def detect_modality(num_channels: int, filename: str = "") -> ModalityType:
    """
    Infers satellite imagery modality based on channel count and filename cues.
    
    Args:
        num_channels: Number of image channels (C)
        filename: Optional path or filename for keyword matching
    
    Returns:
        ModalityType enum
    """
    fn_lower = filename.lower()
    if "sar" in fn_lower or "s1" in fn_lower or "vv" in fn_lower or "vh" in fn_lower:
        return ModalityType.SAR
    if num_channels in (1, 2) and ("rgb" not in fn_lower):
        return ModalityType.SAR
    elif num_channels == 3:
        return ModalityType.OPTICAL_RGB
    elif num_channels >= 4:
        return ModalityType.MULTISPECTRAL
    return ModalityType.OPTICAL_RGB


def clean_invalid_pixels(data: np.ndarray, fill_value: Optional[float] = None) -> np.ndarray:
    """
    Replaces NaN, Inf, or NoData values with finite statistics (median or fill_value).
    
    Args:
        data: Numpy array (C, H, W) or (H, W)
        fill_value: Optional specific value to replace NaNs with
    
    Returns:
        Cleaned numpy array with no NaNs or Infs
    """
    data = data.copy()
    invalid_mask = ~np.isfinite(data)
    if not np.any(invalid_mask):
        return data

    if fill_value is not None:
        data[invalid_mask] = fill_value
    else:
        # Fill each channel with its valid median or 0.0
        if data.ndim == 3:
            for c in range(data.shape[0]):
                ch = data[c]
                valid = ch[np.isfinite(ch)]
                med = np.median(valid) if len(valid) > 0 else 0.0
                ch[~np.isfinite(ch)] = med
                data[c] = ch
        else:
            valid = data[np.isfinite(data)]
            med = np.median(valid) if len(valid) > 0 else 0.0
            data[invalid_mask] = med
            
    return data


def compute_spectral_indices(
    data: np.ndarray,
    band_map: Optional[Dict[str, int]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes key remote sensing spectral indices from multi-band imagery.
    
    Standard Indices:
    - NDVI (Normalized Difference Vegetation Index) = (NIR - Red) / (NIR + Red + eps)
    - NDWI (Normalized Difference Water Index)      = (Green - NIR) / (Green + NIR + eps)
    - NDBI (Normalized Difference Built-up Index)  = (SWIR - NIR) / (SWIR + NIR + eps)
    
    Args:
        data: Multi-band array in (C, H, W) format
        band_map: Dict mapping band name ('red', 'green', 'blue', 'nir', 'swir') to channel index
    
    Returns:
        Dictionary containing index summary statistics (mean, max, min) and 2D arrays
    """
    if data.ndim != 3 or data.shape[0] < 4:
        # Default mapping for 3-channel optical: estimate pseudo vegetation index
        return {}
    
    # Default 4+ band mapping (e.g. B, G, R, NIR or R, G, B, NIR)
    if band_map is None:
        # Common standard: 0: Red, 1: Green, 2: Blue, 3: NIR
        band_map = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}
        if data.shape[0] >= 5:
            band_map['swir'] = 4

    indices = {}
    eps = 1e-7

    # Calculate NDVI if Red and NIR are present
    if 'red' in band_map and 'nir' in band_map and data.shape[0] > max(band_map['red'], band_map['nir']):
        red = data[band_map['red']].astype(np.float32)
        nir = data[band_map['nir']].astype(np.float32)
        ndvi = (nir - red) / (nir + red + eps)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        indices['ndvi_mean'] = float(np.mean(ndvi))
        indices['ndvi_max'] = float(np.max(ndvi))
        indices['ndvi_min'] = float(np.min(ndvi))
        indices['ndvi_map'] = ndvi

    # Calculate NDWI if Green and NIR are present
    if 'green' in band_map and 'nir' in band_map and data.shape[0] > max(band_map['green'], band_map['nir']):
        green = data[band_map['green']].astype(np.float32)
        nir = data[band_map['nir']].astype(np.float32)
        ndwi = (green - nir) / (green + nir + eps)
        ndwi = np.clip(ndwi, -1.0, 1.0)
        indices['ndwi_mean'] = float(np.mean(ndwi))
        indices['ndwi_map'] = ndwi

    # Calculate NDBI if SWIR and NIR are present
    if 'swir' in band_map and 'nir' in band_map and data.shape[0] > max(band_map['swir'], band_map['nir']):
        swir = data[band_map['swir']].astype(np.float32)
        nir = data[band_map['nir']].astype(np.float32)
        ndbi = (swir - nir) / (swir + nir + eps)
        ndbi = np.clip(ndbi, -1.0, 1.0)
        indices['ndbi_mean'] = float(np.mean(ndbi))
        indices['ndbi_map'] = ndbi

    return indices


def process_sar_image(
    data: np.ndarray,
    to_db: bool = True,
    apply_speckle_filter: bool = True
) -> np.ndarray:
    """
    Processes Synthetic Aperture Radar (SAR) backscatter imagery.
    
    1. Converts linear amplitude or power to Decibel (dB) scale: 10 * log10(val^2 + eps)
    2. Applies a Lee/median speckle noise reduction filter.
    3. Normalizes SAR intensity into standardized range [0, 1].
    
    Args:
        data: SAR data in (C, H, W) or (H, W) format
        to_db: Whether to convert amplitude to dB scale
        apply_speckle_filter: Whether to apply speckle suppression
    
    Returns:
        Processed SAR array (C, H, W) in float32
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        
    data = data.astype(np.float32)
    eps = 1e-6
    
    # 1. Decibel conversion
    if to_db:
        # Check if values are linear (non-negative, large range)
        if np.min(data) >= 0:
            data = 10.0 * np.log10(np.square(data) + eps)
            
    # 2. Speckle noise filtering (Median filter on each polarization channel)
    if apply_speckle_filter:
        for c in range(data.shape[0]):
            data[c] = scipy.ndimage.median_filter(data[c], size=3)
            
    # 3. Robust min-max normalization via 2nd and 98th percentiles (radar clipping)
    for c in range(data.shape[0]):
        p2, p98 = np.percentile(data[c], (2, 98))
        if p98 > p2:
            data[c] = np.clip((data[c] - p2) / (p98 - p2), 0.0, 1.0)
        else:
            data[c] = np.zeros_like(data[c])
            
    return data


def load_remote_sensing_image(
    file_path: str,
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, ModalityType, Dict[str, Any]]:
    """
    Universal Remote Sensing image loader supporting:
    - Standard image formats (JPG, PNG, BMP)
    - GeoTIFF / TIFF files (.tif, .tiff)
    - Multispectral and SAR satellite products
    
    Args:
        file_path: Path to satellite image file
        target_size: Optional (height, width) to resize image to
        
    Returns:
        Tuple of:
        - data: Numpy array in (C, H, W) float32 format
        - modality: Detected ModalityType
        - metadata: Dictionary containing georeferencing, original dimensions, etc.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Satellite image not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    metadata: Dict[str, Any] = {"file_path": file_path, "format": ext}
    data: Optional[np.ndarray] = None

    # Method 1: Try rasterio (if available) for GeoTIFF metadata
    if ext in ('.tif', '.tiff', '.geotiff') and HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                raw_data = src.read()  # (C, H, W)
                metadata["crs"] = str(src.crs) if src.crs else "Unknown"
                metadata["bounds"] = list(src.bounds) if src.bounds else None
                metadata["transform"] = list(src.transform) if src.transform else None
                metadata["nodata"] = src.nodata
                data = raw_data.astype(np.float32)
        except Exception:
            data = None

    # Method 2: Try tifffile (fast TIFF reader)
    if data is None and ext in ('.tif', '.tiff', '.geotiff') and HAS_TIFFFILE:
        try:
            raw_data = tifffile.imread(file_path)
            # Standardize shape to (C, H, W)
            if raw_data.ndim == 2:
                data = raw_data[np.newaxis, ...].astype(np.float32)
            elif raw_data.ndim == 3:
                if raw_data.shape[2] <= 16 and raw_data.shape[0] > 16:
                    # (H, W, C) -> (C, H, W)
                    data = np.transpose(raw_data, (2, 0, 1)).astype(np.float32)
                else:
                    data = raw_data.astype(np.float32)
        except Exception:
            data = None

    # Method 3: Standard PIL Image loader
    if data is None:
        pil_img = Image.open(file_path)
        metadata["mode"] = pil_img.mode
        arr = np.array(pil_img)
        if arr.ndim == 2:
            data = arr[np.newaxis, ...].astype(np.float32)
        elif arr.ndim == 3:
            # (H, W, C) -> (C, H, W)
            data = np.transpose(arr, (2, 0, 1)).astype(np.float32)
            # Remove alpha channel if 4-channel PNG
            if data.shape[0] == 4 and ext in ('.png', '.jpg', '.jpeg'):
                data = data[:3]

    if data is None:
        raise ValueError(f"Unable to read satellite image format: {file_path}")

    # Clean invalid pixels (NaN, Inf)
    data = clean_invalid_pixels(data)
    
    # Store original dimensions
    c, h, w = data.shape
    metadata["original_shape"] = (c, h, w)
    
    # Detect modality
    modality = detect_modality(num_channels=c, filename=file_path)
    metadata["modality"] = modality.value
    
    # Resize if requested
    if target_size is not None and (h != target_size[0] or w != target_size[1]):
        resized_channels = []
        for i in range(c):
            ch = data[i]
            # Normalize channel to 0-255 temporarily for bilinear interpolation
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                ch_norm = (ch - ch_min) / (ch_max - ch_min) * 255.0
                img_ch = Image.fromarray(ch_norm.astype(np.uint8))
                img_ch_res = img_ch.resize((target_size[1], target_size[0]), Image.BILINEAR)
                ch_res = np.array(img_ch_res).astype(np.float32) / 255.0 * (ch_max - ch_min) + ch_min
            else:
                ch_res = np.zeros(target_size, dtype=np.float32)
            resized_channels.append(ch_res)
        data = np.stack(resized_channels, axis=0)

    return data, modality, metadata


class RemoteSensingPreprocessor:
    """
    Production-ready preprocessor for Remote Sensing Neural Networks.
    Converts diverse satellite data into normalized tensors suitable for PyTorch backbones.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        target_channels: int = 3,
        normalize_method: str = "imagenet"
    ):
        """
        Args:
            image_size: Target (height, width) for model input (default: 224x224)
            target_channels: Number of channels expected by model backbone (default: 3)
            normalize_method: Normalization strategy ('imagenet', 'minmax', 'standard')
        """
        self.image_size = image_size
        self.target_channels = target_channels
        self.normalize_method = normalize_method
        
        # ImageNet RGB statistics
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def preprocess_tensor(
        self,
        data: np.ndarray,
        modality: ModalityType
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalizes and prepares a numpy array for PyTorch model ingestion.
        
        Args:
            data: Raw satellite data array (C, H, W)
            modality: Detected modality
            
        Returns:
            Tuple of (preprocessed_array_float32, extra_info_dict)
        """
        extra_info = {}
        c, h, w = data.shape
        
        # 1. Compute spectral indices if multispectral
        if modality == ModalityType.MULTISPECTRAL and c >= 4:
            extra_info['spectral_indices'] = compute_spectral_indices(data)
            
        # 2. SAR processing
        if modality == ModalityType.SAR:
            data = process_sar_image(data, to_db=True, apply_speckle_filter=True)
            
        # 3. Channel adaptation (e.g. 1-ch SAR -> 3-ch, or 13-band -> 3-ch RGB)
        if c == 1 and self.target_channels == 3:
            data = np.repeat(data, 3, axis=0)
        elif c > self.target_channels:
            # If multi-band and targeting 3 channels, select first 3 (RGB or pseudo-RGB)
            data = data[:self.target_channels]
        elif c < self.target_channels:
            # Pad remaining channels with zeros
            pad = np.zeros((self.target_channels - c, h, w), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        # 4. Normalization
        if modality == ModalityType.OPTICAL_RGB or (data.shape[0] == 3 and self.normalize_method == "imagenet"):
            # Scale to [0, 1] if in [0, 255]
            if data.max() > 1.0:
                data = data / 255.0
            data = np.clip(data, 0.0, 1.0)
            data = (data - self.imagenet_mean) / self.imagenet_std
        else:
            # Robust min-max per channel
            for i in range(data.shape[0]):
                p2, p98 = np.percentile(data[i], (2, 98))
                if p98 > p2:
                    data[i] = np.clip((data[i] - p2) / (p98 - p2), 0.0, 1.0)
                else:
                    data[i] = np.zeros_like(data[i])

        return data.astype(np.float32), extra_info

    def __call__(self, file_path_or_array: Union[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Callable interface to preprocess an image path or numpy array.
        """
        if isinstance(file_path_or_array, str):
            data, modality, meta = load_remote_sensing_image(file_path_or_array, target_size=self.image_size)
        else:
            data = file_path_or_array
            modality = detect_modality(data.shape[0])
            meta = {"original_shape": data.shape}

        processed_data, extra_info = self.preprocess_tensor(data, modality)
        meta.update(extra_info)
        meta["modality"] = modality.value
        return processed_data, meta
