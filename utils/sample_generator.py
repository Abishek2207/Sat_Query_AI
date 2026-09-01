"""
Synthetic Satellite Image Generator for Remote Sensing Starter Workflows.

Generates realistic starter multi-modal satellite samples:
- Optical RGB (.png / .tif)
- Multispectral 4-Band (Red, Green, Blue, Near-Infrared) with proper NDVI signatures
- SAR Radar Backscatter with radar speckle noise and polarization characteristics

Allows a beginner to immediately test and verify training, evaluation, inference,
and visualization without waiting to download gigabytes of raw satellite data.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def create_base_texture(h: int, w: int, roughness: float = 0.5) -> np.ndarray:
    """Generates low-frequency terrain gradient texture using smoothed noise."""
    noise = np.random.randn(h // 4, w // 4)
    # Resize up using simple nearest/bilinear approximation
    img_noise = Image.fromarray((noise - noise.min()) / (noise.max() - noise.min() + 1e-6) * 255.0)
    img_large = img_noise.resize((w, h), Image.BICUBIC)
    return np.array(img_large, dtype=np.float32) / 255.0


def generate_sample_image(
    class_name: str,
    size: Tuple[int, int] = (64, 64),
    modality: str = "optical_rgb"
) -> np.ndarray:
    """
    Synthesizes a realistic satellite image for a given land cover class.
    
    Args:
        class_name: One of the 10 standard LULC classes
        size: (height, width)
        modality: 'optical_rgb', 'multispectral', or 'sar'
        
    Returns:
        Numpy array (C, H, W) in float32 scaled [0, 1]
    """
    h, w = size
    tex = create_base_texture(h, w)
    
    # Base channels: Red, Green, Blue, NIR
    r = np.zeros((h, w), dtype=np.float32)
    g = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)
    nir = np.zeros((h, w), dtype=np.float32)

    if class_name == "AnnualCrop":
        # Plowed fields with stripe patterns (crops in rows)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        stripes = np.sin(grid_x * 0.4 + grid_y * 0.1) * 0.15
        r = 0.55 + stripes + tex * 0.15
        g = 0.45 + stripes * 0.5 + tex * 0.12
        b = 0.25 + tex * 0.08
        nir = 0.65 + tex * 0.20  # Moderate NDVI

    elif class_name == "Forest":
        # Deep green, high NIR reflection
        r = 0.12 + tex * 0.08
        g = 0.38 + tex * 0.15
        b = 0.15 + tex * 0.05
        nir = 0.85 + tex * 0.12  # Very high NIR (strong vegetation)

    elif class_name == "HerbaceousVegetation":
        # Light green/yellow grassland
        r = 0.35 + tex * 0.12
        g = 0.52 + tex * 0.15
        b = 0.22 + tex * 0.08
        nir = 0.60 + tex * 0.15

    elif class_name == "Highway":
        # Background terrain with asphalt roadway
        r = 0.40 + tex * 0.10
        g = 0.45 + tex * 0.10
        b = 0.35 + tex * 0.10
        nir = 0.40 + tex * 0.10
        # Draw diagonal/horizontal road line
        for y in range(h):
            road_x = int(w * 0.5 + np.sin(y * 0.05) * 5)
            w_road = max(2, w // 16)
            for dx in range(-w_road, w_road + 1):
                px = np.clip(road_x + dx, 0, w - 1)
                r[y, px] = 0.22  # dark asphalt
                g[y, px] = 0.22
                b[y, px] = 0.24
                nir[y, px] = 0.15

    elif class_name == "Industrial":
        # Concrete gray with large rectangular rooftops
        r = 0.60 + tex * 0.08
        g = 0.62 + tex * 0.08
        b = 0.65 + tex * 0.08
        nir = 0.30 + tex * 0.05  # Low NIR
        # Draw warehouse blocks
        for bx in [w//6, w//2]:
            for by in [h//6, h//2]:
                bw, bh = w//3 - 2, h//3 - 2
                r[by:by+bh, bx:bx+bw] = 0.80
                g[by:by+bh, bx:bx+bw] = 0.78
                b[by:by+bh, bx:bx+bw] = 0.82
                nir[by:by+bh, bx:bx+bw] = 0.25

    elif class_name == "Pasture":
        # Uniform green pasture
        r = 0.28 + tex * 0.10
        g = 0.58 + tex * 0.12
        b = 0.25 + tex * 0.08
        nir = 0.72 + tex * 0.12

    elif class_name == "PermanentCrop":
        # Grid of orchard dots
        r = 0.30 + tex * 0.10
        g = 0.48 + tex * 0.12
        b = 0.20 + tex * 0.08
        nir = 0.68 + tex * 0.15
        for gx in range(4, w - 4, max(4, w // 8)):
            for gy in range(4, h - 4, max(4, h // 8)):
                r[gy-1:gy+2, gx-1:gx+2] = 0.15
                g[gy-1:gy+2, gx-1:gx+2] = 0.60
                nir[gy-1:gy+2, gx-1:gx+2] = 0.85

    elif class_name == "Residential":
        # Suburban houses and streets
        r = 0.45 + tex * 0.10
        g = 0.45 + tex * 0.10
        b = 0.42 + tex * 0.10
        nir = 0.35 + tex * 0.08
        # Small house roofs (red, blue, gray)
        roof_colors = [(0.75, 0.25, 0.20), (0.25, 0.40, 0.70), (0.70, 0.70, 0.70)]
        for rx in range(6, w - 8, max(8, w // 4)):
            for ry in range(6, h - 8, max(8, h // 4)):
                rc = roof_colors[(rx + ry) % len(roof_colors)]
                r[ry:ry+4, rx:rx+4] = rc[0]
                g[ry:ry+4, rx:rx+4] = rc[1]
                b[ry:ry+4, rx:rx+4] = rc[2]
                nir[ry:ry+4, rx:rx+4] = 0.20

    elif class_name == "River":
        # Green land with winding blue river
        r = 0.25 + tex * 0.12
        g = 0.48 + tex * 0.15
        b = 0.20 + tex * 0.08
        nir = 0.65 + tex * 0.15
        for y in range(h):
            center_x = int(w * 0.45 + np.sin(y * 0.1) * 8)
            rw = max(3, w // 10)
            for dx in range(-rw, rw + 1):
                px = np.clip(center_x + dx, 0, w - 1)
                r[y, px] = 0.08
                g[y, px] = 0.30
                b[y, px] = 0.65
                nir[y, px] = 0.02  # Water absorbs NIR strongly!

    else:  # SeaLake
        # Deep blue water, uniform or wave ripples
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        waves = np.sin(grid_x * 0.2 + grid_y * 0.1) * 0.04
        r = 0.06 + waves
        g = 0.22 + waves + tex * 0.04
        b = 0.58 + waves + tex * 0.05
        nir = 0.02 + tex * 0.01  # Very low NIR

    # Clip channels
    r = np.clip(r, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    nir = np.clip(nir, 0.0, 1.0)

    if modality == "sar":
        # SAR intensity: rough surface = high backscatter, smooth water = low backscatter + speckle noise
        intensity = (r * 0.3 + g * 0.4 + b * 0.3)
        speckle = np.random.gamma(shape=2.0, scale=0.5, size=(h, w)).astype(np.float32)
        sar_backscatter = np.clip(intensity * speckle, 0.0, 1.5)
        return sar_backscatter[np.newaxis, ...]  # (1, H, W)
    elif modality == "multispectral":
        return np.stack([r, g, b, nir], axis=0)  # (4, H, W)
    else:
        return np.stack([r, g, b], axis=0)  # (3, H, W)


def generate_synthetic_dataset(
    output_dir: str,
    num_samples_per_class: int = 10,
    size: Tuple[int, int] = (64, 64),
    include_geotiff: bool = True
) -> str:
    """
    Generates a starter dataset structure for quick local training and testing.
    
    Args:
        output_dir: Root directory where dataset will be created
        num_samples_per_class: Number of sample images per class
        size: Image (height, width)
        include_geotiff: Whether to save sample GeoTIFF files alongside PNGs
        
    Returns:
        Path to generated dataset root
    """
    classes = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for c_name in classes:
        c_dir = os.path.join(output_dir, c_name)
        os.makedirs(c_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            # Generate RGB image
            rgb_arr = generate_sample_image(c_name, size=size, modality="optical_rgb")
            rgb_uint8 = (np.transpose(rgb_arr, (1, 2, 0)) * 255.0).astype(np.uint8)
            img_pil = Image.fromarray(rgb_uint8)
            
            # Save PNG
            png_path = os.path.join(c_dir, f"{c_name}_{i+1:03d}.png")
            img_pil.save(png_path)
            
            # Save multi-band TIFF for a subset
            if include_geotiff and i == 0 and HAS_TIFFFILE:
                multi_arr = generate_sample_image(c_name, size=size, modality="multispectral")
                tif_path = os.path.join(c_dir, f"{c_name}_{i+1:03d}_multispectral.tif")
                tifffile.imwrite(tif_path, (multi_arr * 10000).astype(np.uint16))
                
    # Also create a few standalone test samples in data/samples
    sample_dir = os.path.join(os.path.dirname(output_dir), "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1. Optical test
    opt_img = generate_sample_image("Residential", size=(128, 128), modality="optical_rgb")
    Image.fromarray((np.transpose(opt_img, (1, 2, 0)) * 255.0).astype(np.uint8)).save(
        os.path.join(sample_dir, "sample_optical_urban.png")
    )
    
    # 2. Multispectral test
    multi_img = generate_sample_image("Forest", size=(128, 128), modality="multispectral")
    if HAS_TIFFFILE:
        tifffile.imwrite(
            os.path.join(sample_dir, "sample_multispectral_forest.tif"),
            (multi_img * 10000).astype(np.uint16)
        )
        
    # 3. SAR test
    sar_img = generate_sample_image("River", size=(128, 128), modality="sar")
    if HAS_TIFFFILE:
        tifffile.imwrite(
            os.path.join(sample_dir, "sample_sar_river.tif"),
            (sar_img[0] * 255.0).astype(np.uint8)
        )
    else:
        Image.fromarray((sar_img[0] * 255.0).astype(np.uint8)).save(
            os.path.join(sample_dir, "sample_sar_river.png")
        )
        
    return output_dir
