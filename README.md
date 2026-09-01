# 🛰️ Remote Sensing & Satellite Image Analysis AI Module

A modular, beginner-friendly, production-ready Deep Learning framework for Earth Observation (EO) and Satellite Imagery Analysis built with **PyTorch**.

---

## 🌟 Overview & Capabilities

This module analyzes satellite and aerial remote-sensing imagery to perform automated **Land Use & Land Cover (LULC)** classification, feature reasoning, spectral analysis, and pixel segmentation.

### 🛰️ Supported Imagery Modalities:
1. **Optical Satellite Imagery (RGB)**: Standard aerial/satellite imagery (JPG, PNG, TIFF) captured by Sentinel-2, Landsat, PlanetScope, or drones.
2. **Multispectral Imagery**: Multi-band imagery (GeoTIFF) including Red, Green, Blue, Near-Infrared (NIR), and Shortwave-Infrared (SWIR) bands. Automatically calculates:
   - **NDVI** (Normalized Difference Vegetation Index): $\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$ (Identifies dense forest vs sparse crops)
   - **NDWI** (Normalized Difference Water Index): $\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$ (Delineates rivers, lakes, reservoirs)
   - **NDBI** (Normalized Difference Built-up Index): $\frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}$ (Pinpoints urban and industrial zones)
3. **SAR (Synthetic Aperture Radar)**: Radar imagery (e.g. Sentinel-1 VV/VH). Converts raw amplitude to calibrated Decibels (dB: $10 \log_{10}(\text{amplitude}^2)$), applies speckle suppression filtering, and evaluates surface roughness.

### 🏷️ 10 Standard Land Use / Land Cover (LULC) Classes:
- **Annual Crop**: Cultivated farmland, seasonal crops, plowed soil
- **Forest**: Dense woodlands, tree canopy, natural green cover
- **Herbaceous Vegetation**: Grasslands, prairies, shrubland
- **Highway**: Roads, expressways, paved transportation corridors
- **Industrial**: Warehouses, factory complexes, high built-up surface
- **Pasture**: Livestock grazing meadows, agricultural grass
- **Permanent Crop**: Orchards, vineyards, plantation groves
- **Residential**: Urban and suburban housing, building clusters, streets
- **River**: Flowing inland waterways and riparian corridors
- **Sea / Lake**: Open water bodies, oceans, lakes, reservoirs

---

## 📁 Project Architecture

```
remote_sensing_model/
│
├── data/
│   ├── raw/                       # Place raw downloaded datasets (EuroSAT, UC Merced)
│   ├── processed/                 # Partitioned or preprocessed datasets
│   ├── samples/                   # Standalone test images (Optical, GeoTIFF, SAR)
│   └── dataset.py                 # Dataset loader with stratified Train/Val/Test splits
│
├── models/
│   ├── backbone.py                # Transfer learning backbones + multi-channel adapters
│   ├── classifier.py              # LULC Classifier with explainable feature reasoning
│   └── segmentation.py            # U-Net architecture for pixel-level land cover mapping
│
├── preprocessing/
│   ├── preprocess.py              # Multi-modal loader, GeoTIFF parser, NDVI/NDWI, SAR dB
│   └── augmentation.py            # D4 Dihedral group rotations (rot90, flips) & radiometric jitter
│
├── training/
│   ├── train.py                   # Training loop, early stopping, LR schedulers, GPU/CPU auto-select
│   ├── validate.py                # Validation and loss calculation loop
│   └── config.yaml                # Clean YAML configuration file for all hyperparameters
│
├── evaluation/
│   ├── metrics.py                 # Accuracy, Precision, Recall, F1, Confusion Matrix, IoU, Dice
│   └── evaluate.py                # Standalone evaluation runner with publication-ready plots
│
├── inference/
│   └── predict.py                 # Production API `predict_image()` and CLI inference tool
│
├── utils/
│   ├── visualization.py           # Multi-panel satellite plots, confusion matrices, mask overlays
│   └── sample_generator.py        # Synthetic multi-modal satellite image generator
│
├── requirements.txt               # Dependencies
├── README.md                      # Documentation & guides
└── main.py                        # Unified CLI entrypoint (demo, train, evaluate, predict)
```

---

## ⚡ Quick Start Installation

### Step 1: Clone or Navigate to Project Directory
```powershell
cd remote_sensing_model
```

### Step 2: Install Dependencies

#### For CPU (Standard Laptop / Desktop):
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow matplotlib pyyaml scikit-learn scipy tifffile
```

#### For NVIDIA GPU (CUDA Acceleration):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow matplotlib pyyaml scikit-learn scipy tifffile
```

---

## 🚀 One-Command End-to-End Demo

If you are a beginner and want to test the entire system right away without downloading gigabytes of satellite data, run:

```bash
python main.py demo
```

This single command will:
1. Generate realistic multi-modal satellite samples (RGB, GeoTIFF, SAR).
2. Train a lightweight Remote Sensing model.
3. Evaluate test accuracy and save the Confusion Matrix plot.
4. Run inference on Optical, Multispectral, and SAR satellite images, outputting structured JSON and multi-panel inspection plots.

---

## 📊 Public Dataset Setup Guide

### Recommended Dataset: **EuroSAT (Sentinel-2 Satellite Dataset)**
EuroSAT is the benchmark dataset for Earth observation consisting of 27,000 Sentinel-2 satellite images across 10 LULC classes.

#### Option A: EuroSAT RGB (Recommended for quick start, ~90 MB)
1. Download EuroSAT RGB from: [EuroSAT Dataset (RGB)](https://github.com/phelber/eurosat) or [Kaggle EuroSAT](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
2. Extract the archive into:
   ```
   remote_sensing_model/data/raw/EuroSAT/
      ├── AnnualCrop/
      ├── Forest/
      ├── HerbaceousVegetation/
      ├── Highway/
      ├── Industrial/
      ├── Pasture/
      ├── PermanentCrop/
      ├── Residential/
      ├── River/
      └── SeaLake/
   ```

#### Option B: EuroSAT Multispectral (13-band GeoTIFF, ~2 GB)
1. Download from [EuroSAT All Bands (GeoTIFF)](https://github.com/phelber/eurosat)
2. Extract into `data/raw/EuroSAT_MS/`.

---

## 🛠️ Step-by-Step Usage

### 1. Generating Starter Sample Imagery
```bash
python main.py create-samples --count 20
```

### 2. Training the Model
You can configure training via `training/config.yaml` or directly from the CLI:

```bash
# Train with ResNet-18 backbone for 15 epochs:
python main.py train --data_dir data/raw/EuroSAT_Sample --backbone resnet18 --epochs 15 --batch_size 16

# Train lightweight model on CPU:
python main.py train --backbone custom_cnn --epochs 5 --batch_size 16
```

### 3. Evaluating Model Performance
Evaluates the trained model, prints accuracy, precision, recall, and generates a confusion matrix heatmap:

```bash
python main.py evaluate --checkpoint checkpoints/best_model.pth --output_dir reports
```

### 4. Running Inference on Single Satellite Images
```bash
# Analyze an optical satellite image
python main.py predict --image data/samples/sample_optical_urban.png --visualize

# Analyze a multispectral GeoTIFF image
python main.py predict --image data/samples/sample_multispectral_forest.tif --visualize

# Analyze a SAR radar image
python main.py predict --image data/samples/sample_sar_river.tif --visualize
```

---

## 🔌 API Integration into Larger AI Systems

You can integrate this remote sensing module into any Python application, Flask/FastAPI server, or geospatial workflow:

```python
from inference.predict import predict_image

# Run analysis on any satellite image (.tif, .png, .jpg)
result = predict_image("path/to/satellite_image.tif", visualize=True)

print("Predicted Class:", result["prediction"])
print(f"Confidence: {result['confidence'] * 100:.1f}%")
print("Detected Features:", result["features"])
print("Explanation:", result["explanation"])
```

### Example Structured Output:
```json
{
  "prediction": "Residential",
  "confidence": 0.9421,
  "features": [
    "Urban and suburban housing",
    "High density of buildings and rooftops",
    "Interspersed street networks",
    "Low to moderate urban vegetation"
  ],
  "top_k": [
    {"class": "Residential", "confidence": 0.9421},
    {"class": "Industrial", "confidence": 0.0382},
    {"class": "Highway", "confidence": 0.0125}
  ],
  "modality": "optical_rgb",
  "spatial_info": {
    "crs": "EPSG:32632",
    "bounds": [450000.0, 5200000.0, 452000.0, 5202000.0],
    "original_shape": [3, 256, 256]
  },
  "spectral_indices": {
    "ndvi_mean": 0.215,
    "ndwi_mean": -0.180
  },
  "explanation": "The image was identified as 'Residential' with 94.2% confidence. Measured Mean NDVI is 0.215, indicating low/sparse vegetation. Key identified signatures include: Urban and suburban housing, High density of buildings and rooftops, Interspersed street networks."
}
```

---

## 🧠 Model Architectures Supported

| Model Backbone | Best Used For | Memory Footprint | Transfer Learning |
| :--- | :--- | :--- | :--- |
| `resnet18` | Standard satellite classification & rapid training | Very Low (~45 MB) | Yes (ImageNet) |
| `resnet50` | High-accuracy complex feature extraction | Medium (~100 MB) | Yes (ImageNet) |
| `mobilenet_v3_small` | Edge devices / Raspberry Pi / Low CPU | Ultra Low (~15 MB) | Yes (ImageNet) |
| `efficientnet_b0` | Optimal accuracy-to-compute ratio | Low (~30 MB) | Yes (ImageNet) |
| `custom_cnn` | Fast testing with zero external weight downloads | Minimal (~5 MB) | Scratch |
| `RemoteSensingUNet` | Pixel-level land-cover and water-body mask segmentation | Low (~35 MB) | Scratch |

---

## 🔬 Remote Sensing Physics & Indices Reference

- **NDVI (Vegetation Index)**: High values ($>0.5$) indicate healthy chlorophyllic green vegetation, moderate values ($0.2-0.4$) indicate shrubs/crops, low/negative values indicate water, urban concrete, or bare rock.
- **NDWI (Water Index)**: Water bodies reflect green light and strongly absorb NIR wavelengths. Positive values ($>0.0$) delineate water bodies like rivers, lakes, and oceans.
- **SAR Radar Backscatter**: Radar actively transmits microwave pulses. Rough surfaces and urban buildings cause high corner reflection (bright backscatter), while calm water mirrors radar pulses away (dark backscatter).
