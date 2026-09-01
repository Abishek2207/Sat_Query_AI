"""
Sat Query AI — Vercel Serverless API
=====================================
FastAPI-based REST API for the Remote Sensing AI module.
Deployed on Vercel as a Python serverless function.

Endpoints:
  GET  /              → Health check + API info
  GET  /classes       → List of supported LULC classes
  POST /predict       → Classify an uploaded satellite image
  POST /indices       → Compute NDVI/NDWI for a multispectral image
"""

import os
import io
import sys
import json
import base64
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Vercel serverless: add project root to path so modules resolve correctly
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sat Query AI — Remote Sensing API",
    description="AI-powered satellite & remote sensing image analysis. Supports Optical RGB, Multispectral GeoTIFF, and SAR imagery.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LULC class taxonomy
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

CHECKPOINT_PATH = str(ROOT / "checkpoints" / "best_model.pth")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check and API overview."""
    return {
        "service": "Sat Query AI — Remote Sensing API",
        "status": "online",
        "version": "1.0.0",
        "description": "AI-powered Land Use / Land Cover classification for satellite imagery",
        "modalities_supported": ["Optical RGB (PNG/JPG)", "Multispectral GeoTIFF (4-band)", "SAR Radar (single-band TIF)"],
        "classes": CLASSES,
        "endpoints": {
            "docs":    "/docs",
            "predict": "POST /predict  — Upload satellite image, get LULC prediction",
            "indices": "POST /indices  — Upload multispectral image, get NDVI/NDWI",
            "classes": "GET  /classes  — List all supported land cover classes"
        }
    }


@app.get("/classes", tags=["Info"])
def get_classes():
    """Returns the list of 10 supported LULC classes."""
    return {
        "total": len(CLASSES),
        "classes": CLASSES,
        "taxonomy": "EuroSAT Land Use / Land Cover",
        "descriptions": {
            "AnnualCrop":            "Annually harvested crop fields (wheat, maize, etc.)",
            "Forest":                "Dense tree canopy, mixed/deciduous/coniferous forest",
            "HerbaceousVegetation":  "Non-agricultural grassland, meadows, shrubs",
            "Highway":               "Major road infrastructure and transportation networks",
            "Industrial":            "Factories, warehouses, industrial zones",
            "Pasture":               "Grazing land and managed grass fields",
            "PermanentCrop":         "Orchards, vineyards, perennial plantations",
            "Residential":           "Urban housing, suburbs, built-up residential areas",
            "River":                 "Flowing water bodies, streams, canals",
            "SeaLake":               "Standing water: lakes, reservoirs, coastal sea areas"
        }
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a satellite image into one of 10 LULC classes.

    **Accepts:** PNG, JPG, GeoTIFF (.tif)
    **Returns:** Predicted class, confidence, top-3 predictions, spectral indices, features
    """
    # Validate file type
    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"
        )

    # Check model checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not found. Train the model first using: python main.py train"
        )

    # Save upload to temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from inference.predict import predict_image
        result = predict_image(
            image_path=tmp_path,
            checkpoint_path=CHECKPOINT_PATH,
            visualize=False
        )
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            **result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/indices", tags=["Inference"])
async def compute_indices(file: UploadFile = File(...)):
    """
    Compute spectral indices (NDVI, NDWI, NDBI) from a multispectral GeoTIFF.

    **Accepts:** GeoTIFF with at least 4 bands (R, G, B, NIR)
    **Returns:** NDVI mean/max/min, NDWI mean, NDBI mean, vegetation health assessment
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Spectral indices require a GeoTIFF (.tif) multispectral image.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from preprocessing.preprocess import load_image_universal
        import numpy as np

        img_array, modality = load_image_universal(tmp_path)

        if modality != "MULTISPECTRAL" or img_array.shape[0] < 4:
            raise HTTPException(
                status_code=400,
                detail=f"Image has {img_array.shape[0]} band(s). Need ≥4 bands (R,G,B,NIR) for spectral indices."
            )

        # Bands: 0=R, 1=G, 2=B, 3=NIR
        R   = img_array[0].astype(float)
        G   = img_array[1].astype(float)
        NIR = img_array[3].astype(float)

        eps = 1e-8
        ndvi = (NIR - R)  / (NIR + R  + eps)
        ndwi = (G   - NIR) / (G   + NIR + eps)
        ndbi = (R   - NIR) / (R   + NIR + eps)  # simplified proxy

        def vegetation_health(ndvi_mean: float) -> str:
            if ndvi_mean > 0.6:  return "Dense healthy vegetation 🌿"
            if ndvi_mean > 0.3:  return "Moderate vegetation cover 🌱"
            if ndvi_mean > 0.1:  return "Sparse vegetation / mixed land"
            return "Bare soil / built surface / water"

        return {
            "status": "success",
            "filename": file.filename,
            "modality": modality,
            "bands": int(img_array.shape[0]),
            "spectral_indices": {
                "NDVI": {
                    "mean": round(float(ndvi.mean()), 4),
                    "max":  round(float(ndvi.max()),  4),
                    "min":  round(float(ndvi.min()),  4),
                    "description": "Normalized Difference Vegetation Index (higher = more vegetation)"
                },
                "NDWI": {
                    "mean": round(float(ndwi.mean()), 4),
                    "description": "Normalized Difference Water Index (positive = water bodies)"
                },
                "NDBI": {
                    "mean": round(float(ndbi.mean()), 4),
                    "description": "Normalized Difference Built-up Index (positive = urban/built surface)"
                }
            },
            "vegetation_health": vegetation_health(float(ndvi.mean()))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index computation error: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Vercel ASGI handler
# ---------------------------------------------------------------------------
handler = Mangum(app)
