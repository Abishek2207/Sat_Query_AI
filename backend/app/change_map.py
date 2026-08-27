"""
backend/app/change_map.py

Deterministic baseline for bi-temporal change analysis.

IMPORTANT:
- This is a pixel-differencing baseline ONLY.
- It does NOT use a learned model.
- Results are labeled as BASELINE to prevent false scientific claims.
- Accuracy is NOT guaranteed beyond structural pixel comparison.
"""
import io
import base64
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from PIL import Image
from typing import Dict, Any


def compute_change_baseline(
    img_t1_bytes: bytes,
    img_t2_bytes: bytes,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Performs deterministic pixel-differencing change analysis.
    Returns a structured result including a base64 PNG change mask.
    Does NOT use a trained model. Not suitable for scientific accuracy.
    """
    try:
        with MemoryFile(img_t1_bytes) as mem1, MemoryFile(img_t2_bytes) as mem2:
            with mem1.open() as src1, mem2.open() as src2:

                # 1. Spatial alignment check
                if src1.width != src2.width or src1.height != src2.height:
                    return {
                        "status": "INVALID_INPUT",
                        "answer": "Spatial dimensions are incompatible. Images must have the same width and height.",
                        "evidence": [],
                    }

                # 2. CRS compatibility warning (not a hard block for baseline)
                crs_match = src1.crs == src2.crs
                crs_note = "CRS match confirmed." if crs_match else "WARNING: CRS mismatch detected. Baseline may be unreliable."

                # 3. Read first band, normalise to [0,1]
                data1 = src1.read(1).astype(np.float32)
                data2 = src2.read(1).astype(np.float32)

                max1 = data1.max() if data1.max() > 0 else 1.0
                max2 = data2.max() if data2.max() > 0 else 1.0
                data1 /= max1
                data2 /= max2

                # 4. Pixel-level difference and threshold
                diff = np.abs(data2 - data1)
                mask = (diff > threshold).astype(np.uint8) * 255

                change_pixel_count = int(np.sum(mask > 0))
                total_pixels = mask.size
                change_percent = (change_pixel_count / total_pixels) * 100

                # 5. Generate PNG visualisation of the change mask
                mask_img = Image.fromarray(mask, mode='L').convert('RGB')
                buf = io.BytesIO()
                mask_img.save(buf, format='PNG')
                mask_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                # 6. Build plain-language answer
                if change_percent < 1.0:
                    description = "No significant change detected between the two images."
                elif change_percent < 10.0:
                    description = f"Minor change detected in approximately {change_percent:.1f}% of the image area."
                elif change_percent < 40.0:
                    description = f"Moderate change detected in {change_percent:.1f}% of the image area."
                else:
                    description = f"Significant change detected across {change_percent:.1f}% of the image area."

                return {
                    "status": "SUCCESS",
                    "answer": description,
                    "confidence": None,   # Baseline only — not from a trained model
                    "evidence": [
                        f"Baseline pixel-differencing applied (threshold={threshold}).",
                        f"Changed pixels: {change_pixel_count:,} of {total_pixels:,} ({change_percent:.1f}%).",
                        crs_note,
                        "BASELINE: This result is from a deterministic algorithm, not a learned model.",
                    ],
                    "visual_output": f"data:image/png;base64,{mask_b64}",
                    "provenance": {
                        "model": "Deterministic_PixelDiff_Baseline",
                        "model_version": "1.0",
                        "adaptation_dataset": None,
                        "adaptation_method": None,
                        "remote_sensing_adapted": False,
                        "inference_timestamp": "",  # filled by caller
                        "device": device if "device" in globals() or "device" in locals() else "cpu", "input_filenames": [],       # filled by caller
                        "input_modalities": ["unknown", "unknown"],
                        "crs": str(src1.crs) if src1.crs else None,
                        "geospatial_evidence_generated": True,
                    },
                }

    except Exception as exc:
        return {
            "status": "ERROR",
            "answer": f"Change map computation failed: {exc}",
            "evidence": [],
        }
