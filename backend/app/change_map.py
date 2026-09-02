import io
import base64
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from PIL import Image
from typing import Dict, Any

def simple_dilate(mask):
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out

def compute_change_baseline(
    img_t1_bytes: bytes,
    img_t2_bytes: bytes,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    try:
        with MemoryFile(img_t1_bytes) as mem1, MemoryFile(img_t2_bytes) as mem2:
            with mem1.open() as src1, mem2.open() as src2:
                if src1.width != src2.width or src1.height != src2.height:
                    return {"status": "INVALID_INPUT", "answer": "Spatial dimensions mismatch.", "evidence": []}

                crs_match = src1.crs == src2.crs
                crs_note = "CRS match confirmed." if crs_match else "WARNING: CRS mismatch."

                data1 = src1.read(1).astype(np.float32)
                data2 = src2.read(1).astype(np.float32)
                
                # Cloud/No-data masking (assuming 0 is NoData for Sentinel-2 here)
                valid_mask = (data1 > 0) & (data2 > 0)
                
                # Radiometric normalization
                p99_1 = np.percentile(data1[valid_mask], 99) if np.any(valid_mask) else 1.0
                p99_2 = np.percentile(data2[valid_mask], 99) if np.any(valid_mask) else 1.0
                
                # Avoid div by zero
                p99_1 = max(p99_1, 1.0)
                p99_2 = max(p99_2, 1.0)

                d1_norm = np.clip(data1 / p99_1, 0, 1)
                d2_norm = np.clip(data2 / p99_2, 0, 1)

                diff = np.abs(d2_norm - d1_norm)
                # Apply valid mask
                diff[~valid_mask] = 0
                
                # Simple morphological dilation for connected components (simulated structure)
                raw_change = diff > threshold
                clean_change = simple_dilate(raw_change)
                
                mask = clean_change.astype(np.uint8) * 255

                change_pixel_count = int(np.sum(clean_change))
                valid_pixels = int(np.sum(valid_mask))
                
                if valid_pixels > 0:
                    change_percent = (change_pixel_count / valid_pixels) * 100
                else:
                    change_percent = 0.0

                mask_img = Image.fromarray(mask, mode='L').convert('RGB')
                buf = io.BytesIO()
                mask_img.save(buf, format='PNG')
                mask_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                if change_percent < 1.0:
                    description = "No significant change detected between the two images."
                elif change_percent < 10.0:
                    description = f"Minor change detected in approximately {change_percent:.1f}% of the valid image area."
                elif change_percent < 40.0:
                    description = f"Moderate change detected in {change_percent:.1f}% of the valid image area."
                else:
                    description = f"Significant change detected across {change_percent:.1f}% of the valid image area."

                model_name = "Real_BiTemporal_Change_Baseline"
                ev = {
                    "claim": f"Change detected in {change_percent:.1f}% of the area.",
                    "evidence": f"Radiometrically normalized pixel-differencing applied. Total valid pixels: {valid_pixels}. {crs_note}",
                    "region": None, "timestamp": None, "modality": "optical_imagery",
                    "confidence": 1.0, "confidence_type": "rule-based",
                    "status": "VERIFIED", "source": "Change Map Specialist",
                    "model": model_name, "model_version": "1.1"
                }

                return {
                    "status": "SUCCESS",
                    "answer": description,
                    "confidence": None,
                    "evidence": [ev],
                    "visual_output": f"data:image/png;base64,{mask_b64}",
                    "provenance": {
                        "model": model_name,
                        "model_version": "1.1",
                        "adaptation_dataset": None,
                        "adaptation_method": None,
                        "remote_sensing_adapted": False,
                        "inference_timestamp": "", 
                        "device": "cpu",
                        "input_filenames": [], 
                        "input_modalities": ["optical", "optical"],
                        "crs": str(src1.crs) if src1.crs else None,
                        "geospatial_evidence_generated": True,
                    },
                }

    except Exception as exc:
        return {"status": "ERROR", "answer": f"Change map computation failed: {exc}", "evidence": []}
