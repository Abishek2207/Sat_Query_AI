"""
change_map.py — Bi-temporal Change Detection Specialist

Two operating modes (selected automatically):

1. PRETRAINED_FEATURE_CHANGE_DETECTION
   Uses ResNet18 backbone pretrained on ImageNet (torchvision IMAGENET1K_V1).
   Sentinel-2 bands B04/B03/B02 are used as Red/Green/Blue — the standard
   Sentinel-2 true-colour representation.
   A Siamese forward pass extracts layer-3 feature maps (~256 channels) for
   both images, then per-pixel cosine distance is computed and upsampled to
   the original resolution. This is a pretrained-feature approach, NOT a
   dedicated remote-sensing change-detection checkpoint.

2. DETERMINISTIC_CHANGE_BASELINE (fallback)
   Radiometrically normalised pixel-differencing + morphological dilation.
   Used when neural path fails or encoder cannot be loaded.
"""

import io
import gc
import base64
import time
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from PIL import Image
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _simple_dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def _mask_to_b64png(arr_uint8: np.ndarray) -> str:
    img = Image.fromarray(arr_uint8, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _describe_change(pct: float) -> str:
    if pct < 1.0:
        return "No significant change detected between the two images."
    elif pct < 10.0:
        return f"Minor change detected in approximately {pct:.1f}% of the valid image area."
    elif pct < 40.0:
        return f"Moderate change detected in {pct:.1f}% of the valid image area."
    else:
        return f"Significant change detected across {pct:.1f}% of the valid image area."


# ---------------------------------------------------------------------------
# Sentinel-2 RGB extractor
# B02=band1 (Blue), B03=band2 (Green), B04=band3 (Red) in the validated dataset.
# Returns float32 (3, H, W) in [0, 1] or None on failure.
# ---------------------------------------------------------------------------

def _read_s2_rgb(src) -> Optional[np.ndarray]:
    try:
        n = src.count
        if n >= 3:
            b02 = src.read(1).astype(np.float32)
            b03 = src.read(2).astype(np.float32)
            b04 = src.read(3).astype(np.float32)
        elif n == 1:
            b02 = b03 = b04 = src.read(1).astype(np.float32)
        else:
            return None

        def _pct_norm(arr: np.ndarray) -> np.ndarray:
            valid = arr[arr > 0]
            if valid.size == 0:
                return np.zeros_like(arr)
            p2 = np.percentile(valid, 2)
            p98 = np.percentile(valid, 98)
            p98 = max(p98, p2 + 1.0)
            return np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)

        # Standard Sentinel-2 true colour: R=B04, G=B03, B=B02
        rgb = np.stack([_pct_norm(b04), _pct_norm(b03), _pct_norm(b02)], axis=0)
        return rgb.astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Siamese ResNet18 encoder (lazy-loaded, cached globally)
# ---------------------------------------------------------------------------

_CHANGE_ENCODER_CACHE: dict = {}


def _get_change_encoder():
    if "model" in _CHANGE_ENCODER_CACHE:
        return _CHANGE_ENCODER_CACHE["model"]

    import torch
    import torchvision.models as tvm

    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    backbone = tvm.resnet18(weights=weights)
    backbone.eval()

    # Use layers up to layer3 (stride-8); gives (256, H/8, W/8) feature maps
    encoder = torch.nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )
    encoder.eval()
    _CHANGE_ENCODER_CACHE["model"] = encoder
    return encoder


def _siamese_change_inference(
    rgb1: np.ndarray,
    rgb2: np.ndarray,
    threshold: float = 0.35,
):
    import torch
    import torch.nn.functional as F

    encoder = _get_change_encoder()
    H, W = rgb1.shape[1], rgb1.shape[2]

    # ImageNet normalisation
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _to_tensor(arr):
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,3,H,W)
        return (t - imagenet_mean) / imagenet_std

    with torch.inference_mode():
        f1 = encoder(_to_tensor(rgb1))   # (1,C,Hf,Wf)
        f2 = encoder(_to_tensor(rgb2))
        cos_sim = F.cosine_similarity(f1, f2, dim=1)  # (1,Hf,Wf)
        dist = (1.0 - cos_sim) / 2.0                  # 0=same, 1=maximally different
        dist_up = F.interpolate(
            dist.unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False
        )
        dist_np = dist_up.squeeze().numpy()

    change_mask = (dist_np > threshold).astype(np.uint8)
    return dist_np, change_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_change_baseline(
    img_t1_bytes: bytes,
    img_t2_bytes: bytes,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    t_start = time.time()

    try:
        with MemoryFile(img_t1_bytes) as m1, MemoryFile(img_t2_bytes) as m2:
            with m1.open() as src1, m2.open() as src2:

                if src1.width != src2.width or src1.height != src2.height:
                    return {
                        "status": "INVALID_INPUT",
                        "answer": "Spatial dimensions mismatch between before/after images.",
                        "evidence": [],
                    }

                crs_match = src1.crs == src2.crs
                crs_note = "CRS match confirmed." if crs_match else "WARNING: CRS mismatch between inputs."

                # ---- Attempt neural path --------------------------------
                neural_ok = False
                rgb1 = _read_s2_rgb(src1)
                rgb2 = _read_s2_rgb(src2)

                if rgb1 is not None and rgb2 is not None:
                    try:
                        dist_map, change_mask = _siamese_change_inference(rgb1, rgb2, threshold=0.35)
                        valid_pixels   = rgb1.shape[1] * rgb1.shape[2]
                        changed_pixels = int(change_mask.sum())
                        change_pct     = (changed_pixels / valid_pixels) * 100.0

                        # Heatmap PNG from distance map
                        heat = (np.clip(dist_map, 0, 1) * 255).astype(np.uint8)
                        mask_b64 = _mask_to_b64png(heat)

                        method      = (
                            "Pretrained ImageNet feature-based temporal change analysis. "
                            "Siamese ResNet18 (torchvision IMAGENET1K_V1) cosine feature distance "
                            "on Sentinel-2 RGB subset (B04=Red, B03=Green, B02=Blue). "
                            "B08 (NIR) is preserved in the dataset but NOT consumed by this backbone. "
                            "This is NOT a remote-sensing-trained change detector."
                        )
                        method_type = "PRETRAINED_FEATURE_CHANGE_DETECTION"
                        model_name  = "ResNet18_Siamese_ImageNet"
                        model_ver   = "torchvision-IMAGENET1K_V1"
                        conf_note   = (
                            "Feature-space change percentage is not model confidence. "
                            "Threshold=0.35 on cosine distance. No RS fine-tuning. "
                            "Semantic change classification not available from this backbone."
                        )
                        neural_ok   = True
                    except Exception:
                        neural_ok = False

                # ---- Deterministic fallback --------------------------------
                if not neural_ok:
                    data1 = src1.read(1).astype(np.float32)
                    data2 = src2.read(1).astype(np.float32)
                    valid_mask = (data1 > 0) & (data2 > 0)
                    p99_1 = max(np.percentile(data1[valid_mask], 99) if np.any(valid_mask) else 1.0, 1.0)
                    p99_2 = max(np.percentile(data2[valid_mask], 99) if np.any(valid_mask) else 1.0, 1.0)
                    d1_n = np.clip(data1 / p99_1, 0, 1)
                    d2_n = np.clip(data2 / p99_2, 0, 1)
                    diff = np.abs(d2_n - d1_n)
                    diff[~valid_mask] = 0
                    raw_change  = diff > threshold
                    clean_change = _simple_dilate(raw_change)
                    valid_pixels   = int(np.sum(valid_mask))
                    changed_pixels = int(np.sum(clean_change))
                    change_pct = (changed_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0
                    mask_b64   = _mask_to_b64png(clean_change.astype(np.uint8) * 255)
                    method      = f"Radiometrically normalised pixel-differencing + morphological dilation (threshold={threshold})"
                    method_type = "DETERMINISTIC_CHANGE_BASELINE"
                    model_name  = "Real_BiTemporal_Change_Baseline"
                    model_ver   = "1.2"
                    conf_note   = f"Pixel-difference threshold={threshold}; nodata=0."

                # ---- Assemble response --------------------------------
                description = _describe_change(change_pct)
                prefix = f"[{method_type}]"
                answer = f"{prefix} {description}"

                ev = {
                    "claim":           f"Change detected in {change_pct:.1f}% of the area.",
                    "evidence":        f"Method: {method}. {crs_note} {conf_note}",
                    "region":          None,
                    "timestamp":       None,
                    "modality":        "optical_imagery",
                    "confidence":      None,
                    "confidence_type": None,
                    "status":          "VERIFIED",
                    "source":          "Change Map Specialist",
                    "model":           model_name,
                    "model_version":   model_ver,
                }

                latency_s = round(time.time() - t_start, 2)

                return {
                    "status":       "SUCCESS",
                    "answer":       answer,
                    "confidence":   None,
                    "evidence":     [ev],
                    "visual_output": f"data:image/png;base64,{mask_b64}" if mask_b64 else None,
                    "provenance": {
                        "model":                    model_name,
                        "model_version":            model_ver,
                        "method":                   method,
                        "method_type":              method_type,
                        "adaptation_dataset":       None,
                        "adaptation_method":        None,
                        "remote_sensing_adapted":   False,
                        "inference_timestamp":      "",
                        "device":                   "cpu",
                        "input_modalities":         ["optical", "optical"],
                        "crs":                      str(src1.crs) if src1.crs else None,
                        "geospatial_evidence_generated": True,
                        "latency_s":                latency_s,
                    },
                }

    except Exception as exc:
        return {"status": "ERROR", "answer": f"Change map computation failed: {exc}", "evidence": []}
