"""
optical_sar.py — Optical + SAR Joint Analysis Specialist

Two operating modes (selected automatically):

1. PRETRAINED_FEATURE_OPTICAL_SAR (when 4-band optical and 2-band SAR are available)
   Uses two separate ResNet18-derived encoders:
   - Optical branch: standard ResNet18 (ImageNet pretrained) on Sentinel-2 RGB (B04/B03/B02)
   - SAR branch:     a custom 2-channel convolutional stem feeding into ResNet18 layer1-3,
                     with a randomly-initialised input conv (the rest of the backbone uses
                     ImageNet weights). The random stem is clearly documented.
   Semantic feature similarity between optical and SAR embeddings is then computed.
   This is PRETRAINED_FEATURE_OPTICAL_SAR — NOT a jointly trained optical-SAR model.

2. STATISTICAL_OPTICAL_SAR_BASELINE (fallback)
   Co-registration verification, Pearson correlation, backscatter statistics.

IMPORTANT constraints:
  - sar_after.tif does NOT exist in the current dataset.
  - This specialist operates on single-epoch (before) optical + SAR only.
  - Any query about SAR temporal change must receive DATA_UNAVAILABLE.
"""

import gc
import time
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from scipy.stats import pearsonr
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_s2_rgb(src) -> Optional[np.ndarray]:
    """Return float32 (3, H, W) in [0, 1] from Sentinel-2 src, or None."""
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

        def _pct_norm(arr):
            valid = arr[arr > 0]
            if valid.size == 0:
                return np.zeros_like(arr)
            p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
            p98 = max(p98, p2 + 1.0)
            return np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)

        return np.stack([_pct_norm(b04), _pct_norm(b03), _pct_norm(b02)], axis=0).astype(np.float32)
    except Exception:
        return None


def _read_sar_normed(src) -> Optional[np.ndarray]:
    """Return float32 (2, H, W) in [0, 1] from SAR src, or None."""
    try:
        if src.count < 2:
            return None
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)

        def _log_norm(arr):
            # Log transform for SAR power values, then percentile-normalise
            arr = np.where(arr > 0, arr, 1e-6)
            arr_db = 10 * np.log10(arr)
            p2, p98 = np.percentile(arr_db, 2), np.percentile(arr_db, 98)
            p98 = max(p98, p2 + 1.0)
            return np.clip((arr_db - p2) / (p98 - p2), 0.0, 1.0)

        return np.stack([_log_norm(vv), _log_norm(vh)], axis=0).astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dual-branch feature encoder (lazy-loaded, cached globally)
# ---------------------------------------------------------------------------

_SAR_OPT_CACHE: dict = {}


def _get_encoders():
    if "optical" in _SAR_OPT_CACHE:
        return _SAR_OPT_CACHE["optical"], _SAR_OPT_CACHE["sar"]

    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    # --- Optical encoder: ResNet18 pretrained ImageNet, RGB input ---
    backbone_opt = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    enc_opt = nn.Sequential(
        backbone_opt.conv1,   # 3-channel input
        backbone_opt.bn1,
        backbone_opt.relu,
        backbone_opt.maxpool,
        backbone_opt.layer1,
        backbone_opt.layer2,
        backbone_opt.layer3,  # (256, H/8, W/8)
    )
    enc_opt.eval()

    # --- SAR encoder: 2-channel random stem + ImageNet backbone layers ---
    # The 2-channel conv1 is randomly initialised (documented below).
    # Layers 1-3 reuse ImageNet weights from a second ResNet18.
    backbone_sar = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    sar_stem = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # random init of sar_stem — explicitly noted
    enc_sar = nn.Sequential(
        sar_stem,             # 2-channel random-init input conv
        backbone_sar.bn1,     # ImageNet BN
        backbone_sar.relu,
        backbone_sar.maxpool,
        backbone_sar.layer1,  # ImageNet weights
        backbone_sar.layer2,
        backbone_sar.layer3,  # (256, H/8, W/8)
    )
    enc_sar.eval()

    _SAR_OPT_CACHE["optical"] = enc_opt
    _SAR_OPT_CACHE["sar"] = enc_sar
    return enc_opt, enc_sar


def _dual_branch_similarity(
    rgb: np.ndarray,      # (3, H, W) in [0, 1]
    sar: np.ndarray,      # (2, H, W) in [0, 1]
) -> float:
    """
    Extract pooled feature embeddings from optical and SAR branches,
    return cosine similarity in [-1, 1].
    1.0 = fully aligned (semantically similar), lower = divergent.
    """
    import torch
    import torch.nn.functional as F

    enc_opt, enc_sar = _get_encoders()

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    t_rgb = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0)
    t_rgb = (t_rgb - imagenet_mean) / imagenet_std

    # SAR: normalise each channel with mean/std of the input
    t_sar = torch.tensor(sar, dtype=torch.float32).unsqueeze(0)  # (1,2,H,W)
    for c in range(t_sar.shape[1]):
        ch = t_sar[0, c]
        t_sar[0, c] = (ch - ch.mean()) / (ch.std() + 1e-6)

    with torch.inference_mode():
        f_opt = enc_opt(t_rgb)   # (1,256,Hf,Wf)
        f_sar = enc_sar(t_sar)

        # Global average pool to get (1,256) embedding each
        e_opt = F.adaptive_avg_pool2d(f_opt, (1, 1)).flatten()
        e_sar = F.adaptive_avg_pool2d(f_sar, (1, 1)).flatten()

        cos_sim = float(F.cosine_similarity(e_opt.unsqueeze(0), e_sar.unsqueeze(0)).item())

    return cos_sim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_optical_sar_pair(
    opt_bytes: bytes,
    sar_bytes: bytes,
) -> Dict[str, Any]:
    t_start = time.time()

    try:
        with MemoryFile(opt_bytes) as mem_opt, MemoryFile(sar_bytes) as mem_sar:
            with mem_opt.open() as src_opt, mem_sar.open() as src_sar:

                # --- Grid/CRS validation ---
                issues = []
                if src_opt.width != src_sar.width or src_opt.height != src_sar.height:
                    issues.append(
                        f"Dimension mismatch: optical={src_opt.width}x{src_opt.height}, "
                        f"SAR={src_sar.width}x{src_sar.height}."
                    )
                crs_opt = str(src_opt.crs) if src_opt.crs else "UNKNOWN"
                crs_sar = str(src_sar.crs) if src_sar.crs else "UNKNOWN"
                if crs_opt != crs_sar:
                    issues.append(f"CRS mismatch: optical={crs_opt}, SAR={crs_sar}.")
                if issues:
                    return {
                        "status": "INVALID_INPUT",
                        "answer": "Optical+SAR pair not co-registered: " + " ".join(issues),
                        "evidence": [],
                    }

                # --- Basic SAR validity ---
                sar_b1 = src_sar.read(1).astype(np.float32)
                if np.count_nonzero(sar_b1 > 0) == 0:
                    return {
                        "status": "DATA_UNAVAILABLE",
                        "answer": "Sentinel-1 SAR raster is empty.",
                        "evidence": [],
                    }

                # --- Statistical baseline measurements (always computed) ---
                opt_b1 = src_opt.read(1).astype(np.float32)
                sar_vv = src_sar.read(1).astype(np.float32)
                sar_vh = src_sar.read(2).astype(np.float32) if src_sar.count >= 2 else sar_vv

                opt_mean, opt_std = float(np.mean(opt_b1)), float(np.std(opt_b1))
                vv_mean,  vv_std  = float(np.mean(sar_vv)), float(np.std(sar_vv))
                vh_mean,  vh_std  = float(np.mean(sar_vh)), float(np.std(sar_vh))

                pearson_vv = 0.0
                if opt_std > 0 and vv_std > 0:
                    try:
                        pearson_vv, _ = pearsonr(opt_b1.flatten(), sar_vv.flatten())
                    except Exception:
                        pass

                sar_class = (
                    "Urban/Built-up (high backscatter)"
                    if vv_mean > 0.5
                    else "Vegetation/Water/Low Backscatter"
                )

                # VV/VH ratio — discriminates surface types
                vvvh_ratio = float(np.mean(sar_vv / (sar_vh + 1e-9)))

                # --- Attempt pretrained-feature path ---
                neural_ok = False
                cos_sim   = None

                rgb = _read_s2_rgb(src_opt)
                sar_normed = _read_sar_normed(src_sar)

                if rgb is not None and sar_normed is not None:
                    try:
                        cos_sim  = _dual_branch_similarity(rgb, sar_normed)
                        neural_ok = True
                    except Exception:
                        neural_ok = False

                # --- Build answer ---
                if neural_ok:
                    method_type = "EXPERIMENTAL_OPTICAL_SAR_FEATURE_COMPARISON"
                    method = (
                        "Experimental dual-branch ResNet18: optical branch uses ImageNet-pretrained weights on "
                        "Sentinel-2 RGB (B04/B03/B02). SAR branch uses a randomly-initialised 2-channel input "
                        "conv stem (NOT pretrained on SAR) followed by ImageNet-pretrained layers 1-3. "
                        "Feature-space cosine similarity from global-average-pooled embeddings. "
                        "This is NOT a jointly trained or dedicated optical-SAR model."
                    )
                    model_name = "Dual_ResNet18_Experimental_OpticalSAR"
                    model_ver  = "optical=IMAGENET1K_V1; sar_stem=random_init"
                    sim_str = f"Feature-space cosine similarity (optical vs SAR) = {cos_sim:.3f} (NOT model confidence)"
                    answer = (
                        f"[EXPERIMENTAL_OPTICAL_SAR_FEATURE_COMPARISON] Single-epoch optical-SAR analysis. "
                        f"SAR VV mean={vv_mean:.4f}, VH mean={vh_mean:.4f}, VV/VH ratio={vvvh_ratio:.2f} "
                        f"(heuristic class: {sar_class}). {sim_str}. "
                        f"Optical-SAR Pearson correlation (band1 vs VV) = {pearson_vv:.3f}."
                    )
                else:
                    method_type = "STATISTICAL_OPTICAL_SAR_BASELINE"
                    method = "Spatial co-registration verification + Pearson correlation + SAR backscatter statistics."
                    model_name = "Real_Optical_SAR_Multimodal_Baseline"
                    model_ver  = "1.2"
                    answer = (
                        f"[STATISTICAL_OPTICAL_SAR_BASELINE] Single-epoch optical-SAR analysis. "
                        f"SAR scene dominated by {sar_class}. "
                        f"Optical-SAR Pearson correlation (band1 vs VV) = {pearson_vv:.3f}."
                    )

                ev_opt = {
                    "claim":           "Optical metadata verified",
                    "evidence":        f"Band1 mean={opt_mean:.2f}, std={opt_std:.2f}",
                    "region":          None, "timestamp": None, "modality": "optical",
                    "confidence":      1.0, "confidence_type": "rule-based",
                    "status":          "VERIFIED", "source": "Optical-SAR Specialist",
                    "model":           model_name, "model_version": model_ver,
                }
                ev_sar = {
                    "claim":           f"SAR backscatter class: {sar_class}",
                    "evidence":        f"VV mean={vv_mean:.4f}, std={vv_std:.4f}; VH mean={vh_mean:.4f}, std={vh_std:.4f}; VV/VH={vvvh_ratio:.2f}",
                    "region":          None, "timestamp": None, "modality": "sar",
                    "confidence":      1.0, "confidence_type": "rule-based",
                    "status":          "VERIFIED", "source": "Optical-SAR Specialist",
                    "model":           model_name, "model_version": model_ver,
                }
                evidence = [ev_opt, ev_sar]

                if neural_ok:
                    ev_neural = {
                        "claim":           "Experimental optical-SAR feature-space comparison",
                        "evidence":        (
                            f"Feature-space cosine similarity={cos_sim:.3f}. "
                            "NOTE: This is NOT model confidence. "
                            "SAR encoder input stem is randomly initialised (2→64 channels, no SAR pretraining); "
                            "ImageNet-pretrained layers 1-3 are shared. "
                            "This is an experimental comparison, NOT a jointly trained optical-SAR model."
                        ),
                        "region":          None, "timestamp": None, "modality": "optical_sar",
                        "confidence":      None, "confidence_type": None,
                        "status":          "PARTIALLY_VERIFIED", "source": "Optical-SAR Specialist",
                        "model":           model_name, "model_version": model_ver,
                    }
                    evidence.append(ev_neural)
                else:
                    ev_stat = {
                        "claim":           "Statistical cross-modal correlation computed",
                        "evidence":        f"Pearson correlation (optical band1 vs SAR VV) = {pearson_vv:.3f}",
                        "region":          None, "timestamp": None, "modality": "optical_sar",
                        "confidence":      1.0, "confidence_type": "rule-based",
                        "status":          "VERIFIED", "source": "Optical-SAR Specialist",
                        "model":           model_name, "model_version": model_ver,
                    }
                    evidence.append(ev_stat)

                latency_s = round(time.time() - t_start, 2)

                return {
                    "status":       "SUCCESS",
                    "answer":       answer,
                    "confidence":   None,
                    "evidence":     evidence,
                    "conflict":     (pearson_vv < -0.5),
                    "visual_output": None,
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
                        "input_modalities":         ["optical", "sar"],
                        "crs":                      crs_opt,
                        "geospatial_evidence_generated": False,
                        "latency_s":                latency_s,
                    },
                }

    except Exception as exc:
        return {"status": "ERROR", "answer": f"Optical-SAR verification failed: {exc}", "evidence": []}
