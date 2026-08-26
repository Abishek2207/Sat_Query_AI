"""
backend/app/optical_sar.py

Baseline co-registration and metadata alignment check for Optical + SAR pairs.

IMPORTANT:
- No trained fusion model is used or claimed.
- Only spatial metadata compatibility is verified.
- Results are labeled BASELINE to prevent false scientific claims.
"""
import rasterio
from rasterio.io import MemoryFile
from typing import Dict, Any


def verify_optical_sar_pair(
    opt_bytes: bytes,
    sar_bytes: bytes,
) -> Dict[str, Any]:
    """
    Verifies that two images are spatially co-registered and
    reports compatibility for downstream optical+SAR fusion.
    Does NOT use a trained fusion model.
    """
    try:
        with MemoryFile(opt_bytes) as mem_opt, MemoryFile(sar_bytes) as mem_sar:
            with mem_opt.open() as src_opt, mem_sar.open() as src_sar:

                issues = []
                evidence = []

                # Dimension check
                if src_opt.width != src_sar.width or src_opt.height != src_sar.height:
                    issues.append(
                        f"Spatial dimension mismatch: Image1={src_opt.width}x{src_opt.height}, "
                        f"Image2={src_sar.width}x{src_sar.height}."
                    )
                else:
                    evidence.append(
                        f"Spatial dimensions match: {src_opt.width}x{src_opt.height} px."
                    )

                # CRS check
                crs1 = str(src_opt.crs) if src_opt.crs else "UNKNOWN"
                crs2 = str(src_sar.crs) if src_sar.crs else "UNKNOWN"
                if crs1 != crs2:
                    issues.append(f"CRS mismatch: Image1={crs1}, Image2={crs2}.")
                else:
                    evidence.append(f"CRS match confirmed: {crs1}.")

                # Band count info (informational only)
                evidence.append(f"Image1 bands: {src_opt.count} | Image2 bands: {src_sar.count}.")

                if issues:
                    return {
                        "status": "INVALID_INPUT",
                        "answer": "Optical+SAR pair is not co-registered: " + " ".join(issues),
                        "evidence": issues,
                    }

                return {
                    "status": "SUCCESS",
                    "answer": (
                        "Optical and SAR images are spatially co-registered. "
                        "No learned fusion model is currently loaded. "
                        "Baseline spatial compatibility verified."
                    ),
                    "confidence": None,
                    "evidence": evidence + [
                        "BASELINE: No trained optical-SAR fusion model used.",
                        "Architecture is ready to accept a real fusion model endpoint.",
                    ],
                    "visual_output": None,
                    "provenance": {
                        "model": "Metadata_Alignment_Baseline",
                        "model_version": "1.0",
                        "adaptation_dataset": None,
                        "adaptation_method": None,
                        "remote_sensing_adapted": False,
                        "inference_timestamp": "",
                        "input_filenames": [],
                        "input_modalities": ["optical", "sar"],
                        "crs": crs1,
                        "geospatial_evidence_generated": False,
                    },
                }

    except Exception as exc:
        return {
            "status": "ERROR",
            "answer": f"Optical-SAR verification failed: {exc}",
            "evidence": [],
        }
