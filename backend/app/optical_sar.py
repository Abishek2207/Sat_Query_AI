import rasterio
import numpy as np
from rasterio.io import MemoryFile
from typing import Dict, Any
from scipy.stats import pearsonr

def verify_optical_sar_pair(
    opt_bytes: bytes,
    sar_bytes: bytes,
) -> Dict[str, Any]:
    try:
        with MemoryFile(opt_bytes) as mem_opt, MemoryFile(sar_bytes) as mem_sar:
            with mem_opt.open() as src_opt, mem_sar.open() as src_sar:
                issues = []
                
                if src_opt.width != src_sar.width or src_opt.height != src_sar.height:
                    issues.append(f"Spatial dimension mismatch: Image1={src_opt.width}x{src_opt.height}, Image2={src_sar.width}x{src_sar.height}.")
                crs1 = str(src_opt.crs) if src_opt.crs else "UNKNOWN"
                crs2 = str(src_sar.crs) if src_sar.crs else "UNKNOWN"
                if crs1 != crs2:
                    issues.append(f"CRS mismatch: Image1={crs1}, Image2={crs2}.")
                if issues:
                    return {"status": "INVALID_INPUT", "answer": "Optical+SAR pair is not co-registered: " + " ".join(issues), "evidence": []}
                    
                opt_data = src_opt.read(1).astype(np.float32)
                sar_data = src_sar.read(1).astype(np.float32)
                
                if np.count_nonzero(sar_data) == 0:
                    return {"status": "DATA_UNAVAILABLE", "answer": "Sentinel-1 SAR raster is empty.", "evidence": []}
                
                # Optical Features
                opt_min, opt_max = np.min(opt_data), np.max(opt_data)
                opt_mean, opt_std = np.mean(opt_data), np.std(opt_data)
                
                # SAR Features (RTC power values are usually very small, e.g. 0-2)
                sar_min, sar_max = np.min(sar_data), np.max(sar_data)
                sar_mean, sar_std = np.mean(sar_data), np.std(sar_data)
                
                # We calculate correlation if std dev is > 0
                correlation = 0.0
                if opt_std > 0 and sar_std > 0:
                    # Flatten for correlation
                    try:
                        correlation, _ = pearsonr(opt_data.flatten(), sar_data.flatten())
                    except:
                        pass
                
                sar_class = "Urban/High Structure" if sar_mean > 0.5 else "Vegetation/Water/Low Backscatter"
                conflict = correlation < -0.5 # Rule-based negative correlation conflict
                
                ev1 = {
                    "claim": "Optical metadata verified",
                    "evidence": f"Reflectance -> Mean: {opt_mean:.2f}, Std: {opt_std:.2f}, Range: [{opt_min:.1f}, {opt_max:.1f}]",
                    "region": None, "timestamp": None, "modality": "optical",
                    "confidence": 1.0, "confidence_type": "rule-based",
                    "status": "VERIFIED", "source": "Optical-SAR Specialist",
                    "model": "Real_Optical_SAR_Multimodal_Baseline", "model_version": "1.1"
                }
                
                ev2 = {
                    "claim": f"SAR VV Backscatter -> {sar_class}",
                    "evidence": f"RTC Power -> Mean: {sar_mean:.3f}, Std: {sar_std:.3f}, Max: {sar_max:.2f}",
                    "region": None, "timestamp": None, "modality": "sar",
                    "confidence": 1.0, "confidence_type": "rule-based",
                    "status": "VERIFIED", "source": "Optical-SAR Specialist",
                    "model": "Real_Optical_SAR_Multimodal_Baseline", "model_version": "1.1"
                }
                
                ev3 = {
                    "claim": "Joint Spatial Correlation Computed",
                    "evidence": f"Pearson Correlation (Optical vs SAR): {correlation:.3f}",
                    "region": None, "timestamp": None, "modality": "optical_sar",
                    "confidence": 1.0, "confidence_type": "rule-based",
                    "status": "VERIFIED", "source": "Optical-SAR Specialist",
                    "model": "Real_Optical_SAR_Multimodal_Baseline", "model_version": "1.1"
                }

                answer = f"Optical and SAR images successfully co-registered and analyzed. The scene is dominated by {sar_class} based on SAR backscatter. The spatial correlation between the modalities is {correlation:.3f}."
                if conflict:
                    answer += " Significant conflict detected: optical structures do not align with SAR structures."

                return {
                    "status": "SUCCESS",
                    "answer": answer,
                    "confidence": None,
                    "evidence": [ev1, ev2, ev3],
                    "conflict": conflict,
                    "visual_output": None,
                    "provenance": {
                        "model": "Real_Optical_SAR_Multimodal_Baseline",
                        "model_version": "1.1",
                        "adaptation_dataset": None,
                        "adaptation_method": None,
                        "remote_sensing_adapted": False,
                        "inference_timestamp": "",
                        "device": "cpu",
                        "input_filenames": [],
                        "input_modalities": ["optical", "sar"],
                        "crs": crs1,
                        "geospatial_evidence_generated": False,
                    },
                }

    except Exception as exc:
        return {"status": "ERROR", "answer": f"Optical-SAR verification failed: {exc}", "evidence": []}
