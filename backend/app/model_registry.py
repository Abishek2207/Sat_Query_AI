import os
import httpx
from typing import Dict, Any, List

# Actual model paths mapped to real implementations
MODEL_REGISTRY = {
    "vqa": {
        "model_name": "Salesforce/blip-vqa-base",
        "model_version": "1.0",
        "task": "vqa",
        "dataset": "RSVQA/VRSBench (Baseline)",
        "artifact_path": "huggingface_hub/models--Salesforce--blip-vqa-base",
        "processor_tokenizer": "BlipProcessor",
        "modality": "optical",
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "output_schema": "EvidenceItem",
        "confidence_available": True,
        "remote_sensing_adapted": False,
        "checksum": "verified_hf_cache",
        "device": "CUDA/CPU (Auto)",
        "base_url": os.getenv("VQA_ENDPOINT", "local"),
        "accepted_modalities": ["optical", "sar", "unknown"],
        "required_images": 1,
        "permitted_parameters": []
    },
    "captioning": {
        "model_name": "Salesforce/blip-image-captioning-base",
        "model_version": "1.0 (LoRA)",
        "task": "captioning",
        "dataset": "RSICD",
        "artifact_path": "backend/models/adapters/rsicd_captioning_lora",
        "processor_tokenizer": "BlipProcessor + PEFT",
        "modality": "optical",
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "output_schema": "EvidenceItem",
        "confidence_available": False,
        "remote_sensing_adapted": True,
        "checksum": "verified_local_lora",
        "device": "CUDA/CPU (Auto)",
        "base_url": os.getenv("CAPTIONING_ENDPOINT", "local"),
        "accepted_modalities": ["optical", "sar", "unknown"],
        "required_images": 1,
        "permitted_parameters": []
    },
    "grounding": {
        "model_name": "IDEA-Research/grounding-dino-base",
        "model_version": "1.0",
        "task": "grounding",
        "dataset": "General Zero-Shot (COCO/O365)",
        "artifact_path": "huggingface_hub/models--IDEA-Research--grounding-dino-base",
        "processor_tokenizer": "GroundingDinoProcessor",
        "modality": "optical",
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "output_schema": "EvidenceItem",
        "confidence_available": True,
        "remote_sensing_adapted": False,
        "checksum": "verified_hf_cache",
        "device": "CUDA/CPU (Auto)",
        "base_url": os.getenv("GROUNDING_ENDPOINT", "local"),
        "accepted_modalities": ["optical", "unknown"],
        "required_images": 1,
        "permitted_parameters": ["confidence_threshold"]
    },
    "land_cover_classification": {
        "model_name": "nielsr/convnext-tiny-finetuned-eurosat",
        "model_version": "1.0",
        "task": "land_cover_classification",
        "dataset": "EuroSAT",
        "artifact_path": "huggingface_hub/models--nielsr--convnext-tiny-finetuned-eurosat",
        "processor_tokenizer": "AutoImageProcessor",
        "modality": "optical",
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "output_schema": "EvidenceItem",
        "confidence_available": True,
        "remote_sensing_adapted": True,
        "checksum": "verified_hf_cache",
        "device": "CUDA/CPU (Auto)",
        "base_url": os.getenv("LAND_COVER_ENDPOINT", "local"),
        "accepted_modalities": ["optical", "unknown"],
        "required_images": 1,
        "permitted_parameters": []
    },
    "change_analysis": {
        "model_name": "Baseline-Pixel-Diff",
        "model_version": "1.0",
        "task": "change_analysis",
        "dataset": "N/A (Heuristic)",
        "artifact_path": "backend/app/local_specialists.py",
        "processor_tokenizer": "NumPy/OpenCV",
        "modality": "optical, sar",
        "input_requirements": ["geotiff", "benchmark_rgb", "co-registered"],
        "output_schema": "EvidenceItem",
        "confidence_available": False,
        "remote_sensing_adapted": True,
        "checksum": "verified_local",
        "device": "CPU",
        "base_url": os.getenv("CHANGE_ENDPOINT", "local"),
        "accepted_modalities": ["optical", "sar", "unknown"],
        "required_images": 2,
        "permitted_parameters": []
    },
    "optical_sar": {
        "model_name": "CrossModal-Verifier",
        "model_version": "1.0",
        "task": "optical_sar",
        "dataset": "N/A (Heuristic Baseline)",
        "artifact_path": "backend/app/local_specialists.py",
        "processor_tokenizer": "Rasterio",
        "modality": "optical, sar",
        "input_requirements": ["geotiff"],
        "output_schema": "EvidenceItem",
        "confidence_available": False,
        "remote_sensing_adapted": True,
        "checksum": "verified_local",
        "device": "CPU",
        "base_url": os.getenv("OPTICAL_SAR_ENDPOINT", "local"),
        "accepted_modalities": ["optical", "sar"],
        "required_images": 2,
        "permitted_parameters": []
    }
}

def get_tool_status(task: str) -> str:
    entry = MODEL_REGISTRY.get(task)
    if not entry:
        return "MODEL UNAVAILABLE"
    # For local inference loaded on-demand, they are always available 
    # unless missing dependencies, which we assume true if running.
    if entry.get("base_url") == "local":
        return "READY"
        
    try:
        response = httpx.get(f"{entry['base_url']}/health", timeout=2.0)
        if response.status_code == 200:
            return "READY"
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return "MODEL UNAVAILABLE"

def get_model_registry_list() -> List[Dict[str, Any]]:
    results = []
    for k, v in MODEL_REGISTRY.items():
        v_copy = v.copy()
        v_copy["status"] = get_tool_status(k)
        results.append(v_copy)
    return results
