import os
import httpx

MODEL_REGISTRY = {
    "vqa": {
        "model": "Salesforce/blip-vqa-base",
        "version": "1.0",
        "base_url": os.getenv("VQA_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "accepted_modalities": ["optical", "sar", "unknown"],
        "required_images": 1,
        "permitted_parameters": [],
        "output_schema": "EvidenceItem"
    },
    "captioning": {
        "model": "Salesforce/blip-image-captioning-base",
        "version": "1.0",
        "base_url": os.getenv("CAPTIONING_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "accepted_modalities": ["optical", "sar", "unknown"],
        "required_images": 1,
        "permitted_parameters": [],
        "output_schema": "EvidenceItem"
    },
    "grounding": {
        "model": "IDEA-Research/grounding-dino-base",
        "version": "1.0",
        "base_url": os.getenv("GROUNDING_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "accepted_modalities": ["optical", "unknown"],
        "required_images": 1,
        "permitted_parameters": ["confidence_threshold"],
        "output_schema": "EvidenceItem"
    },
    "change_analysis": {
        "model": "Baseline-Pixel-Diff",
        "version": "1.0",
        "base_url": os.getenv("CHANGE_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "accepted_modalities": ["optical", "sar", "unknown"],
        "required_images": 2,
        "permitted_parameters": [],
        "output_schema": "EvidenceItem"
    },
    "optical_sar": {
        "model": "CrossModal-Verifier",
        "version": "1.0",
        "base_url": os.getenv("OPTICAL_SAR_ENDPOINT", ""),
        "input_requirements": ["geotiff"],
        "accepted_modalities": ["optical", "sar"],
        "required_images": 2,
        "permitted_parameters": [],
        "output_schema": "EvidenceItem"
    },
    "change_map": {
        "model": "Baseline-Pixel-Diff",
        "version": "1.0",
        "base_url": os.getenv("CHANGE_MAP_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "accepted_modalities": ["optical", "sar", "unknown"],
        "required_images": 2,
        "permitted_parameters": [],
        "output_schema": "EvidenceItem"
    },
    "land_cover_classification": {
        "model": "nielsr/convnext-tiny-finetuned-eurosat",
        "version": "1.0",
        "base_url": os.getenv("LAND_COVER_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "accepted_modalities": ["optical", "unknown"],
        "required_images": 1,
        "permitted_parameters": [],
        "output_schema": "EvidenceItem"
    }
}

def get_tool_status(task: str) -> str:
    entry = MODEL_REGISTRY.get(task)
    if not entry:
        return "UNKNOWN"
    if not entry.get("base_url"):
        return "UNAVAILABLE"
    try:
        response = httpx.get(f"{entry['base_url']}/health", timeout=2.0)
        if response.status_code == 200:
            return "AVAILABLE"
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return "UNAVAILABLE"
