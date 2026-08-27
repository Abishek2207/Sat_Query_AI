import os

import httpx


MODEL_REGISTRY = {
    "vqa": {
        "base_url": os.getenv("VQA_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "required_images": 1,
        "permitted_parameters": [],
    },

    "captioning": {
        "base_url": os.getenv("CAPTIONING_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "required_images": 1,
        "permitted_parameters": [],
    },

    "grounding": {
        "base_url": os.getenv("GROUNDING_ENDPOINT", ""),
        "input_requirements": ["geotiff"],
        "required_images": 1,
        "permitted_parameters": ["confidence_threshold"],
    },

    "change_analysis": {
        "base_url": os.getenv("CHANGE_ENDPOINT", ""),
        "input_requirements": ["geotiff"],
        "required_images": 2,
        "permitted_parameters": [],
    },

    "optical_sar": {
        "base_url": os.getenv("OPTICAL_SAR_ENDPOINT", ""),
        "input_requirements": ["geotiff"],
        "required_images": 2,
        "permitted_parameters": [],
    },

    "change_map": {
        "base_url": os.getenv("CHANGE_MAP_ENDPOINT", ""),
        "input_requirements": ["geotiff"],
        "required_images": 2,
        "permitted_parameters": [],
    },
    
    "land_cover_classification": {
        "base_url": os.getenv("LAND_COVER_ENDPOINT", ""),
        "input_requirements": ["geotiff", "benchmark_rgb"],
        "required_images": 1,
        "permitted_parameters": [],
    },
}


def get_tool_status(task: str) -> str:

    entry = MODEL_REGISTRY.get(task)

    if not entry:
        return "UNKNOWN"

    if not entry["base_url"]:
        return "UNAVAILABLE"

    try:
        response = httpx.get(
            f"{entry['base_url']}/health",
            timeout=2.0,
        )

        if response.status_code == 200:
            return "AVAILABLE"

    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    return "UNAVAILABLE"
