import os
from pathlib import Path
from typing import Dict, Any, List

DATASETS_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))

def check_dataset_status(dataset_id: str) -> Dict[str, Any]:
    dataset_path = DATASETS_DIR / dataset_id
    if not dataset_path.exists():
        return {"status": "NOT_AVAILABLE", "samples": 0}
        
    # Check for actual images
    images_dir = dataset_path / "images"
    has_images = False
    image_count = 0
    if images_dir.exists() and images_dir.is_dir():
        image_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.tif', '.tiff'))])
        if image_count > 0:
            has_images = True

    # Check for manifests/ground truth
    has_manifest = (dataset_path / "adaptation_manifest.jsonl").exists() or \
                   (dataset_path / f"{dataset_id}_manifest.jsonl").exists() or \
                   list(dataset_path.glob("*.parquet"))

    if has_images and has_manifest:
        return {"status": "READY", "samples": image_count}
    elif has_manifest and not has_images:
        return {"status": "PARTIAL", "samples": 0, "reason": "IMAGE PATCHES REQUIRED"}
    elif has_images and not has_manifest:
        return {"status": "PARTIAL", "samples": image_count, "reason": "ANNOTATIONS MISSING"}
         
    return {"status": "NOT_AVAILABLE", "samples": 0}

def get_dataset_registry() -> List[Dict[str, Any]]:
    registry = [
        {
            "dataset_id": "bigearthnet",
            "name": "BigEarthNet",
            "source": "https://bigearth.net/",
            "task_types": ["classification", "captioning"],
            "image_modality": ["multispectral", "sar"]
        },
        {
            "dataset_id": "vrsbench",
            "name": "VRSBench",
            "source": "https://vrsbench.github.io/",
            "task_types": ["vqa", "captioning", "grounding"],
            "image_modality": ["optical"]
        },
        {
            "dataset_id": "rsvqa_lr",
            "name": "RSVQA-LR",
            "source": "https://rsvqa.sylvainlobry.com/",
            "task_types": ["vqa"],
            "image_modality": ["optical"]
        },
        {
            "dataset_id": "rsvqa_hr",
            "name": "RSVQA-HR",
            "source": "https://rsvqa.sylvainlobry.com/",
            "task_types": ["vqa"],
            "image_modality": ["optical"]
        },
        {
            "dataset_id": "cdvqa",
            "name": "CDVQA",
            "source": "Public Repository",
            "task_types": ["change_vqa"],
            "image_modality": ["optical"]
        },
        {
            "dataset_id": "isro_sac",
            "name": "ISRO/SAC Evaluation",
            "source": "Restricted",
            "task_types": ["optical_sar", "vqa"],
            "image_modality": ["optical", "sar"]
        },
        {
            "dataset_id": "rsicd",
            "name": "RSICD",
            "source": "https://github.com/201528014227051/RSICD_optimal",
            "task_types": ["captioning"],
            "image_modality": ["optical"]
        }
    ]
    
    for ds in registry:
        status_info = check_dataset_status(ds["dataset_id"])
        ds["status"] = status_info["status"]
        ds["samples"] = status_info["samples"]
        if "reason" in status_info:
            ds["reason"] = status_info["reason"]
        
    return registry
