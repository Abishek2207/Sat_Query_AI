import os
import json
from pathlib import Path
from typing import Dict, Any, List

DATASETS_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))

def check_dataset_status(dataset_id: str) -> Dict[str, Any]:
    dataset_path = DATASETS_DIR / dataset_id
    if not dataset_path.exists():
        return {"status": "NOT_AVAILABLE", "samples": 0}
        
    # Check if images directory exists and has files
    images_dir = dataset_path / "images"
    if images_dir.exists() and images_dir.is_dir():
        try:
            samples = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.tif'))])
            if samples > 0:
                return {"status": "READY", "samples": samples}
        except:
            pass
            
    # For text/manifest only datasets
    if (dataset_path / "adaptation_manifest.jsonl").exists() or (dataset_path / f"{dataset_id}_manifest.jsonl").exists():
         return {"status": "READY", "samples": -1}
         
    return {"status": "NOT_AVAILABLE", "samples": 0}

def get_dataset_registry() -> List[Dict[str, Any]]:
    registry = [
        {
            "dataset_id": "bigearthnet",
            "name": "BigEarthNet.txt",
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
        }
    ]
    
    for ds in registry:
        status_info = check_dataset_status(ds["dataset_id"])
        ds["status"] = status_info["status"]
        ds["samples"] = status_info["samples"]
        
    return registry

# Evaluation runner logic has moved to backend/app/evaluations/runner.py
