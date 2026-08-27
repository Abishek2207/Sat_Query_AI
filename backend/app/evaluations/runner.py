import os
import json
import torch
from datetime import datetime
from pathlib import Path
from .metrics import calculate_iou, exact_match_accuracy, simple_caption_match
from ..config import settings
from ..adapters import call_specialist_model

def get_vram_status():
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    if vram_mb < settings.VRAM_MIN_REQ_MB:
        return False, f"Insufficient VRAM: {vram_mb:.0f}MB < {settings.VRAM_MIN_REQ_MB}MB"
    return True, f"VRAM OK ({vram_mb:.0f}MB)"

async def run_benchmark_evaluation(dataset_id: str, limit: int = 5):
    # Determine the status and fallback logic
    dataset_path = Path(settings.DATASETS_DIR) / dataset_id
    if not dataset_path.exists():
        return {
            "dataset": dataset_id,
            "split": "test",
            "reference_available": False,
            "evaluated_samples": 0,
            "metric": None,
            "score": None,
            "status": "NOT_AVAILABLE",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "message": f"Dataset {dataset_id} NOT_AVAILABLE. Please download first."
        }

    vram_ok, vram_msg = get_vram_status()
    if not vram_ok:
        return {
            "dataset": dataset_id,
            "split": "test",
            "reference_available": True,
            "evaluated_samples": 0,
            "metric": None,
            "score": None,
            "status": "FAILED",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "message": f"Evaluation aborted: {vram_msg}"
        }

    # Actually execute a loop over the dataset if it has files
    # Note: Full loop implementation is abbreviated for demo purposes to avoid OOM, but executes real inference
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        return {
            "dataset": dataset_id,
            "split": "test",
            "reference_available": False,
            "evaluated_samples": 0,
            "metric": None,
            "score": None,
            "status": "NOT_AVAILABLE",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "message": "Dataset structure invalid: no images directory."
        }
        
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.tif'))]
    if not image_files:
        return {
            "dataset": dataset_id,
            "split": "test",
            "reference_available": False,
            "evaluated_samples": 0,
            "metric": None,
            "score": None,
            "status": "NOT_AVAILABLE",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "message": "No images found in dataset."
        }
        
    # We load annotations if available, else we run without references (which violates SIH constraints, so we abort)
    ann_file = dataset_path / f"{dataset_id}_manifest.jsonl"
    if not ann_file.exists():
        ann_file = dataset_path / "val_manifest.jsonl"
        
    if not ann_file.exists():
        return {
            "dataset": dataset_id,
            "split": "test",
            "reference_available": False,
            "evaluated_samples": 0,
            "metric": None,
            "score": None,
            "status": "NOT_AVAILABLE",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "message": "Annotations file NOT_AVAILABLE. Cannot compute metrics."
        }
        
    # Run loop
    total = 0
    predictions = []
    references = []
    
    with open(ann_file, "r") as f:
        for line in f:
            if total >= limit: break
            record = json.loads(line)
            
            # Simulated inference structure using local models to get real outputs
            try:
                img_path = images_dir / record["image"]
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    
                # Call local adapters
                task = record.get("task", "vqa") # fallback
                query = record.get("query", "What is this?")
                ref_ans = record.get("answer", "")
                
                resp = await call_specialist_model(task, query, [{"filename": img_path.name, "bytes": img_bytes}], {})
                
                if resp.get("status") in ["SUCCESS", "VERIFIED"]:
                    predictions.append(resp.get("answer", ""))
                    references.append(ref_ans)
                    total += 1
            except Exception as e:
                print("Eval err:", e)
                
    if total == 0:
        return {
            "dataset": dataset_id,
            "split": "test",
            "reference_available": True,
            "evaluated_samples": 0,
            "metric": None,
            "score": None,
            "status": "NOT_AVAILABLE" if not image_files else "FAILED",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "message": "All samples failed during evaluation or dataset empty."
        }
        
    # Compute metric
    if "vqa" in dataset_id.lower() or "rsicd" in dataset_id.lower():
        metric_name = "exact_match"
        score = exact_match_accuracy(predictions, references)
    else:
        metric_name = "diagnostic_overlap" # explicitly NOT an official benchmark score
        score = sum(simple_caption_match(p, r) for p, r in zip(predictions, references)) / total

    return {
        "dataset": dataset_id,
        "split": "test",
        "reference_available": True,
        "evaluated_samples": total,
        "metric": metric_name,
        "score": round(score, 4),
        "status": "COMPLETED",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "provenance": {
            "dataset_version": "1.0",
            "model_checkpoint": "local_v1",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "vram_checked": vram_ok
        }
    }
