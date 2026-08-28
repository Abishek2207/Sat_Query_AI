import os
from pathlib import Path

DATASETS_DIR = Path("datasets")
DATASETS = ["bigearthnet", "vrsbench", "rsvqa_lr", "rsvqa_hr", "cdvqa", "isro_sac", "rsicd"]

print("Dataset | Physically Available | Actual Samples | Ground Truth | Status")
print("-" * 80)
for ds in DATASETS:
    path = DATASETS_DIR / ds
    images_dir = path / "images"
    
    available = "Yes" if path.exists() else "No"
    samples = 0
    gt = "No"
    status = "NOT_AVAILABLE"
    
    if path.exists():
        if images_dir.exists():
            samples = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.tif'))])
        
        has_manifest = any(f.endswith(".jsonl") for f in os.listdir(path) if os.path.isfile(path / f))
        if has_manifest:
            gt = "Yes"
            
        if samples > 0 and has_manifest:
            status = "READY"
        elif has_manifest:
            status = "PARTIAL"
            
    print(f"{ds:15} | {available:20} | {samples:<14} | {gt:<12} | {status}")
