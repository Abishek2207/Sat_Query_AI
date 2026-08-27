import os
import json
import hashlib
from pathlib import Path

# Paths configuration
DATASETS_DIR = os.getenv("DATASETS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets")))

DATASETS = {
    "vrsbench": ["images", "vrsbench_manifest.jsonl"],
    "rsvqa_lr": ["images", "rsvqa_lr_manifest.jsonl"],
    "rsvqa_hr": ["images", "rsvqa_hr_manifest.jsonl"],
    "cdvqa": ["images", "cdvqa_manifest.jsonl"],
    "isro_sac": ["images", "isro_sac_manifest.jsonl"],
}

def generate_checksum(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def prepare_datasets():
    print(f"Preparing datasets directory at: {DATASETS_DIR}")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    for ds_id, structure in DATASETS.items():
        ds_path = Path(DATASETS_DIR) / ds_id
        os.makedirs(ds_path, exist_ok=True)
        
        for item in structure:
            item_path = ds_path / item
            if "images" in item:
                os.makedirs(item_path, exist_ok=True)
            elif "manifest" in item:
                # We will not create fake dataset records.
                # If it doesn't exist, we leave it empty.
                if not item_path.exists():
                    print(f"[{ds_id}] Missing official manifest {item}. (Requires manual download).")
                    
    print("Dataset directory structure verified. Missing data must be downloaded manually from official sources.")

if __name__ == "__main__":
    prepare_datasets()
