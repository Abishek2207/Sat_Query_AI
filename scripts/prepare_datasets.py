import os
import json
import hashlib
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"

DATASET_CONFIGS = {
    "bigearthnet": {
        "requires_auth": False,
        "is_public": True,
        "url": "https://bigearth.net/downloads/",
        "directories": ["images", "annotations"]
    },
    "vrsbench": {
        "requires_auth": False,
        "is_public": True,
        "url": "https://vrsbench.github.io/",
        "directories": ["images", "annotations"]
    },
    "rsvqa_lr": {
        "requires_auth": False,
        "is_public": True,
        "url": "https://rsvqa.sylvainlobry.com/",
        "directories": ["images", "annotations"]
    },
    "rsvqa_hr": {
        "requires_auth": False,
        "is_public": True,
        "url": "https://rsvqa.sylvainlobry.com/",
        "directories": ["images", "annotations"]
    },
    "cdvqa": {
        "requires_auth": False,
        "is_public": True,
        "url": "Public Repository",
        "directories": ["images", "annotations"]
    },
    "isro_sac": {
        "requires_auth": True,
        "is_public": False,
        "url": "Restricted Target",
        "directories": ["images", "annotations"]
    },
    "rsicd": {
        "requires_auth": False,
        "is_public": True,
        "url": "https://github.com/201528014227051/RSICD_optimal",
        "directories": ["images", "annotations"]
    }
}

def verify_file_integrity(filepath: Path, expected_hash: str = None) -> bool:
    """Verifies SHA256 integrity of a file if hash is provided."""
    if not filepath.exists():
        return False
    if not expected_hash:
        return True # Cannot verify, assume ok if exists
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash

def initialize_directories(dataset_id: str, dirs: list):
    """Creates the exact required directory structure."""
    ds_path = DATASETS_DIR / dataset_id
    ds_path.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        (ds_path / d).mkdir(parents=True, exist_ok=True)

def run_preparation():
    print("--- SatQuery AI Dataset Preparation Pipeline ---")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    for ds_id, config in DATASET_CONFIGS.items():
        print(f"\nProcessing dataset: {ds_id}")
        initialize_directories(ds_id, config["directories"])
        
        # Check actual files
        img_dir = DATASETS_DIR / ds_id / "images"
        images_exist = len(list(img_dir.glob("*.*"))) > 0
        
        if config["requires_auth"] or not config["is_public"]:
            print(f"  [STATUS] MANUAL_DOWNLOAD_REQUIRED")
            print(f"  [REASON] Dataset is restricted or requires authentication.")
            print(f"  [ACTION] Please download from: {config['url']} into {img_dir}")
            continue
            
        if images_exist:
            print(f"  [STATUS] Data physically present.")
            # We explicitly do NOT generate fake manifests here. 
            # If manifests are missing, they must be manually downloaded/extracted from the source.
            manifest_path = DATASETS_DIR / ds_id / f"{ds_id}_manifest.jsonl"
            if manifest_path.exists():
                print(f"  [STATUS] Manifest present.")
            else:
                print(f"  [WARNING] Manifest missing. Do NOT generate synthetic annotations. Download official JSON.")
        else:
            print(f"  [STATUS] MISSING")
            print(f"  [ACTION] Download required from public source: {config['url']}")
            # Here we would invoke robust download scripts (e.g. wget/curl wrappers).
            # We do NOT pretend it downloaded.
            print(f"  [INFO] Automated download skipped. Execute manual sync.")

if __name__ == "__main__":
    run_preparation()
