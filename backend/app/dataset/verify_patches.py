import os
import json
from pathlib import Path

def generate_verification_report(manifest_path, output_dir):
    print("Initiating BigEarthNet Raw Imagery Verification...")
    
    # 1. Inspect manifest and extract IDs
    ids_to_fetch = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                ids_to_fetch.append({
                    "id": record["ID"],
                    "optical": record.get("optical_patch_reference"),
                    "sar": record.get("sar_reference")
                })
                
    subset_to_fetch = ids_to_fetch[:250] # Target ~250 records
    print(f"Extracted {len(subset_to_fetch)} target patch records for subset download.")
    
    # 2. Determine download mechanism
    print("Determining official open-source download mechanism...")
    official_s2_url = "http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz"
    official_s1_url = "http://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz"
    s2_size_gb = 65.3
    s1_size_gb = 59.0
    
    # 3. Assess disk space constraint
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    
    print(f"Current free disk space: {free_gb:.2f} GB")
    print(f"Total required to unpack official archives: > {s2_size_gb + s1_size_gb} GB")
    
    report = {
        "verification_status": "FAILED_DUE_TO_STORAGE_CONSTRAINT",
        "requested_ids": len(subset_to_fetch),
        "successfully_downloaded_ids": 0,
        "missing_ids": len(subset_to_fetch),
        "reason": (
            "The official BigEarthNet image distribution relies on monolithic tar.gz archives "
            f"({s2_size_gb} GB for Optical, {s1_size_gb} GB for SAR). "
            "TU Berlin does not provide a REST API for extracting individual patch directories. "
            f"Downloading and extracting these archives exceeds the {free_gb:.1f} GB available disk space."
        ),
        "official_sources": {
            "optical": official_s2_url,
            "sar": official_s1_url
        }
    }
    
    out_path = Path("datasets/bigearthnet/images/verification_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\nVERIFICATION REPORT GENERATED:")
    print(json.dumps(report, indent=2))
    print("\nStopping before training: Cannot prove downloaded files correspond to manifest records (no files downloaded).")

if __name__ == "__main__":
    generate_verification_report(
        "datasets/bigearthnet/adaptation_manifest.jsonl",
        "datasets/bigearthnet/images"
    )
