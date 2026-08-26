"""
backend/app/dataset/prepare_adaptation.py

Prepares a reproducible subset of the BigEarthNet dataset for adaptation.
Ensures we do not load all 9.5M rows into RAM.
Generates an adaptation manifest.
"""
import pyarrow.parquet as pq
import pandas as pd
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List

def prepare_subset(parquet_path: str, output_manifest_path: str, sample_size: int = 1000, seed: int = 42):
    """
    Streams the parquet file and extracts a subset of records across splits,
    writing them into a JSONL manifest.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {parquet_path}")

    pf = pq.ParquetFile(path)
    
    # We will sample roughly evenly across train, val, test if possible, or just take the first N matching some random state.
    # To keep it memory efficient, we can iterate over batches, sample a fraction, and stop when we have enough.
    
    records = []
    
    # We just want to extract up to `sample_size` total records
    for batch in pf.iter_batches(batch_size=10000):
        df = batch.to_pandas()
        # Take a random sample from this batch
        n_sample = min(sample_size - len(records), int(len(df) * 0.1) + 1) # take 10% of batch
        if n_sample <= 0:
            continue
            
        sampled = df.sample(n=n_sample, random_state=seed)
        for _, row in sampled.iterrows():
            record = {
                "ID": int(row.get("ID")),
                "question": str(row.get("input")),
                "answer": str(row.get("output")),
                "sar_reference": str(row.get("s1_name")),
                "optical_patch_reference": str(row.get("patch_id")),
                "category": str(row.get("category")),
                "split": str(row.get("split")),
                "geographic_metadata": {
                    "latitude": float(row.get("latitude", 0.0)),
                    "longitude": float(row.get("longitude", 0.0)),
                    "country": str(row.get("country")),
                    "season": str(row.get("season")),
                    "climate_zone": str(row.get("climate_zone"))
                }
            }
            records.append(record)
            
        if len(records) >= sample_size:
            break
            
    # Write manifest
    out_path = Path(output_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records[:sample_size]:
            f.write(json.dumps(r) + "\n")
            
    print(f"Successfully generated adaptation manifest with {len(records[:sample_size])} records at {output_manifest_path}")
    return records[:sample_size]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare adaptation manifest")
    parser.add_argument("--config", type=str, default="datasets/bigearthnet/adaptation_config.yaml", help="Path to adaptation config")
    parser.add_argument("--output", type=str, default="datasets/bigearthnet/adaptation_manifest.jsonl", help="Output manifest path")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_path = config.get("adaptation_dataset", "datasets/bigearthnet/BigEarthNet.txt.parquet")
    sample_size = config.get("sample_size", 1000)
    
    prepare_subset(dataset_path, args.output, sample_size=sample_size)
