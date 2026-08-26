import pytest
import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pathlib import Path

from backend.app.dataset.prepare_adaptation import prepare_subset

@pytest.fixture
def synthetic_parquet_for_prep(tmp_path):
    # Create a synthetic parquet file
    data = {
        'ID': list(range(1, 101)),
        's1_name': [f'S1_{i}' for i in range(1, 101)],
        'patch_id': [f'S2_{i}' for i in range(1, 101)],
        'input': ['Q' * i for i in range(1, 101)],
        'output': ['A' * i for i in range(1, 101)],
        'type': ['binary'] * 100,
        'category': ['land_cover'] * 100,
        'split': ['train'] * 80 + ['val'] * 10 + ['test'] * 10,
        'latitude': [45.1] * 100,
        'longitude': [10.1] * 100,
        'country': ['France'] * 100,
        'season': ['Summer'] * 100,
        'climate_zone': ['Temperate'] * 100
    }
    table = pa.table(data)
    file_path = tmp_path / "synthetic_data.parquet"
    pq.write_table(table, file_path)
    return str(file_path)

def test_prepare_subset(synthetic_parquet_for_prep, tmp_path):
    out_path = tmp_path / "manifest.jsonl"
    
    # Extract 10 records
    records = prepare_subset(synthetic_parquet_for_prep, str(out_path), sample_size=10, seed=42)
    
    assert len(records) == 10
    assert out_path.exists()
    
    # Check the schema of the manifest
    rec = records[0]
    assert "ID" in rec
    assert "question" in rec
    assert "answer" in rec
    assert "sar_reference" in rec
    assert "optical_patch_reference" in rec
    assert "category" in rec
    assert "split" in rec
    assert "geographic_metadata" in rec
    assert "latitude" in rec["geographic_metadata"]
