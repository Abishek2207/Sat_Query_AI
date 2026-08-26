import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pathlib import Path
from backend.app.dataset.loader import BigEarthNetLoader

@pytest.fixture
def synthetic_bigearthnet_parquet(tmp_path):
    """Creates a tiny synthetic Parquet file matching the exact verified schema."""
    data = {
        'ID': [1, 2],
        's1_name': ['S1A_IW_GRDH_1', 'S1A_IW_GRDH_2'],
        'patch_id': ['S2A_MSIL2A_1', 'S2A_MSIL2A_2'],
        'input': ['Is there an urban area?', 'What is the land cover?'],
        'output': ['Yes', 'Forest'],
        'type': ['binary', 'open'],
        'category': ['land_cover', 'land_cover'],
        'split': ['train', 'val'],
        'latitude': [45.1, 45.2],
        'longitude': [10.1, 10.2],
        'country': ['France', 'France'],
        'season': ['Summer', 'Winter'],
        'climate_zone': ['Temperate', 'Temperate']
    }
    table = pa.table(data)
    
    dataset_dir = tmp_path / "datasets" / "bigearthnet"
    dataset_dir.mkdir(parents=True)
    file_path = dataset_dir / "BigEarthNet.txt.parquet"
    
    pq.write_table(table, file_path)
    
    config_path = dataset_dir / "config.yaml"
    config = {
        "dataset": {
            "path": str(file_path),
            "sample_size": 2,
            "validation_split": 0.2,
            "random_seed": 42,
            "batch_size": 1
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    return str(config_path)

def test_lazy_initialization_and_schema(synthetic_bigearthnet_parquet):
    loader = BigEarthNetLoader(synthetic_bigearthnet_parquet)
    schema = loader.get_schema()
    assert 'patch_id' in schema
    assert loader.get_total_rows() == 2

def test_stream_split_and_semantic_mapping(synthetic_bigearthnet_parquet):
    loader = BigEarthNetLoader(synthetic_bigearthnet_parquet)
    train_records = list(loader.stream_split("train"))
    
    assert len(train_records) == 1
    assert train_records[0]['question'] == 'Is there an urban area?'
    assert train_records[0]['answer'] == 'Yes'
    assert train_records[0]['sar_reference'] == 'S1A_IW_GRDH_1'
    assert train_records[0]['optical_patch_reference'] == 'S2A_MSIL2A_1'
    assert 'caption' not in train_records[0]

def test_sample_records(synthetic_bigearthnet_parquet):
    loader = BigEarthNetLoader(synthetic_bigearthnet_parquet)
    samples = loader.sample_records(1)
    assert len(samples) == 1
    assert "question" in samples[0]
