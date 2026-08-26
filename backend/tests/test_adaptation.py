import pytest
import os
import yaml
import json
from pathlib import Path

from backend.app.adaptation.provenance import save_provenance, load_provenance
from backend.app.adaptation.train import verify_data_availability, run_adaptation
from backend.app.adaptation.inference import is_model_adapted, get_provenance_info

def test_provenance_read_write(tmp_path):
    out_dir = tmp_path / "models"
    metadata = {"model": "test_model", "remote_sensing_adapted": True}
    
    # Save
    save_provenance(str(out_dir), metadata)
    assert (out_dir / "provenance.json").exists()
    
    # Load
    loaded = load_provenance(str(out_dir))
    assert loaded["model"] == "test_model"
    assert loaded["remote_sensing_adapted"] is True

def test_verify_data_availability_missing_dir(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"optical_patch_reference": "patch1"}\n')
    
    img_dir = tmp_path / "images" # doesn't exist
    assert not verify_data_availability(str(manifest), str(img_dir))

def test_verify_data_availability_missing_image(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"optical_patch_reference": "patch1"}\n')
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # patch1.tif is missing
    assert not verify_data_availability(str(manifest), str(img_dir))

def test_verify_data_availability_success(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"optical_patch_reference": "patch1"}\n')
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "patch1.tif").touch()
    
    assert verify_data_availability(str(manifest), str(img_dir))

def test_run_adaptation_data_required(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("status: CONFIGURED\n")
    
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"optical_patch_reference": "patch1"}\n')
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Run adaptation without the image present
    result = run_adaptation(str(config), str(manifest), str(img_dir))
    
    assert result["status"] == "DATA_REQUIRED"
    
    with open(config, "r") as f:
        new_cfg = yaml.safe_load(f)
        assert new_cfg["status"] == "DATA_REQUIRED"

def test_is_model_adapted_missing(tmp_path):
    out_dir = tmp_path / "models"
    assert not is_model_adapted(str(out_dir))
    
    prov = get_provenance_info(str(out_dir))
    assert prov["remote_sensing_adapted"] is False

def test_is_model_adapted_success(tmp_path):
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    save_provenance(str(out_dir), {"remote_sensing_adapted": True})
    
    # Must also have adapter weights
    (out_dir / "adapter_model.safetensors").touch()
    
    assert is_model_adapted(str(out_dir))
    
    prov = get_provenance_info(str(out_dir))
    assert prov["remote_sensing_adapted"] is True
