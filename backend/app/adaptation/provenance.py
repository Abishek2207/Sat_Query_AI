"""
backend/app/adaptation/provenance.py

Handles saving and loading the provenance metadata for the adapted model.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

def load_provenance(output_dir: str) -> Optional[Dict[str, Any]]:
    """Loads provenance.json if it exists, else returns None."""
    path = Path(output_dir) / "provenance.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_provenance(output_dir: str, metadata: Dict[str, Any]):
    """Saves provenance.json into the given output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_path = out_path / "provenance.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
