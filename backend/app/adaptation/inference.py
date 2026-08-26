"""
backend/app/adaptation/inference.py

Helpers for loading the adapted model during inference.
"""
from pathlib import Path
from backend.app.adaptation.provenance import load_provenance

def is_model_adapted(output_dir: str = "models/rs_adapted_vqa") -> bool:
    """Checks if a valid, successfully trained adapter exists."""
    prov = load_provenance(output_dir)
    if prov and prov.get("remote_sensing_adapted") is True:
        # Verify adapter weights actually exist
        if (Path(output_dir) / "adapter_model.bin").exists() or (Path(output_dir) / "adapter_model.safetensors").exists():
            return True
    return False

def get_provenance_info(output_dir: str = "models/rs_adapted_vqa") -> dict:
    """Returns provenance metadata if adapted, otherwise returns default unadapted info."""
    prov = load_provenance(output_dir)
    if prov and is_model_adapted(output_dir):
        return prov
        
    return {
        "model": "Salesforce/blip-vqa-base",
        "model_version": "baseline",
        "adaptation_dataset": None,
        "adaptation_method": None,
        "remote_sensing_adapted": False,
    }
    
def load_specialist_model(output_dir: str = "models/rs_adapted_vqa"):
    """
    Loads the LoRA adapted model for inference if it exists.
    Throws exception if not found or dependencies missing.
    """
    if not is_model_adapted(output_dir):
        raise FileNotFoundError(f"No fully trained adapter found at {output_dir}")
        
    from transformers import AutoProcessor
    from peft import AutoPeftModelForCausalLM
    
    processor = AutoProcessor.from_pretrained(output_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
    return processor, model
