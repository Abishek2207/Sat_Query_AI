import gc
import io
import torch
from typing import List, Dict, Any
from datetime import datetime, timezone
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Global variables for lazy loading
_processor = None
_model = None
_device = None

def _load_model():
    global _processor, _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "nielsr/convnext-tiny-finetuned-eurosat"
        _processor = AutoImageProcessor.from_pretrained(model_name)
        _model = AutoModelForImageClassification.from_pretrained(model_name).to(_device)
        _model.eval()

def run_eurosat_classification(files_data: List[Dict]) -> Dict[str, Any]:
    try:
        if not files_data or not files_data[0].get("bytes"):
            return {
                "status": "INVALID_INPUT",
                "answer": "No image data provided for EuroSAT classification.",
                "evidence": ["Image bytes missing."]
            }

        _load_model()
        
        # Parse image
        img_bytes = files_data[0]["bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Preprocess
        inputs = _processor(images=image, return_tensors="pt").to(_device)
        
        with torch.inference_mode():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
        # Get top predictions
        top_k = min(3, len(probs))
        top_probs, top_indices = torch.topk(probs, top_k)
        
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        best_class = _model.config.id2label[top_indices[0]]
        best_conf = float(top_probs[0])
        
        # Formulate Evidence
        evidence = [
            f"EuroSAT classifier predicted {best_class} with {int(best_conf * 100)}% confidence."
        ]
        
        labels = ["Top", "Second", "Third"]
        for i in range(top_k):
            class_name = _model.config.id2label[top_indices[i]]
            evidence.append(f"{labels[i]} prediction: {class_name}")

        answer = f"The image is classified as {best_class}."
        
        prov = {
            "model": "nielsr/convnext-tiny-finetuned-eurosat",
            "model_version": "1.0",
            "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_filenames": [f["filename"] for f in files_data],
            "input_modalities": ["optical_imagery"],
            "adaptation_dataset": "EuroSAT",
            "remote_sensing_adapted": True,
            "geospatial_evidence_generated": False,
            "device": _device
        }

        # Cleanup memory for CPU
        del inputs, outputs, logits, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": best_conf,
            "evidence": evidence,
            "provenance": prov
        }
        
    except Exception as e:
        return {
            "status": "MODEL_UNAVAILABLE",
            "answer": f"EuroSAT classification failed: {str(e)}",
            "evidence": [f"Error: {str(e)}"]
        }
