"""
backend/app/local_specialists.py

Attempts to load and run actual Hugging Face models locally if endpoints are missing.
Fails gracefully to MODEL_UNAVAILABLE if RAM/dependencies are insufficient.
"""
import io
import torch
import gc
from PIL import Image
from typing import Dict, Any, List

def _load_image(files_data: List[Dict]) -> Image.Image:
    img_bytes = files_data[0]["bytes"]
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return image
    except Exception:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(img_bytes) as mem:
            with mem.open() as src:
                count = src.count
                if count >= 3:
                    arr = src.read([1, 2, 3])
                else:
                    arr = src.read([1, 1, 1])
                import numpy as np
                arr = np.transpose(arr, (1, 2, 0))
                arr_max = arr.max() if arr.max() > 0 else 1
                arr = (arr / arr_max * 255).astype(np.uint8)
                return Image.fromarray(arr)

def run_local_vqa(query: str, files_data: List[Dict]) -> Dict[str, Any]:
    from datetime import datetime, timezone
    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
        
        image = _load_image(files_data)
        
        inputs = processor(image, query, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            out = model.generate(**inputs)
            
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        # Cleanup memory
        del model
        del processor
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": None,
            "evidence": ["Inference performed locally using Salesforce/blip-vqa-base on " + device],
            "provenance": {
                "model": "Salesforce/blip-vqa-base",
                "model_version": "1.0",
                "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "device": device if "device" in globals() or "device" in locals() else "cpu", "input_filenames": [f["filename"] for f in files_data],
                "input_modalities": ["optical_imagery"],
                "adaptation_dataset": None,
                "remote_sensing_adapted": False,
                "geospatial_evidence_generated": False
            }
        }
    except Exception as e:
        return {
            "status": "MODEL_UNAVAILABLE",
            "answer": f"Local inference failed: {str(e)}",
            "evidence": ["Hardware/Dependency error preventing local execution."]
        }

def run_local_captioning(files_data: List[Dict]) -> Dict[str, Any]:
    import os
    from datetime import datetime, timezone
    import sys
    
    # Ensure src is in path so loader works from anywhere
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    try:
        from src.models.loader import load_blip_rsicd
        
        adapter_path = os.path.join(project_root, "models", "rsicd_blip_lora", "adapter.pt")
        adapter_loaded = False
        
        if os.path.exists(adapter_path):
            processor, model, device = load_blip_rsicd(adapter_path=adapter_path)
            adapter_loaded = True
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            model.eval()
            
        image = _load_image(files_data)
        
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=30)
            
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        # Cleanup memory
        del model
        del processor
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        prov = {
            "model": "Salesforce/blip-image-captioning-base",
            "model_version": "1.0",
            "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "device": device if "device" in globals() or "device" in locals() else "cpu", "input_filenames": [f["filename"] for f in files_data],
            "input_modalities": ["optical_imagery"],
            "adaptation_dataset": "arampacha/rsicd" if adapter_loaded else None,
            "remote_sensing_adapted": adapter_loaded,
            "geospatial_evidence_generated": False
        }
        
        if adapter_loaded:
            prov["adaptation_method"] = "Native PyTorch LoRA"
            prov["adapter_loaded"] = True
            
        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": None,
            "evidence": [f"Caption generated locally using BLIP on {device}. Adapter loaded: {adapter_loaded}"],
            "provenance": prov
        }
    except Exception as e:
        return {
            "status": "MODEL_UNAVAILABLE",
            "answer": f"Local inference failed: {str(e)}",
            "evidence": ["Hardware/Dependency error preventing local execution."]
        }

def run_local_grounding(query: str, files_data: List[Dict], confidence_threshold: float = 0.3) -> Dict[str, Any]:
    from datetime import datetime, timezone
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
        
        image = _load_image(files_data)
        
        # Format query for grounding dino (lowercase, trailing dot)
        text = (query if query.endswith(".") else query + ".").lower()
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
            
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        answer = "Detected objects: "
        detections = []
        if len(results) > 0 and len(results[0]["scores"]) > 0:
            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
                detections.append(f"{label} ({score:.2f}) at {box.tolist()}")
            answer += ", ".join(detections)
        else:
            answer = "No objects detected matching the query."
            
        # Cleanup memory
        del model
        del processor
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": None,
            "evidence": ["Inference performed locally using IDEA-Research/grounding-dino-base on " + device],
            "provenance": {
                "model": "IDEA-Research/grounding-dino-base",
                "model_version": "1.0",
                "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "device": device if "device" in globals() or "device" in locals() else "cpu", "input_filenames": [f["filename"] for f in files_data],
                "input_modalities": ["optical_imagery"],
                "adaptation_dataset": None,
                "remote_sensing_adapted": False,
                "geospatial_evidence_generated": True
            }
        }
    except Exception as e:
        return {
            "status": "MODEL_UNAVAILABLE",
            "answer": f"Local inference failed: {str(e)}",
            "evidence": ["Hardware/Dependency error preventing local execution."]
        }
