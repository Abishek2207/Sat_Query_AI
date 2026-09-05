import io
import gc
import os
import sys
import numpy as np
from PIL import Image
from typing import List, Dict, Any

_GLOBAL_MODELS = {}

def _get_available_memory_mb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        if hasattr(os, 'sysconf'):
            pages = os.sysconf('SC_AVPHYS_PAGES')
            page_size = os.sysconf('SC_PAGE_SIZE')
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024 * 1024)
        return 4096.0

def _enforce_memory_limit(required_mb: int, model_name: str):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    avail_mb = _get_available_memory_mb()
    if avail_mb < required_mb:
        raise MemoryError(f"Insufficient RAM. {model_name} requires ~{required_mb}MB but only {avail_mb:.1f}MB is available. (Render Free limit reached).")

def _free_memory(model_key: str = None):
    if model_key and model_key in _GLOBAL_MODELS:
        del _GLOBAL_MODELS[model_key]
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def _load_image(files_data: List[Dict]) -> Image.Image:
    try:
        return Image.open(io.BytesIO(files_data[0]["bytes"])).convert("RGB")
    except Exception:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(files_data[0]["bytes"]) as memfile:
            with memfile.open() as src:
                arr = src.read(1)
                arr_max = arr.max() if arr.max() > 0 else 1
                arr = (arr / arr_max * 255).astype(np.uint8)
                return Image.fromarray(arr)

def _build_evidence(claim: str, evidence: str, model: str, version: str, region: list = None, conf: float = None, status: str = "VERIFIED") -> Dict[str, Any]:
    return {
        "claim": claim,
        "evidence": evidence,
        "region": region,
        "timestamp": None,
        "modality": "optical_imagery",
        "confidence": conf,
        "confidence_type": "model-intrinsic" if conf is not None else None,
        "status": status,
        "source": "Local Specialist",
        "model": model,
        "model_version": version
    }

def run_local_vqa(query: str, files_data: List[Dict]) -> Dict[str, Any]:
    from datetime import datetime, timezone
    try:
        _enforce_memory_limit(800, "BLIP VQA")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if "vqa" not in _GLOBAL_MODELS:
            from transformers import BlipProcessor, BlipForQuestionAnswering
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
            _GLOBAL_MODELS["vqa"] = (processor, model, device)
            
        processor, model, device = _GLOBAL_MODELS["vqa"]
        
        image = _load_image(files_data)
        inputs = processor(image, query, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            out = model.generate(**inputs)
            
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        model_name = "Salesforce/blip-vqa-base"
        ev = _build_evidence(
            claim=f"Answer to '{query}' is '{answer}'",
            evidence=f"Inference performed locally on {device}.",
            model=model_name,
            version="1.0"
        )
        
        if _get_available_memory_mb() < 2000:
            _free_memory("vqa")
            
        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": None,
            "evidence": [ev],
            "provenance": {
                "model": model_name,
                "model_version": "1.0",
                "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "device": device,
                "input_filenames": [f["filename"] for f in files_data],
                "input_modalities": ["optical_imagery"],
                "remote_sensing_adapted": False,
                "geospatial_evidence_generated": False
            }
        }
    except MemoryError as me:
        return {"status": "BACKEND_RESOURCE_LIMIT", "answer": str(me), "evidence": []}
    except Exception as e:
        return {"status": "MODEL_UNAVAILABLE", "answer": f"Local inference failed: {str(e)}", "evidence": []}

def run_local_captioning(files_data: List[Dict], app_state=None) -> Dict[str, Any]:
    from datetime import datetime, timezone
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path: sys.path.append(project_root)
        
    try:
        _enforce_memory_limit(800, "BLIP Captioning")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if "captioning" not in _GLOBAL_MODELS:
            try:
                from src.models.loader import load_blip_rsicd
                adapter_path = os.path.join(project_root, "models", "rsicd_blip_lora", "adapter.pt")
                if os.path.exists(adapter_path):
                    processor, model, device = load_blip_rsicd(adapter_path=adapter_path)
                    adapter_loaded = True
                    _GLOBAL_MODELS["captioning"] = (processor, model, device, adapter_loaded)
                else:
                    raise FileNotFoundError
            except Exception:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
                model.eval()
                _GLOBAL_MODELS["captioning"] = (processor, model, device, False)
                
        processor, model, device, adapter_loaded = _GLOBAL_MODELS["captioning"]
            
        image = _load_image(files_data)
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
            
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        model_name = "Salesforce/blip-image-captioning-base"
        prov = {
            "model": model_name,
            "model_version": "1.0",
            "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "device": device,
            "input_filenames": [f["filename"] for f in files_data],
            "input_modalities": ["optical_imagery"],
            "adaptation_dataset": "arampacha/rsicd" if adapter_loaded else None,
            "remote_sensing_adapted": adapter_loaded,
            "geospatial_evidence_generated": False
        }
        if adapter_loaded:
            prov["adaptation_method"] = "Native PyTorch LoRA"
            prov["adapter_loaded"] = True
            
        ev = _build_evidence(
            claim=f"Image caption: {answer}",
            evidence=f"Caption generated locally on {device}. Adapter loaded: {adapter_loaded}",
            model=model_name,
            version="1.0"
        )
            
        if _get_available_memory_mb() < 2000:
            _free_memory("captioning")

        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": None,
            "evidence": [ev],
            "provenance": prov
        }
    except MemoryError as me:
        return {"status": "BACKEND_RESOURCE_LIMIT", "answer": str(me), "evidence": []}
    except Exception as e:
        return {"status": "MODEL_UNAVAILABLE", "answer": f"Local inference failed: {str(e)}", "evidence": []}

def run_local_grounding(query: str, files_data: List[Dict], confidence_threshold: float = 0.3) -> Dict[str, Any]:
    from datetime import datetime, timezone
    try:
        _enforce_memory_limit(1000, "GroundingDINO")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "IDEA-Research/grounding-dino-base"
        
        if "grounding" not in _GLOBAL_MODELS:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
            _GLOBAL_MODELS["grounding"] = (processor, model, device)
            
        processor, model, device = _GLOBAL_MODELS["grounding"]
        
        image = _load_image(files_data)
        text = (query if query.endswith(".") else query + ".").lower()
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
            
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=confidence_threshold, text_threshold=confidence_threshold, target_sizes=[image.size[::-1]]
        )
        
        evidence_items = []
        if len(results) > 0 and len(results[0]["scores"]) > 0:
            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
                box_arr = box.tolist()
                ev = _build_evidence(
                    claim=f"Found {label}",
                    evidence=f"Bounding box extracted for {label}.",
                    model=model_name,
                    version="1.0",
                    region=box_arr,
                    conf=float(score)
                )
                evidence_items.append(ev)
                
            answer = "Detected objects matching the query."
        else:
            answer = "No objects detected matching the query."
            
        if _get_available_memory_mb() < 2000:
            _free_memory("grounding")

        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": None,
            "evidence": evidence_items,
            "provenance": {
                "model": model_name,
                "model_version": "1.0",
                "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "device": device,
                "input_filenames": [f["filename"] for f in files_data],
                "input_modalities": ["optical_imagery"],
                "remote_sensing_adapted": False,
                "geospatial_evidence_generated": True
            }
        }
    except MemoryError as me:
        return {"status": "BACKEND_RESOURCE_LIMIT", "answer": str(me), "evidence": []}
    except Exception as e:
        return {"status": "MODEL_UNAVAILABLE", "answer": f"Local inference failed: {str(e)}", "evidence": []}

def run_local_land_cover(files_data: List[Dict]) -> Dict[str, Any]:
    from datetime import datetime, timezone
    try:
        _enforce_memory_limit(500, "EuroSAT ResNet") # Relatively small model, needs ~500MB
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "nielsr/convnext-tiny-finetuned-eurosat"
        
        if "land_cover" not in _GLOBAL_MODELS:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
            _GLOBAL_MODELS["land_cover"] = (processor, model, device)
            
        processor, model, device = _GLOBAL_MODELS["land_cover"]
        
        image = _load_image(files_data)
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
            
        predicted_class_idx = outputs.logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_idx]
        conf = float(torch.nn.functional.softmax(outputs.logits, dim=-1)[0, predicted_class_idx].item())
        
        answer = f"The primary land cover is {label}."
        
        ev = _build_evidence(
            claim=f"Classified as {label}",
            evidence=f"EuroSAT classification logic applied on {device}.",
            model=model_name,
            version="1.0",
            conf=conf
        )
        
        if _get_available_memory_mb() < 1000:
            _free_memory("land_cover")

        return {
            "status": "SUCCESS",
            "answer": answer,
            "confidence": conf,
            "evidence": [ev],
            "provenance": {
                "model": model_name,
                "model_version": "1.0",
                "inference_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "device": device,
                "input_filenames": [f["filename"] for f in files_data],
                "input_modalities": ["optical_imagery"],
                "remote_sensing_adapted": True,
                "geospatial_evidence_generated": False
            }
        }
    except MemoryError as me:
        return {"status": "BACKEND_RESOURCE_LIMIT", "answer": str(me), "evidence": []}
    except Exception as e:
        return {"status": "MODEL_UNAVAILABLE", "answer": f"Local inference failed: {str(e)}", "evidence": []}

