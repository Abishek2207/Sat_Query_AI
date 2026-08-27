import os
import sys
import torch
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.loader import load_blip_rsicd
from backend.app.main import app
from fastapi.testclient import TestClient

def verify():
    results = {}
    
    # Phase 2
    adapter_path = "models/rsicd_blip_lora/adapter.pt"
    results['adapter_exists'] = os.path.exists(adapter_path)
    if results['adapter_exists']:
        results['adapter_size'] = os.path.getsize(adapter_path)
        sd = torch.load(adapter_path, map_location="cpu", weights_only=True)
        results['tensor_count'] = len(sd)
        results['tensor_names'] = list(sd.keys())[:3] # sample
        results['total_params'] = sum(p.numel() for p in sd.values())
        results['is_lora'] = all('lora' in k for k in sd.keys())
    
    # Phase 3, 4, 5
    device = "cpu"
    base_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    base_model.eval()
    
    img_path = "datasets/rsicd/images/image_0.jpg"
    image = Image.open(img_path).convert("RGB")
    inputs = base_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_base = base_model.generate(**inputs, max_new_tokens=30)
    results['baseline_caption'] = base_processor.decode(out_base[0], skip_special_tokens=True)
    
    # Load LoRA
    adapt_processor, adapt_model, _ = load_blip_rsicd()
    inputs_adapt = adapt_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out_adapt = adapt_model.generate(**inputs_adapt, max_new_tokens=30)
    results['adapted_caption'] = adapt_processor.decode(out_adapt[0], skip_special_tokens=True)
    results['prediction_changed'] = (results['baseline_caption'] != results['adapted_caption'])
    
    # Phase 6
    client = TestClient(app)
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    
    response = client.post(
        "/analyze",
        data={"query": "caption", "capability": "captioning", "benchmark_mode": "true"},
        files=[("files", ("image_0.jpg", file_bytes, "image/jpeg"))]
    )
    results['backend_status'] = response.status_code
    if response.status_code == 200:
        rj = response.json()
        results['backend_caption'] = rj.get("answer")
    else:
        results['backend_caption'] = response.text
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    verify()
