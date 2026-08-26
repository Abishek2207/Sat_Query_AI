import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image
import json

def test_reload():
    print("--- Adapter Reload Test ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Load LoRA Adapter
    adapter_path = "models/rsicd_blip_lora"
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    
    # Run on a real image
    manifest = []
    with open("datasets/rsicd/val_manifest.jsonl", "r") as f:
        for line in f:
            manifest.append(json.loads(line))
            
    test_img = manifest[0]
    img = Image.open(test_img["filename"]).convert("RGB")
    
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    
    pred = processor.decode(out[0], skip_special_tokens=True)
    
    print("RELOAD SUCCESSFUL")
    print(f"Image: {test_img['filename']}")
    print(f"Reference: {test_img['caption']}")
    print(f"Prediction: {pred}")

if __name__ == "__main__":
    test_reload()
