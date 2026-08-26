import os
import torch
from PIL import Image
from src.models.loader import load_blip_rsicd

class RSICDCaptioner:
    def __init__(self, adapter_path="models/rsicd_blip_lora/adapter.pt"):
        try:
            self.processor, self.model, self.device = load_blip_rsicd(adapter_path=adapter_path)
        except Exception as e:
            print(f"[Captioner] Critical error initializing model: {e}")
            raise

    def generate(self, image_path: str, max_new_tokens=30) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[Captioner] Image not found at {image_path}")
            
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                
            return self.processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[Captioner] Error during inference generation: {e}")
            raise
