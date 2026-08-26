import os
import glob
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from src.models.loader import load_blip_rsicd

def compare():
    print("--- COMPARING BASELINE VS ADAPTED ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Baseline
    print(f"Loading Base Model on {device}...")
    baseline_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    baseline_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    baseline_model.eval()
    
    # Load Adapted
    print(f"Loading Adapted Model...")
    adapted_processor, adapted_model, _ = load_blip_rsicd()
    
    # Get test images
    search_path = "datasets/rsicd/images/*.jpg"
    images = glob.glob(search_path)[:20]
    
    if not images:
        print("No images found for comparison.")
        return
        
    print(f"Comparing on {len(images)} images...\n")
    
    changed = 0
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        
        # Baseline Inference
        inputs = baseline_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out_base = baseline_model.generate(**inputs, max_new_tokens=30)
        cap_base = baseline_processor.decode(out_base[0], skip_special_tokens=True)
        
        # Adapted Inference
        inputs_adapt = adapted_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out_adapt = adapted_model.generate(**inputs_adapt, max_new_tokens=30)
        cap_adapt = adapted_processor.decode(out_adapt[0], skip_special_tokens=True)
        
        is_diff = (cap_base != cap_adapt)
        if is_diff: changed += 1
        
        print(f"Image: {os.path.basename(img_path)}")
        print(f"  Baseline: {cap_base}")
        print(f"  Adapted:  {cap_adapt}")
        print(f"  Changed:  {is_diff}\n")
        
    change_rate = (changed / len(images)) * 100
    print(f"Total Change Rate: {change_rate:.1f}% ({changed}/{len(images)})")
    
if __name__ == "__main__":
    compare()
