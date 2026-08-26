import json
import os
import sys
from io import BytesIO
from datasets import load_dataset
from huggingface_hub import dataset_info
from PIL import Image

def verify():
    repo_id = "arampacha/rsicd"
    print(f"--- Verifying {repo_id} ---")
    
    # 1. Inspect repository metadata
    try:
        info = dataset_info(repo_id)
        print("Repository accessible.")
        print(f"Tags: {info.tags}")
    except Exception as e:
        print(f"Failed to fetch metadata: {e}")

    # 2. Test streaming 3 real samples
    print("\n--- Streaming 3 real samples ---")
    try:
        ds = load_dataset(repo_id, split="train", streaming=True)
        count = 0
        samples = []
        for sample in ds:
            if count >= 3:
                break
                
            # Extract fields
            img = sample.get("image")
            caption = sample.get("captions")
            filename = sample.get("filename", "unknown")
            
            img_dims = "Unknown"
            if isinstance(img, Image.Image):
                img_dims = f"{img.width}x{img.height}"
            
            samples.append({
                "id": filename,
                "dimensions": img_dims,
                "caption": caption[:2] if isinstance(caption, list) else caption, # print first 2
                "split": "train"
            })
            count += 1
            
        print(json.dumps(samples, indent=2))
        
        if count == 3:
            print("\nSUCCESS: Retrieved exactly 3 samples efficiently.")
        else:
            print(f"\nFAILED: Only retrieved {count} samples.")
            
    except Exception as e:
        print(f"Failed to stream dataset: {e}")

    # 3. Test Loading Model (Without Training)
    print("\n--- Testing Salesforce/blip-image-captioning-base Loading ---")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Target device: {device}")
        
        print("Loading processor...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        print("Loading model...")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        
        print("SUCCESS: Model and processor loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")

if __name__ == "__main__":
    verify()
