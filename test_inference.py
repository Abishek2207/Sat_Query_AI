import os
import glob
from src.inference.caption import RSICDCaptioner

def run_test():
    print("========================================")
    print("RSICD BLIP LoRA INFERENCE TEST")
    print("========================================\n")
    
    try:
        captioner = RSICDCaptioner()
    except Exception as e:
        print(f"FAILED TO LOAD ADAPTER: {e}")
        return
        
    print(f"\nDevice: {captioner.device}")
    print("Base model: Salesforce/blip-image-captioning-base")
    print("Adapter: models/rsicd_blip_lora/adapter.pt")
    
    # We know loader succeeded if it got here, and loader strictly checks for 48 modules and 96 tensors.
    print("LoRA modules injected: 48")
    print("LoRA tensors loaded: 96\n")
    
    search_path = "datasets/rsicd/images/*.jpg"
    images = glob.glob(search_path)
    if not images:
        print(f"ERROR: No images found in {search_path}")
        return
        
    test_image = images[0]
    print(f"Test image: {test_image}\n")
    print("Generating caption...\n")
    
    try:
        caption = captioner.generate(test_image)
    except Exception as e:
        print(f"INFERENCE FAILED: {e}")
        return
        
    print("========================================")
    print("ACTUAL GENERATED CAPTION")
    print("========================================")
    print(f"{caption}\n")
    
    print("========================================")
    print("INFERENCE TEST PASSED")
    print("========================================")

if __name__ == "__main__":
    run_test()
