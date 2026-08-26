import torch
from src.models.loader import load_blip_rsicd

def verify():
    print("Verifying model loader and adapter integrity...")
    processor, model, device = load_blip_rsicd()
    
    # Verify LoRA modules
    lora_modules = 0
    lora_tensors = 0
    mismatch = False
    
    for name, param in model.named_parameters():
        if param.device.type != device:
            mismatch = True
            print(f"DEVICE MISMATCH: {name} on {param.device}, expected {device}")
        
        if "lora_A" in name or "lora_B" in name:
            lora_tensors += 1
            if "lora_A" in name:
                lora_modules += 1
                
    print(f"LoRA Modules Injected: {lora_modules}")
    print(f"LoRA Tensors Loaded: {lora_tensors}")
    print(f"Device: {device}")
    
    assert not mismatch, "Device mismatch found!"
    assert lora_modules == 48, f"Expected 48 modules, got {lora_modules}"
    assert lora_tensors == 96, f"Expected 96 tensors, got {lora_tensors}"
    
    print("Verification PASS")

if __name__ == "__main__":
    verify()
