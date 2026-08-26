import os
import torch
import torch.nn as nn
import math
from transformers import BlipProcessor, BlipForConditionalGeneration

class LoraLinear(nn.Module):
    def __init__(self, original_linear, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.lora_A = nn.Parameter(torch.zeros(original_linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, original_linear.out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        orig_out = self.original_linear(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return orig_out + lora_out * self.scaling

def inject_lora(model, r=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False
    
    replaced = 0
    for name, module in dict(model.named_modules()).items():
        if name.endswith("query") or name.endswith("value"):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            child_name = name.rsplit(".", 1)[1]
            orig_linear = getattr(parent, child_name)
            if isinstance(orig_linear, nn.Linear):
                setattr(parent, child_name, LoraLinear(orig_linear, r, alpha))
                replaced += 1
    return replaced

def load_blip_rsicd(base_name="Salesforce/blip-image-captioning-base", adapter_path="models/rsicd_blip_lora/adapter.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Loader] Initializing base model {base_name} on {device}...")
    
    processor = BlipProcessor.from_pretrained(base_name)
    model = BlipForConditionalGeneration.from_pretrained(base_name)
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"[Loader] Adapter tensor file missing at {adapter_path}")
        
    print(f"[Loader] Injecting native PyTorch LoRA...")
    replaced = inject_lora(model, r=8, alpha=16)
    print(f"[Loader] Injected {replaced} LoRA modules.")
    
    if replaced != 48:
        raise ValueError(f"[Loader] Expected 48 modules to be injected, found {replaced}.")
        
    print(f"[Loader] Loading trained tensors from {adapter_path}...")
    state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
    
    if len(state_dict) != 96:
        raise ValueError(f"[Loader] Expected 96 LoRA tensors in adapter, found {len(state_dict)}.")
        
    missing_keys = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in state_dict and key_b in state_dict:
                module.lora_A.data.copy_(state_dict[key_a])
                module.lora_B.data.copy_(state_dict[key_b])
            else:
                missing_keys.append(name)
                
    if missing_keys:
        raise RuntimeError(f"[Loader] Missing adapter tensors for modules: {missing_keys}")
        
    model.to(device)
    model.eval()
    
    # Strictly ensure everything is on the exact same device
    for name, param in model.named_parameters():
        if param.device.type != device:
            param.data = param.data.to(device)
            
    print(f"[Loader] Success! All tensors placed on {device}.")
    return processor, model, device
