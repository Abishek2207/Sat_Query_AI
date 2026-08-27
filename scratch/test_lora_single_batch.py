import torch
import torch.nn as nn
import math
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import copy

class LoraLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # LoRA weights: in_features x r, and r x out_features
        self.lora_A = nn.Parameter(torch.zeros(original_linear.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, original_linear.out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        orig_out = self.original_linear(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return orig_out + lora_out * self.scaling


def inject_lora(model, r=8, alpha=16):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    replaced_count = 0
    # Traverse and replace targets
    for name, module in dict(model.named_modules()).items():
        if name.endswith("query") or name.endswith("value"):
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
            parent = model.get_submodule(parent_name)
            original_linear = getattr(parent, child_name)
            
            if isinstance(original_linear, nn.Linear):
                lora_layer = LoraLinear(original_linear, r, alpha)
                setattr(parent, child_name, lora_layer)
                replaced_count += 1
                
    return replaced_count

def run_single_batch_test():
    print("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    print("Injecting Custom LoRA...")
    replaced = inject_lora(model)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modules Replaced: {replaced}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Total Parameters: {total_params}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    if trainable_params != 589824:
        print(f"WARNING: Trainable parameters count is {trainable_params} instead of expected 589824.")

    # Get a real RSICD image
    print("\nPreparing a single batch of REAL RSICD data...")
    try:
        img = Image.open("datasets/rsicd/images/image_1.jpg").convert("RGB")
        text = "an aerial view of a green forest"
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
        
    inputs = processor(images=img, text=text, return_tensors="pt").to(device)
    inputs["labels"] = inputs["input_ids"].clone()
    
    # Clone weights for before/after comparison
    print("\nCaching initial LoRA weights...")
    initial_A = None
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            initial_A = module.lora_A.detach().clone()
            break
            
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\nRunning Forward Pass (Before Update)...")
    outputs_before = model(**inputs)
    loss = outputs_before.loss
    print(f"Initial Loss (Finite Check): {loss.item()}")
    
    print("\nRunning Backward Pass...")
    loss.backward()
    
    # Check gradients
    grad_norm = 0.0
    grad_found = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_found = True
            grad_norm += p.grad.norm().item()
            
    print(f"Total Gradient Norm on LoRA parameters: {grad_norm:.6f}")
    if not grad_found or grad_norm == 0.0:
        print("ERROR: No gradients found on LoRA parameters!")
    else:
        print("SUCCESS: Non-zero gradients computed.")
        
    print("\nRunning Optimizer Step...")
    optimizer.step()
    
    # Check weight changes
    weight_changed = False
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            if not torch.equal(module.lora_A, initial_A):
                weight_changed = True
            break
            
    if weight_changed:
        print("SUCCESS: LoRA weights were physically updated.")
    else:
        print("ERROR: LoRA weights did not change after optimizer.step().")
        
    print("\nRunning Forward Pass (After Update)...")
    model.eval()
    with torch.no_grad():
        outputs_after = model(**inputs)
        loss_after = outputs_after.loss
        
    print(f"Loss After Update: {loss_after.item()}")
    if outputs_before.logits.abs().sum().item() != outputs_after.logits.abs().sum().item():
        print("SUCCESS: Logits changed after update.")
    else:
        print("ERROR: Logits did not change.")

if __name__ == "__main__":
    run_single_batch_test()
