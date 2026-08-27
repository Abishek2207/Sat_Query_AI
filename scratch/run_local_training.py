import os
import time
import json
import torch
import math
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.nn as nn

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

def get_trainable_parameters(model):
    trainable = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable += num_params
    return trainable, all_param

def run():
    print("Starting Manual PyTorch LoRA Training...")
    base_dir = Path("datasets/rsicd")
    models_dir = Path("models/rsicd_blip_lora_local")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Datasets
    with open(base_dir / "train_manifest.jsonl") as f:
        train_samples = [json.loads(line) for line in f]
    with open(base_dir / "val_manifest.jsonl") as f:
        val_samples = [json.loads(line) for line in f]
        
    print(f"Loaded {len(train_samples)} training and {len(val_samples)} validation samples.")
    
    # 2. Load Model
    print("Loading Base Model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Inject Native LoRA
    print("Injecting Custom LoRA...")
    replaced = inject_lora(model)
    model.to(device)
    print(f"Injected {replaced} LoRA modules.")
    
    trainable, total = get_trainable_parameters(model)
    print(f"Trainable parameters: {trainable} || Total parameters: {total}")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # 4. ONE BATCH TEST
    print("Running ONE BATCH test...")
    model.train()
    s = train_samples[0]
    img = Image.open(s["filename"]).convert("RGB")
    inputs = processor(images=img, text=s["caption"], return_tensors="pt").to(device)
    inputs["labels"] = inputs["input_ids"].clone()
    
    loss = model(**inputs).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"ONE BATCH TEST SUCCESS! Loss: {loss.item():.4f}")
    
    # 5. FULL TRAINING LOOP (1 epoch)
    print("Starting 1 epoch training loop...")
    start_time = time.time()
    
    total_loss = 0
    steps = 0
    
    for i, s in enumerate(train_samples):
        img = Image.open(s["filename"]).convert("RGB")
        inputs = processor(images=img, text=s["caption"], return_tensors="pt", padding="max_length", max_length=30, truncation=True).to(device)
        inputs["labels"] = inputs["input_ids"].clone()
        
        loss = model(**inputs).loss
        loss.backward()
        
        if (i + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        steps += 1
        
        if steps % 100 == 0:
            print(f"Step {steps}/{len(train_samples)} - Loss: {total_loss/steps:.4f}")
            
    if (i + 1) % 4 != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    avg_loss = total_loss / steps
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds. Final avg loss: {avg_loss:.4f}")
    
    # 6. SAVE ADAPTER
    state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(state_dict, models_dir / "adapter.pt")
    print(f"Saved {len(state_dict)} LoRA tensors to {models_dir / 'adapter.pt'}")
    
    # 7. EVALUATION
    model.eval()
    val_subset = val_samples[:20]
    adapted_predictions = []
    
    for s in val_subset:
        img = Image.open(s["filename"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        pred = processor.decode(out[0], skip_special_tokens=True)
        adapted_predictions.append({
            "filename": s["filename"],
            "reference_caption": s["caption"],
            "adapted_prediction": pred
        })
        
    with open(base_dir / "adapted_predictions_local.jsonl", "w") as f:
        for p in adapted_predictions: f.write(json.dumps(p) + "\n")
        
    results = {
        "dataset": "arampacha/rsicd",
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "model": "Salesforce/blip-image-captioning-base",
        "training_status": "COMPLETED",
        "training_parameters": {
            "r": 8,
            "alpha": 16,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum": 4,
            "learning_rate": 1e-4
        },
        "trainable_parameters": trainable,
        "checkpoint": str(models_dir),
        "final_train_loss": avg_loss,
        "duration_seconds": duration
    }
    
    with open("adaptation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    report = f"""# RSICD Adaptation Report (Local CPU Test)

## Experiment
- **Objective**: Fix PEFT inputs_embeds conflict via Native PyTorch LoRA
- **Dataset**: arampacha/rsicd (400 Train, 100 Val)
- **Base model**: Salesforce/blip-image-captioning-base
- **Adaptation method**: Native Custom LoRA injected into query/value
- **LoRA configuration**: r=8, alpha=16
- **Trainable parameters**: {trainable}
- **Environment**: {device}

## Training
- **Epochs**: 1
- **Loss**: {avg_loss:.4f}
- **Duration**: {duration:.2f}s

## Results
- **Adapter**: `models/rsicd_blip_lora_local/adapter.pt`
- Successfully loaded and generated captions on {len(val_subset)} validation samples.

## Diagnostic
The previous `get_peft_model` implementation forwarded duplicate args during BLIP's multimodal pass. Writing a custom PyTorch LoRA completely avoids PEFT's generic forward-hooks and natively modifies the linear weights, solving the crash.
"""
    with open("ADAPTATION_REPORT.md", "w") as f:
        f.write(report)
        
    print("ALL DONE!")

if __name__ == "__main__":
    run()
