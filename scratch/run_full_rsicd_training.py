import json
import torch
import torch.nn as nn
import math
import time
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import numpy as np

# --- 1. Custom LoRA Implementation ---
class LoraLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
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
        
    replaced_count = 0
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

def extract_lora_state_dict(model):
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            state_dict[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state_dict[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state_dict

def load_lora_state_dict(model, state_dict):
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            module.lora_A.data.copy_(state_dict[f"{name}.lora_A"])
            module.lora_B.data.copy_(state_dict[f"{name}.lora_B"])

# --- 2. Dataset Preparation ---
class RSICDDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        caption = item["captions"][0] if isinstance(item["captions"], list) else item["captions"]
        
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "pixel_values": inputs.pixel_values.squeeze(),
            "labels": inputs.input_ids.squeeze().clone()
        }

def run_full_training():
    print("--- SYSTEM DIAGNOSTICS ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device set to: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    print("\n--- LOADING DATASET ---")
    ds = load_dataset("arampacha/rsicd", split="train")
    
    # Exact specification: 2000 train, 500 val
    train_data = ds.select(range(2000))
    val_data = ds.select(range(2000, 2500))
    
    print(f"Dataset Name: arampacha/rsicd")
    print(f"Train Samples Count: {len(train_data)}")
    print(f"Validation Samples Count: {len(val_data)}")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    train_dataset = RSICDDataset(train_data, processor)
    val_dataset_full = RSICDDataset(val_data, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset_full, batch_size=4, shuffle=False)
    
    print("\n--- MODEL INITIALIZATION ---")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    replaced = inject_lora(model)
    print(f"Injected Custom LoRA into {replaced} modules.")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params} ({(trainable_params/total_params)*100:.4f}%)")
    
    if trainable_params != 589824:
        raise ValueError(f"CRITICAL ERROR: Expected 589,824 trainable parameters, got {trainable_params}.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 3
    grad_accum = 4
    history = []
    
    print("\n--- STARTING TRAINING LOOP ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        print(f"\n[Epoch {epoch+1}/{epochs}] Started.")
        
        for idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            
            if not math.isfinite(loss.item()):
                raise ValueError("CRITICAL ERROR: Training loss is not finite (NaN/Inf). Diverged.")
                
            epoch_loss += loss.item() * grad_accum
            
            if (idx + 1) % grad_accum == 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
                
                if steps % 100 == 0:
                    print(f"  Step {steps} | Train Loss: {loss.item() * grad_accum:.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation Pass
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Calculating Validation Loss"):
                outputs = model(
                    input_ids=batch["input_ids"].to(device), 
                    pixel_values=batch["pixel_values"].to(device), 
                    attention_mask=batch["attention_mask"].to(device), 
                    labels=batch["labels"].to(device)
                )
                val_loss_sum += outputs.loss.item()
                
        avg_val_loss = val_loss_sum / len(val_loader)
        
        print(f"--> [Epoch {epoch+1} Completed] Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        history.append({
            "epoch": epoch+1, 
            "avg_train_loss": avg_train_loss, 
            "avg_val_loss": avg_val_loss
        })
        
    elapsed = time.time() - start_time
    print(f"\n--- TRAINING FINISHED in {elapsed:.2f} seconds ---")
    
    # --- 4. Save Adapter & History ---
    print("\nSaving Checkpoints to /kaggle/working/...")
    adapter_dir = "/kaggle/working/rsicd_blip_lora"
    os.makedirs(adapter_dir, exist_ok=True)
    
    lora_state = extract_lora_state_dict(model)
    torch.save(lora_state, f"{adapter_dir}/adapter.pt")
    
    with open("/kaggle/working/rsicd_training_history.json", "w") as f:
        json.dump({
            "history": history, 
            "elapsed_seconds": elapsed, 
            "train_samples": 2000, 
            "val_samples": 500,
            "gpu_used": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        }, f, indent=2)
        
    # Free memory before evaluation
    del model
    del optimizer
    torch.cuda.empty_cache()
    
    # --- 5. Reload & Evaluate ---
    print("\n--- EVALUATION: BASELINE VS ADAPTED (100 REAL RSICD VALIDATION IMAGES) ---")
    # Requires: pip install evaluate nltk rouge_score
    try:
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
    except Exception as e:
        print("WARNING: evaluation metrics not found. Please pip install evaluate nltk rouge_score.")
        raise e
    
    eval_subset = val_data.select(range(100))
    
    print("Loading Fresh Baseline Model...")
    baseline_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    baseline_model.eval()
    
    baseline_preds, baseline_logits_sample, refs = [], [], []
    
    print("Generating Baseline Predictions...")
    with torch.no_grad():
        for item in tqdm(eval_subset):
            img = item["image"].convert("RGB")
            caption = item["captions"][0] if isinstance(item["captions"], list) else item["captions"]
            refs.append([caption])
            
            inputs = processor(images=img, return_tensors="pt").to(device)
            out = baseline_model.generate(**inputs, max_new_tokens=20)
            baseline_preds.append(processor.decode(out[0], skip_special_tokens=True))
            
            # Extract logits for physical difference verification
            inputs_with_text = processor(images=img, text=caption, return_tensors="pt", padding="max_length", max_length=32).to(device)
            inputs_with_text["labels"] = inputs_with_text["input_ids"].clone()
            baseline_logits_sample.append(baseline_model(**inputs_with_text).logits.cpu())
            
    print("Reloading Custom LoRA Adapter onto Baseline...")
    inject_lora(baseline_model)
    load_lora_state_dict(baseline_model, torch.load(f"{adapter_dir}/adapter.pt", weights_only=True))
    baseline_model.eval()
    
    adapted_preds, adapted_logits_sample = [], []
    changed_preds = 0
    
    print("Generating Adapted Predictions...")
    with torch.no_grad():
        for i, item in enumerate(tqdm(eval_subset)):
            img = item["image"].convert("RGB")
            caption = item["captions"][0] if isinstance(item["captions"], list) else item["captions"]
            
            inputs = processor(images=img, return_tensors="pt").to(device)
            out = baseline_model.generate(**inputs, max_new_tokens=20)
            pred = processor.decode(out[0], skip_special_tokens=True)
            adapted_preds.append(pred)
            
            if pred != baseline_preds[i]:
                changed_preds += 1
                
            inputs_with_text = processor(images=img, text=caption, return_tensors="pt", padding="max_length", max_length=32).to(device)
            inputs_with_text["labels"] = inputs_with_text["input_ids"].clone()
            adapted_logits_sample.append(baseline_model(**inputs_with_text).logits.cpu())
            
    # Calculate Differences
    diffs = [torch.abs(b - a).mean().item() for b, a in zip(baseline_logits_sample, adapted_logits_sample)]
    max_diffs = [torch.abs(b - a).max().item() for b, a in zip(baseline_logits_sample, adapted_logits_sample)]
    
    mean_abs_diff = np.mean(diffs)
    max_logit_diff = np.max(max_diffs)
    
    print("\nComputing BLEU & ROUGE-L Metrics...")
    base_bleu = bleu_metric.compute(predictions=baseline_preds, references=refs)
    adapt_bleu = bleu_metric.compute(predictions=adapted_preds, references=refs)
    base_rouge = rouge_metric.compute(predictions=baseline_preds, references=refs)
    adapt_rouge = rouge_metric.compute(predictions=adapted_preds, references=refs)
    
    report = {
        "dataset": "arampacha/rsicd",
        "train_samples": 2000,
        "val_samples_for_metrics": 100,
        "model": "Salesforce/blip-image-captioning-base",
        "lora_parameters": 589824,
        "training_time_seconds": elapsed,
        "epochs": epochs,
        "history": history,
        "changed_predictions_count": changed_preds,
        "mean_absolute_logit_difference": mean_abs_diff,
        "max_logit_difference": max_logit_diff,
        "metrics": {
            "baseline": {
                "bleu": base_bleu["bleu"],
                "rougeL": base_rouge["rougeL"]
            },
            "adapted": {
                "bleu": adapt_bleu["bleu"],
                "rougeL": adapt_rouge["rougeL"]
            }
        }
    }
    
    report_path = "/kaggle/working/FINAL_REAL_TRAINING_EVIDENCE.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n--- FINAL EVIDENCE REPORT ---")
    print(json.dumps(report, indent=2))
    print(f"SUCCESS: Evidence saved to {report_path}")

if __name__ == "__main__":
    run_full_training()
