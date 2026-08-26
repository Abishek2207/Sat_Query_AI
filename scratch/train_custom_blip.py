import os
import json
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

def run():
    print("--- custom PyTorch BLIP training loop (No DataLoader) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    target_modules = ["query", "value"]
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )
    
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    peft_model.to(device)
    
    # Load manifest manually
    train_samples = []
    with open("datasets/rsicd/train_manifest.jsonl", "r") as f:
        for line in f:
            train_samples.append(json.loads(line))
            
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-5)
    
    print("\n--- Testing Single Batch ---")
    peft_model.train()
    
    s = train_samples[0]
    img = Image.open(s["filename"]).convert("RGB")
    encoding = processor(images=img, text=s["caption"], return_tensors="pt", padding="max_length", truncation=True, max_length=30)
    encoding["labels"] = encoding["input_ids"].clone()
    encoding["labels"][encoding["labels"] == processor.tokenizer.pad_token_id] = -100
    batch = {k: v.to(device) for k, v in encoding.items()}
    
    outputs = peft_model(**batch)
    loss = outputs.loss
    print(f"First batch loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print("SUCCESS: Single batch forward/backward pass completed.")
    
    print("\n--- Starting Full Epoch ---")
    gradient_accumulation_steps = 4
    total_loss = 0
    start_time = time.time()
    
    for step, s in enumerate(train_samples):
        img = Image.open(s["filename"]).convert("RGB")
        encoding = processor(images=img, text=s["caption"], return_tensors="pt", padding="max_length", truncation=True, max_length=30)
        encoding["labels"] = encoding["input_ids"].clone()
        encoding["labels"][encoding["labels"] == processor.tokenizer.pad_token_id] = -100
        batch = {k: v.to(device) for k, v in encoding.items()}
        
        outputs = peft_model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        
        total_loss += outputs.loss.item()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/400 - Loss: {outputs.loss.item():.4f}")
            
    final_loss = total_loss / len(train_samples)
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f}s. Avg Loss: {final_loss:.4f}")
    
    out_dir = Path("models/rsicd_blip_lora")
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    
    with open(out_dir / "training_config.json", "w") as f:
        json.dump({"lr": 1e-5, "epochs": 1, "batch_size": 1, "grad_accum": 4}, f)
    with open(out_dir / "training_log.json", "w") as f:
        json.dump([{"avg_loss": final_loss, "duration": duration}], f)
        
    print(f"Adapter saved to {out_dir}")
    
    print("\n--- Inference on 20 validation samples ---")
    val_samples = []
    with open("datasets/rsicd/val_manifest.jsonl", "r") as f:
        for line in f:
            val_samples.append(json.loads(line))
            
    baseline_predictions = []
    adapted_predictions = []
    
    raw_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    raw_model.eval()
    peft_model.eval()
    
    for i in range(20):
        s = val_samples[i]
        img = Image.open(s["filename"]).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            base_out = raw_model.generate(**inputs, max_new_tokens=30)
            adapt_out = peft_model.generate(**inputs, max_new_tokens=30)
            
        base_pred = processor.decode(base_out[0], skip_special_tokens=True)
        adapt_pred = processor.decode(adapt_out[0], skip_special_tokens=True)
        
        baseline_predictions.append({
            "filename": s["filename"],
            "reference_caption": s["caption"],
            "baseline_prediction": base_pred
        })
        
        adapted_predictions.append({
            "filename": s["filename"],
            "reference_caption": s["caption"],
            "adapted_prediction": adapt_pred
        })
        
    with open("datasets/rsicd/baseline_predictions.jsonl", "w") as f:
        for p in baseline_predictions: f.write(json.dumps(p) + "\n")
    with open("datasets/rsicd/adapted_predictions.jsonl", "w") as f:
        for p in adapted_predictions: f.write(json.dumps(p) + "\n")
        
    print("Inference completed. Results saved.")
    
if __name__ == "__main__":
    run()
