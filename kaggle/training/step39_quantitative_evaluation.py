import os
import sys
import json
import torch
from PIL import Image
from collections import Counter
import evaluate

# Add backend to path for loader
sys.path.append("/kaggle/working/Sat_Query_AI")
try:
    from src.models.loader import inject_lora
except ImportError:
    pass

from transformers import BlipProcessor, BlipForConditionalGeneration

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_quality_metrics(captions):
    lengths, unique_ratios, repeated_bigrams, repeated_trigrams = [], [], [], []
    for cap in captions:
        tokens = cap.lower().split()
        if not tokens: continue
        lengths.append(len(tokens))
        unique_ratios.append(len(set(tokens)) / len(tokens))
        bigrams = Counter(get_ngrams(tokens, 2))
        trigrams = Counter(get_ngrams(tokens, 3))
        repeated_bigrams.append(sum(c - 1 for c in bigrams.values() if c > 1))
        repeated_trigrams.append(sum(c - 1 for c in trigrams.values() if c > 1))

    return {
        "average_length": sum(lengths) / len(lengths) if lengths else 0,
        "average_unique_ratio": sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0,
        "average_repeated_bigram": sum(repeated_bigrams) / len(repeated_bigrams) if repeated_bigrams else 0,
        "average_repeated_trigram": sum(repeated_trigrams) / len(repeated_trigrams) if repeated_trigrams else 0,
        "exact_duplicate_captions": len(captions) - len(set(captions))
    }

def main():
    print("============================================================")
    print("STEP 39: QUANTITATIVE RSICD EVALUATION (STRICT NO-MOCK)")
    print("============================================================")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Kaggle specific dataset paths
    img_dir = "/kaggle/input/rsicd/images"
    ground_truth_path = "/kaggle/input/rsicd/dataset_rsicd.json"
    ckpt_path = "/kaggle/working/rsicd_lora_checkpoints/best/lora_weights.pt"
    
    # 1. Dataset Loading - STRICT FILTERING for exactly 1093 test images
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Strict Audit: Real ground truth file missing at {ground_truth_path}. Halting to prevent hallucinated data.")
        
    with open(ground_truth_path, 'r') as f:
        gt_dataset = json.load(f)
        
    test_images = []
    references = []
    
    for img_info in gt_dataset.get('images', []):
        if img_info.get('split') == 'test':
            filename = img_info.get('filename')
            filepath = os.path.join(img_dir, filename)
            sentences = [s['raw'] for s in img_info.get('sentences', [])]
            
            if os.path.exists(filepath) and len(sentences) > 0:
                test_images.append(filepath)
                # Cap or pad to exactly 5 references to maintain strict 5465 alignment
                refs = sentences[:5]
                while len(refs) < 5: refs.append(refs[0])
                references.append(refs)

    if len(test_images) != 1093:
        raise ValueError(f"Strict Audit: Expected exactly 1093 test images, found {len(test_images)}. Dataset contaminated.")
    
    total_refs = sum(len(r) for r in references)
    if total_refs != 5465:
        raise ValueError(f"Strict Audit: Expected exactly 5465 references, found {total_refs}.")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Validated Step 36 Generation config
    gen_kwargs = {
        "max_new_tokens": 50,
        "num_beams": 3,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2
    }
    
    # Load evaluation metrics directly via HF Evaluate (Compute at runtime)
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    
    # Optional CIDEr/SPICE using pycocoevalcap (Requires Java + coco-caption repo)
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
        cider_scorer = Cider()
        spice_scorer = Spice()
        cider_available = True
    except ImportError:
        cider_available = False
        print("WARNING: pycocoevalcap missing. CIDEr and SPICE cannot be computed.")
    
    # ==========================
    # Evaluate BASE model
    # ==========================
    print("\n--- Evaluating BASE BLIP ---")
    base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    base_model.eval()
    
    base_preds = []
    with torch.no_grad():
        for p in test_images:
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            out = base_model.generate(**inputs, **gen_kwargs)
            base_preds.append(processor.decode(out[0], skip_special_tokens=True))
            
    base_quality = compute_quality_metrics(base_preds)
    
    base_bleu1 = bleu.compute(predictions=base_preds, references=references, max_order=1)["bleu"]
    base_bleu2 = bleu.compute(predictions=base_preds, references=references, max_order=2)["bleu"]
    base_bleu3 = bleu.compute(predictions=base_preds, references=references, max_order=3)["bleu"]
    base_bleu4 = bleu.compute(predictions=base_preds, references=references, max_order=4)["bleu"]
    base_rouge = rouge.compute(predictions=base_preds, references=references)["rougeL"]
    base_met = meteor.compute(predictions=base_preds, references=references)["meteor"]
    
    base_cider, base_spice = 0.0, 0.0
    if cider_available:
        # COCO expects dict formats: {img_id: [captions]}
        gts_dict = {i: refs for i, refs in enumerate(references)}
        res_dict = {i: [pred] for i, pred in enumerate(base_preds)}
        base_cider, _ = cider_scorer.compute_score(gts_dict, res_dict)
        base_spice, _ = spice_scorer.compute_score(gts_dict, res_dict)
    
    del base_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # ==========================
    # Evaluate LoRA
    # ==========================
    print("\n--- Evaluating RSICD LoRA ---")
    lora_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    replaced = inject_lora(lora_model, r=8, alpha=16)
    assert replaced == 48, f"Strict Audit: Expected 48 modules, got {replaced}"
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Strict Audit: Checkpoint {ckpt_path} missing. Halting.")
        
    state_dict = torch.load(ckpt_path, map_location="cpu")
    assert len(state_dict) == 96, f"Strict Audit: Expected 96 tensors, got {len(state_dict)}"
    
    for name, module in lora_model.named_modules():
        if hasattr(module, 'lora_A'):
            ka, kb = f"{name}.lora_A", f"{name}.lora_B"
            a, b = state_dict[ka], state_dict[kb]
            if a.shape != module.lora_A.shape: a = a.t()
            if b.shape != module.lora_B.shape: b = b.t()
            module.lora_A.data.copy_(a)
            module.lora_B.data.copy_(b)
            
    lora_model.to(device)
    lora_model.eval()
        
    lora_preds = []
    with torch.no_grad():
        for p in test_images:
            img = Image.open(p).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            out = lora_model.generate(**inputs, **gen_kwargs)
            lora_preds.append(processor.decode(out[0], skip_special_tokens=True))
            
    lora_quality = compute_quality_metrics(lora_preds)
    
    lora_bleu1 = bleu.compute(predictions=lora_preds, references=references, max_order=1)["bleu"]
    lora_bleu2 = bleu.compute(predictions=lora_preds, references=references, max_order=2)["bleu"]
    lora_bleu3 = bleu.compute(predictions=lora_preds, references=references, max_order=3)["bleu"]
    lora_bleu4 = bleu.compute(predictions=lora_preds, references=references, max_order=4)["bleu"]
    lora_rouge = rouge.compute(predictions=lora_preds, references=references)["rougeL"]
    lora_met = meteor.compute(predictions=lora_preds, references=references)["meteor"]
    
    lora_cider, lora_spice = 0.0, 0.0
    if cider_available:
        gts_dict = {i: refs for i, refs in enumerate(references)}
        res_dict = {i: [pred] for i, pred in enumerate(lora_preds)}
        lora_cider, _ = cider_scorer.compute_score(gts_dict, res_dict)
        lora_spice, _ = spice_scorer.compute_score(gts_dict, res_dict)
    
    print("\nBASE BLIP")
    print(f"BLEU-1: {base_bleu1}")
    print(f"BLEU-2: {base_bleu2}")
    print(f"BLEU-3: {base_bleu3}")
    print(f"BLEU-4: {base_bleu4}")
    print(f"ROUGE-L: {base_rouge}")
    print(f"METEOR: {base_met}")
    print(f"CIDEr: {base_cider}")
    print(f"SPICE: {base_spice}")

    print("\nRSICD LoRA")
    print(f"BLEU-1: {lora_bleu1}")
    print(f"BLEU-2: {lora_bleu2}")
    print(f"BLEU-3: {lora_bleu3}")
    print(f"BLEU-4: {lora_bleu4}")
    print(f"ROUGE-L: {lora_rouge}")
    print(f"METEOR: {lora_met}")
    print(f"CIDEr: {lora_cider}")
    print(f"SPICE: {lora_spice}")
    
    report_path = "/kaggle/working/rsicd_step39_quantitative_evaluation.json"
    report_data = {
        "dataset": "RSICD",
        "test_images": len(test_images),
        "references": total_refs,
        "base_metrics": {
            "BLEU-1": base_bleu1, "BLEU-2": base_bleu2, "BLEU-3": base_bleu3, "BLEU-4": base_bleu4,
            "ROUGE-L": base_rouge, "METEOR": base_met, "CIDEr": base_cider, "SPICE": base_spice
        },
        "lora_metrics": {
            "BLEU-1": lora_bleu1, "BLEU-2": lora_bleu2, "BLEU-3": lora_bleu3, "BLEU-4": lora_bleu4,
            "ROUGE-L": lora_rouge, "METEOR": lora_met, "CIDEr": lora_cider, "SPICE": lora_spice
        },
        "base_quality": base_quality,
        "lora_quality": lora_quality
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

if __name__ == "__main__":
    main()
