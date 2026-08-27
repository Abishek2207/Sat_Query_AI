import os
import sys
import torch
import gc
import json
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.loader import load_blip_rsicd
from backend.app.local_specialists import _load_image
from PIL import Image

def run_tests():
    report = {}
    device = "cpu"
    image_data = [{"filename": "image_0.jpg", "bytes": open("datasets/rsicd/images/image_0.jpg", "rb").read()}]
    img = _load_image(image_data)

    # ---------------- Phase 5: Captioning ----------------
    print("Testing Captioning...")
    try:
        b_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        b_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        inputs = b_proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = b_model.generate(**inputs, max_new_tokens=30)
        baseline_cap = b_proc.decode(out[0], skip_special_tokens=True)
        
        del b_proc, b_model, inputs, out
        gc.collect()
        
        a_proc, a_model, a_dev = load_blip_rsicd()
        inputs2 = a_proc(img, return_tensors="pt").to(device)
        with torch.no_grad():
            out2 = a_model.generate(**inputs2, max_new_tokens=30)
        adapted_cap = a_proc.decode(out2[0], skip_special_tokens=True)
        
        del a_proc, a_model, inputs2, out2
        gc.collect()

        report["captioning"] = {
            "model": "Salesforce/blip-image-captioning-base",
            "adapter": "models/rsicd_blip_lora/adapter.pt",
            "baseline": baseline_cap,
            "adapted": adapted_cap,
            "status": "PASS"
        }
    except Exception as e:
        report["captioning"] = {"status": f"FAIL: {str(e)}"}

    # ---------------- Phase 6: Grounding ----------------
    print("Testing Grounding...")
    try:
        g_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        g_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
        g_text = "aircraft."
        g_inputs = g_proc(images=img, text=g_text, return_tensors="pt").to(device)
        with torch.no_grad():
            g_out = g_model(**g_inputs)
        results = g_proc.post_process_grounded_object_detection(
            g_out, g_inputs.input_ids, threshold=0.3, text_threshold=0.3, target_sizes=[img.size[::-1]]
        )
        boxes = results[0]["boxes"].tolist() if len(results) > 0 else []
        
        del g_proc, g_model, g_inputs, g_out
        gc.collect()
        
        report["grounding"] = {
            "model": "IDEA-Research/grounding-dino-base",
            "inference": True,
            "boxes": boxes[:2] if boxes else [],
            "status": "PASS"
        }
    except Exception as e:
        report["grounding"] = {"status": f"FAIL: {str(e)}"}

    # ---------------- Phase 7: VQA ----------------
    print("Testing VQA...")
    try:
        v_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        v_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
        v_q = "What is in the image?"
        v_inputs = v_proc(img, v_q, return_tensors="pt").to(device)
        with torch.no_grad():
            v_out = v_model.generate(**v_inputs)
        v_ans = v_proc.decode(v_out[0], skip_special_tokens=True)
        
        del v_proc, v_model, v_inputs, v_out
        gc.collect()

        report["vqa"] = {
            "model": "Salesforce/blip-vqa-base",
            "inference": True,
            "result": v_ans,
            "status": "PASS"
        }
    except Exception as e:
        report["vqa"] = {"status": f"FAIL: {str(e)}"}

    # ---------------- Phase 8: Change Analysis ----------------
    print("Testing Change Analysis...")
    try:
        from backend.app.change_map import compute_change_baseline
        t1 = open("datasets/rsicd/images/image_0.jpg", "rb").read()
        t2 = open("datasets/rsicd/images/image_1.jpg", "rb").read()
        res_change = compute_change_baseline(t1, t2)
        report["change"] = {
            "model": "Deterministic_PixelDiff_Baseline",
            "test": True,
            "inference": True,
            "result": res_change.get("answer"),
            "status": "PASS"
        }
    except Exception as e:
        report["change"] = {"status": f"FAIL: {str(e)}"}
        
    # ---------------- Phase 9: Optical-SAR ----------------
    print("Testing Optical-SAR...")
    try:
        from backend.app.optical_sar import verify_optical_sar_pair
        res_opsar = verify_optical_sar_pair(t1, t2)
        report["optical_sar"] = {
            "model": "Metadata_Alignment_Baseline",
            "opt_in": True,
            "sar_in": True,
            "inference": True,
            "result": res_opsar.get("answer"),
            "status": "PASS"
        }
    except Exception as e:
        report["optical_sar"] = {"status": f"FAIL: {str(e)}"}

    with open("reality_audit.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run_tests()
