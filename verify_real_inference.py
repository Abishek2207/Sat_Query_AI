import time
import json
import psutil
import torch
import warnings
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def run_real_inference():
    with open("datasets/sentinel2/optical_before.tif", "rb") as f:
        opt_b = f.read()
    with open("datasets/sentinel2/optical_after.tif", "rb") as f:
        opt_a = f.read()
    with open("datasets/sentinel1/sar_before.tif", "rb") as f:
        sar_b = f.read()

    results = []

    print("==================================================")
    print("STARTING REAL INFERENCE VERIFICATION")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("==================================================")

    # 1. VQA
    print("\n--- A. VQA ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "How many buildings are visible?"}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "VQA", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 2. RSICD CAPTIONING
    print("\n--- B. RSICD CAPTIONING ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "Describe the scene"}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "Captioning", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 3. GROUNDING
    print("\n--- C. GROUNDING ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "Highlight the buildings"}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    print(f"Evidence Regions: {[e.get('region') for e in res.get('evidence', [])]}")
    results.append({"test": "Grounding", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 4. LAND COVER
    print("\n--- D. LAND COVER ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "What is the dominant land cover?"}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "Land Cover", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 5. TEMPORAL CHANGE
    print("\n--- E. TEMPORAL CHANGE ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "What changed between these two images?"}, files=[
        ("files", ("opt_b.tif", opt_b, "image/tiff")),
        ("files", ("opt_a.tif", opt_a, "image/tiff"))
    ]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "Temporal Change", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 6. OPTICAL + SAR
    print("\n--- F. OPTICAL + SAR ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "Analyze this area using optical and SAR together."}, files=[
        ("files", ("opt_b.tif", opt_b, "image/tiff")),
        ("files", ("sar_b.tif", sar_b, "image/tiff"))
    ]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "Optical+SAR", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 7. MULTI-TOOL
    print("\n--- G. MULTI-TOOL ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "Describe the image and highlight the buildings."}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    print(f"Selected Tools: {res.get('selected_tools')}")
    results.append({"test": "Multi-Tool", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace"), "selected_tools": res.get("selected_tools")})

    # 8. HALLUCINATION TRAP
    print("\n--- H. HALLUCINATION TRAP ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "Is there a hospital in this image?"}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "Hallucination Trap", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 9. MISSING TEMPORAL INPUT
    print("\n--- I. MISSING TEMPORAL INPUT ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "What changed?"}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "Missing Temporal", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    # 10. MISSING SAR INPUT
    print("\n--- J. MISSING SAR INPUT ---")
    start = time.time()
    res = client.post("/analyze", data={"query": "Analyze SAR"}, files=[("files", ("opt_b.tif", opt_b, "image/tiff"))]).json()
    duration = time.time() - start
    print(f"Task: {res.get('task')}, Status: {res.get('status')}")
    print(f"Answer: {res.get('answer')}")
    results.append({"test": "Missing SAR", "status": res.get("status"), "task": res.get("task"), "answer": res.get("answer"), "duration": duration, "trace": res.get("execution_trace")})

    with open("real_inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nMEMORY USAGE:")
    process = psutil.Process()
    print(f"{process.memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    run_real_inference()
