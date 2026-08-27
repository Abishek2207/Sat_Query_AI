import sys
import os
import json
import base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app

def run_e2e_tests():
    client = TestClient(app)
    
    with open("datasets/rsicd/images/image_0.jpg", "rb") as f1:
        img0_bytes = f1.read()
    with open("datasets/rsicd/images/image_1.jpg", "rb") as f2:
        img1_bytes = f2.read()

    # 1. Grounding (should return EvidenceItem with BBox)
    print("Testing Grounding...")
    resp = client.post("/analyze", data={"query": "Where is the water body?"}, files=[("files", ("image_0.jpg", img0_bytes, "image/jpeg"))]).json()
    assert resp["task"] == "grounding"
    assert "evidence" in resp and len(resp["evidence"]) > 0
    
    # Check that bounding box exists in at least one evidence item (if it found water)
    # Actually, we just check that the evidence structure matches EvidenceItem.
    ev = resp["evidence"][0]
    assert "claim" in ev
    assert "modality" in ev
    assert ev["status"] in ["VERIFIED", "PARTIALLY_VERIFIED", "UNCERTAIN", "DATA_UNAVAILABLE"]
    print("Grounding OK.")

    # 2. Change Analysis (should return base64 visual and change stats)
    print("Testing Change Analysis...")
    resp = client.post("/analyze", data={"query": "What changed?"}, files=[
        ("files", ("image_0.jpg", img0_bytes, "image/jpeg")),
        ("files", ("image_1.jpg", img1_bytes, "image/jpeg"))
    ]).json()
    assert resp["task"] == "change_analysis"
    assert resp["visual_output"].startswith("data:image/png;base64")
    assert resp["evidence"][0]["claim"].startswith("Change detected")
    print("Change Analysis OK.")
    
    # 3. Optical SAR (should detect conflict if any)
    print("Testing Optical-SAR Conflict...")
    resp = client.post("/analyze", data={"query": "Analyze optical and SAR together"}, files=[
        ("files", ("opt.tif", img0_bytes, "image/tiff")),
        ("files", ("sar.tif", img1_bytes, "image/tiff"))
    ]).json()
    assert resp["task"] == "optical_sar"
    assert "conflict" in resp
    print("Optical-SAR OK.")
    
    # 4. Report Generation
    print("Testing PDF Generation...")
    resp = client.post("/report", data={"query": "What changed?"}, files=[
        ("files", ("image_0.jpg", img0_bytes, "image/jpeg")),
        ("files", ("image_1.jpg", img1_bytes, "image/jpeg"))
    ])
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert len(resp.content) > 1000
    print("PDF Report OK.")

    # 5. History API
    print("Testing History API...")
    resp = client.get("/admin/history").json()
    assert len(resp["history"]) >= 4
    print("History OK.")

    print("\nALL E2E TESTS PASSED.")

if __name__ == "__main__":
    run_e2e_tests()
