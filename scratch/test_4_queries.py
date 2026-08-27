import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app

def run_scenario_tests():
    client = TestClient(app)
    
    with open("datasets/rsicd/images/image_0.jpg", "rb") as f1, open("datasets/rsicd/images/image_1.jpg", "rb") as f2:
        img0_bytes = f1.read()
        img1_bytes = f2.read()

    print("\n--- Query 1: Land Cover (EuroSAT) ---")
    resp1 = client.post(
        "/analyze",
        data={"query": "What type of land cover is visible in this image?"},
        files=[("files", ("image_0.jpg", img0_bytes, "image/jpeg"))]
    )
    print("Status:", resp1.status_code)
    data1 = resp1.json()
    print("Task routed:", data1.get("task"))
    print("Answer:", data1.get("answer"))
    print("Confidence:", data1.get("confidence"))

    print("\n--- Query 2: Objects Present (VQA) ---")
    resp2 = client.post(
        "/analyze",
        data={"query": "What objects are present in this image?"},
        files=[("files", ("image_0.jpg", img0_bytes, "image/jpeg"))]
    )
    data2 = resp2.json()
    print("Task routed:", data2.get("task"))
    print("Answer:", data2.get("answer"))

    print("\n--- Query 3: Describe Image (Captioning) ---")
    resp3 = client.post(
        "/analyze",
        data={"query": "Describe this satellite image."},
        files=[("files", ("image_0.jpg", img0_bytes, "image/jpeg"))]
    )
    data3 = resp3.json()
    print("Task routed:", data3.get("task"))
    print("Answer:", data3.get("answer"))

    print("\n--- Query 4: Change Detection ---")
    resp4 = client.post(
        "/analyze",
        data={"query": "If this is a change-detection image pair, identify the changes."},
        files=[
            ("files", ("image_0.jpg", img0_bytes, "image/jpeg")),
            ("files", ("image_1.jpg", img1_bytes, "image/jpeg"))
        ]
    )
    data4 = resp4.json()
    print("Task routed:", data4.get("task"))
    print("Answer:", data4.get("answer"))

if __name__ == "__main__":
    run_scenario_tests()
