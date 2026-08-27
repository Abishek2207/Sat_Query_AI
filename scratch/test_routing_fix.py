import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app

def run_scenario_tests():
    client = TestClient(app)
    
    with open("datasets/rsicd/images/image_0.jpg", "rb") as f1:
        img0_bytes = f1.read()

    queries = [
        "What type of land cover is visible in this image?",
        "What is the land surface is this?",
        "Which land type is shown?",
        "What objects are present in this image?",
        "Describe this satellite image.",
        "Classify this satellite image.",
        "What kind of area is this?"
    ]
    
    expected_tasks = [
        "land_cover_classification",
        "land_cover_classification",
        "land_cover_classification",
        "vqa",
        "captioning",
        "land_cover_classification",
        "land_cover_classification"
    ]

    all_passed = True

    for q, exp in zip(queries, expected_tasks):
        print(f"\n--- Query: '{q}' ---")
        resp = client.post(
            "/analyze",
            data={"query": q},
            files=[("files", ("image_0.jpg", img0_bytes, "image/jpeg"))]
        )
        data = resp.json()
        task = data.get("task")
        
        print(f"Task routed: {task}")
        print(f"Expected: {exp}")
        
        if task == "land_cover_classification":
            print(f"Answer: {data.get('answer')}")
            print(f"Model: {data.get('provenance', {}).get('model')}")
            
        if task != exp:
            print(">>> FAILED: Wrong task routed!")
            all_passed = False

    if all_passed:
        print("\nALL ROUTING TESTS PASSED EXACTLY AS EXPECTED!")
    else:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_scenario_tests()
