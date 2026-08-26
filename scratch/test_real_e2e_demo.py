import os
import sys
import json
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app.main import app

def run_demo():
    print("--- RUNNING E2E DEMO TEST ---")
    client = TestClient(app)
    
    test_image_path = "datasets/rsicd/images/image_0.jpg"
    
    with open(test_image_path, "rb") as f:
        file_bytes = f.read()
        
    print(f"Uploading {test_image_path} with query 'caption'")
    
    response = client.post(
        "/analyze",
        data={"query": "caption", "capability": "captioning", "benchmark_mode": "true"},
        files=[("files", ("image_0.jpg", file_bytes, "image/jpeg"))]
    )
    
    if response.status_code == 200:
        result = response.json()
        print("DEMO TEST SUCCESS!")
        print(json.dumps(result, indent=2))
        
        os.makedirs("reports", exist_ok=True)
        with open("reports/demo_e2e_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Result saved to reports/demo_e2e_result.json")
    else:
        print("DEMO TEST FAILED!")
        print(response.status_code, response.text)

if __name__ == "__main__":
    run_demo()
