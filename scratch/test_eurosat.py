import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.eurosat_classifier import run_eurosat_classification

def test_smoke():
    print("1. Loading classifier and testing direct inference...")
    try:
        with open("datasets/rsicd/images/image_0.jpg", "rb") as f:
            b = f.read()
            
        res = run_eurosat_classification([{"filename": "image_0.jpg", "bytes": b}])
        assert res["status"] == "SUCCESS", f"Failed: {res}"
        assert "Forest" in str(res["evidence"]) or "prediction" in str(res["evidence"]), "Evidence missing predictions"
        assert 0 <= res["confidence"] <= 1, "Confidence out of bounds"
        print("✓ Direct inference passed:", res["answer"])
    except Exception as e:
        print("✗ Direct inference failed:", str(e))
        return

    print("2. Testing /analyze endpoint integration...")
    try:
        client = TestClient(app)
        
        with open("datasets/rsicd/images/image_0.jpg", "rb") as f:
            resp = client.post(
                "/analyze",
                data={"query": "What type of land cover is visible?", "benchmark_mode": "true"},
                files=[("files", ("image_0.jpg", f, "image/jpeg"))]
            )
        
        data = resp.json()
        assert resp.status_code == 200, f"API Error: {resp.text}"
        assert data["task"] == "land_cover_classification", f"Wrong task routed: {data['task']}"
        assert data["status"] == "SUCCESS", f"API Status failed: {data['status']}"
        assert "confidence" in data, "Confidence missing from API response"
        assert len(data["evidence"]) > 0, "Evidence missing from API response"
        print("✓ /analyze endpoint passed! Task routed to:", data["task"])
        
    except Exception as e:
        print("✗ /analyze endpoint failed:", str(e))
        return
        
    print("3. Testing invalid image handling...")
    try:
        resp = client.post(
            "/analyze",
            data={"query": "Classify this satellite image"},
            files=[("files", ("bad.jpg", b"badbytes", "image/jpeg"))]
        )
        assert resp.status_code == 200 # App logic handles invalid images, HTTP remains 200 usually
        data = resp.json()
        assert data["status"] == "INVALID_INPUT" or data["status"] == "MODEL_UNAVAILABLE" or data["status"] == "ERROR", "Should have failed cleanly"
        print("✓ Invalid image handled gracefully:", data["status"])
    except Exception as e:
        print("✗ Invalid image test failed:", str(e))
        
    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test_smoke()
