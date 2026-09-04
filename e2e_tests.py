import subprocess
import time
import requests
import os
import sys

def run_tests():
    print("Starting backend...")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"])
    
    # Wait for server to start
    for _ in range(10):
        try:
            res = requests.get("http://127.0.0.1:8000/health")
            if res.status_code == 200:
                print("Server is up!")
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        print("Failed to start server.")
        proc.kill()
        return

    results = []

    try:
        base_url = "http://127.0.0.1:8000"
        
        def test_endpoint(name, files, query):
            print(f"\n--- {name} ---")
            start = time.time()
            res = requests.post(f"{base_url}/analyze", files=files, data={'query': query})
            elapsed = time.time() - start
            if res.status_code == 200:
                json_res = res.json()
                print("Status:", json_res.get('status'))
                print("Task:", json_res.get('task'))
                print("Selected Tools:", json_res.get('selected_tools'))
                print(f"Latency: {elapsed:.2f}s")
                results.append((name, query, json_res.get('task'), f"{elapsed:.2f}s", json_res.get('status')))
            else:
                print("Failed:", res.status_code)
                results.append((name, query, "N/A", f"{elapsed:.2f}s", "FAIL"))

        # Test 1: VQA
        test_endpoint("VQA", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "How many buildings are visible?")

        # Test 2: Captioning
        test_endpoint("Captioning", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "Describe the scene.")

        # Test 3: Grounding
        test_endpoint("Grounding", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "Highlight the buildings.")
        
        # Test 4: EuroSAT Classification (Assuming we have a query for this or it routes to VQA/Captioning)
        test_endpoint("EuroSAT Classification", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "What is the land cover class?")

        # Test 5: Multi-tool
        test_endpoint("Multi-tool", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "Describe the image and highlight the buildings.")
            
        # Test 6: Temporal
        test_endpoint("Temporal Change", [
            ('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff')),
            ('files', ('optical_after.tif', open('datasets/sentinel2/optical_after.tif', 'rb'), 'image/tiff'))
        ], "What changed between these two images?")

        # Test 7: Optical + SAR
        test_endpoint("Optical-SAR", [
            ('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff')),
            ('files', ('sar_before.tif', open('datasets/sentinel1/sar_before.tif', 'rb'), 'image/tiff'))
        ], "Analyze this area using optical and SAR together.")

        # Test 8: Hallucination
        test_endpoint("Hallucination Trap", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "Is there a hospital in this image?")

        # Test 9: Missing temporal
        test_endpoint("Missing Input (Temporal)", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "What changed between these two images?")
            
        # Test 10: Missing SAR
        test_endpoint("Missing Input (SAR)", [('files', ('optical_before.tif', open('datasets/sentinel2/optical_before.tif', 'rb'), 'image/tiff'))], "Analyze this area using optical and SAR together.")

        print("\n=== SUMMARY ===")
        for r in results:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
            
    finally:
        print("Killing server...")
        proc.kill()
        proc.wait()

if __name__ == "__main__":
    run_tests()
