import json
import time
import requests

def test_endpoint(task, query, img_path, files_list):
    try:
        files = []
        for f in files_list:
            files.append(('files', (f, open(f, 'rb'), 'image/jpeg')))
        data = {'query': query, 'benchmark_mode': 'true'}
        start = time.time()
        res = requests.post("http://127.0.0.1:8000/analyze", data=data, files=files)
        dur = time.time() - start
        
        if res.status_code == 200:
            rj = res.json()
            return {
                "task": task,
                "input_image": img_path,
                "query": query,
                "model": rj.get("provenance", {}).get("model", "unknown"),
                "real_inference": True if rj.get("status") == "SUCCESS" else False,
                "status": rj.get("status"),
                "output": rj.get("answer"),
                "duration": f"{dur:.2f}s"
            }
        else:
            return {
                "task": task,
                "input_image": img_path,
                "query": query,
                "model": "unknown",
                "real_inference": False,
                "status": "ERROR",
                "output": f"HTTP {res.status_code}: {res.text}",
                "duration": f"{dur:.2f}s"
            }
    except Exception as e:
        return {
            "task": task,
            "input_image": img_path,
            "query": query,
            "model": "unknown",
            "real_inference": False,
            "status": "EXCEPTION",
            "output": str(e),
            "duration": "0s"
        }

def run_tests():
    results = []
    
    # Wait for server to be ready
    for i in range(10):
        try:
            requests.get("http://127.0.0.1:8000/health")
            break
        except:
            time.sleep(1)
            
    print("Testing VQA...")
    r1 = test_endpoint("vqa", "How many trees are there?", "datasets/rsicd/images/image_1.jpg", ["datasets/rsicd/images/image_1.jpg"])
    results.append(r1)
    
    print("Testing Captioning...")
    r2 = test_endpoint("captioning", "Caption this image", "datasets/rsicd/images/image_2.jpg", ["datasets/rsicd/images/image_2.jpg"])
    results.append(r2)
    
    print("Testing Grounding...")
    r3 = test_endpoint("grounding", "Find all trees", "datasets/rsicd/images/image_1.jpg", ["datasets/rsicd/images/image_1.jpg"])
    results.append(r3)

    # Change Analysis
    print("Testing Change Analysis...")
    r4 = test_endpoint("change_analysis", "What changed?", "image_1, image_2", ["datasets/rsicd/images/image_1.jpg", "datasets/rsicd/images/image_2.jpg"])
    results.append(r4)

    # Optical SAR
    print("Testing Optical+SAR...")
    r5 = test_endpoint("optical_sar", "compare optical and sar structure", "image_1, image_2", ["datasets/rsicd/images/image_1.jpg", "datasets/rsicd/images/image_2.jpg"])
    results.append(r5)

    with open("datasets/rsicd/real_e2e_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    for r in results:
        print(f"[{r['task'].upper()}] {r['status']} -> {r['output'][:100]}")

if __name__ == "__main__":
    run_tests()
