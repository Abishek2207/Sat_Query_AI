import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app.local_specialists import run_local_captioning

def test():
    files_data = [{"filename": "image_0.jpg"}]
    
    with open("datasets/rsicd/images/image_0.jpg", "rb") as f:
        img_bytes = f.read()
    
    import backend.app.local_specialists
    original_load = backend.app.local_specialists._load_image
    
    # Mock loader
    def mock_load(fd):
        import io
        from PIL import Image
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    backend.app.local_specialists._load_image = mock_load
    
    print("Testing backend logic...")
    res = run_local_captioning(files_data)
    print("Result:", res)
    assert res["status"] == "SUCCESS", "Backend logic failed!"
    print("Backend test PASS")

if __name__ == "__main__":
    test()
