import time
import sys
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

def run_test():
    try:
        start = time.time()
        print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Salesforce/blip-vqa-base"
        
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        
        image_path = "datasets/rsicd/images/image_1.jpg"
        question = "How many trees are there?"
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, question, return_tensors="pt")
        
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        duration = time.time() - start
        
        print("--- VQA Test Results ---")
        print(f"Model: {model_name}")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Duration: {duration:.2f}s")
        
    except Exception as e:
        print("VQA Test Failed:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    run_test()
