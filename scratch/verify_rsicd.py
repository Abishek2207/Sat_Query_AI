import json
from datasets import load_dataset

def verify_rsicd():
    print("Verifying RSICD...")
    try:
        ds = load_dataset("arampacha/rsicd", split="train", streaming=True)
        
        samples_collected = []
        for i, sample in enumerate(ds):
            if i >= 3:
                break
            
            clean_sample = {}
            for k, v in sample.items():
                if k == 'image':
                    clean_sample[k] = f"Present (type: {type(v).__name__})"
                else:
                    clean_sample[k] = v
            samples_collected.append(clean_sample)
            
        print("\nSchema of first sample:")
        if samples_collected:
            for k, v in samples_collected[0].items():
                print(f" - {k}: {type(v).__name__}")
                
        print("\nSample Data:")
        print(json.dumps(samples_collected, indent=2))
        
    except Exception as e:
        print(f"Streaming failed: {e}")

if __name__ == "__main__":
    verify_rsicd()
