import json
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import dataset_info

def verify_vrsbench():
    repo_id = "xiang709/VRSBench"
    print(f"Verifying repository: {repo_id}")
    
    # 1. Info and License
    try:
        info = dataset_info(repo_id)
        print(f"Repository exists.")
        tags = getattr(info, 'tags', [])
        print(f"Tags/License information: {[t for t in tags if 'license' in t.lower()]}")
    except Exception as e:
        print(f"Failed to get info: {e}")
        
    # 2. Configs and Splits
    try:
        configs = get_dataset_config_names(repo_id)
        print(f"Configs: {configs}")
    except Exception as e:
        configs = ["default"]
        print(f"Config error, assuming default: {e}")
        
    try:
        splits = get_dataset_split_names(repo_id)
        print(f"Splits: {splits}")
    except Exception as e:
        print(f"Split error: {e}")
        
    # 3. Stream 3 samples
    print("\nStreaming 3 samples...")
    try:
        # Some datasets don't have multiple configs, or have 'default'
        ds = load_dataset(repo_id, split="train", streaming=True)
        
        samples_collected = []
        for i, sample in enumerate(ds):
            if i >= 3:
                break
            # Remove raw image bytes from print to avoid flooding stdout, 
            # but verify its presence
            has_image = 'image' in sample and sample['image'] is not None
            
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
    verify_vrsbench()
