import time
from datasets import load_dataset, get_dataset_config_names

def test_hf_datasets():
    print("Testing Bingsu/BigEarthNet streaming...")
    try:
        # Load dataset in streaming mode so we don't download everything
        ds = load_dataset("Bingsu/BigEarthNet", "rgb", split="train", streaming=True)
        # Check first record structure
        for record in ds:
            print("Record keys:", record.keys())
            print("Sample record name (if available):", record.get("name") or record.get("patch_id") or "Not found")
            break
    except Exception as e:
        print("Error with Bingsu/BigEarthNet:", e)
        
    print("\nTesting alkzar90/BigEarthNet-S2 streaming...")
    try:
        # Another popular BigEarthNet dataset
        ds = load_dataset("alkzar90/BigEarthNet-S2", split="train", streaming=True)
        for record in ds:
            print("Record keys:", record.keys())
            print("Sample record name (if available):", record.get("name") or record.get("patch_id") or "Not found")
            break
    except Exception as e:
        print("Error with alkzar90/BigEarthNet-S2:", e)

    print("\nTesting m-m-s/BigEarthNet streaming...")
    try:
        ds = load_dataset("m-m-s/BigEarthNet", split="train", streaming=True)
        for record in ds:
            print("Record keys:", record.keys())
            break
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_hf_datasets()
