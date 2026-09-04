import os
import json
import rasterio
import hashlib

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

out_dir = "datasets/paired_multimodal"
files_to_check = {
    "optical_before.tif": {"bands": 4, "dtype": "uint16"},
    "optical_after.tif": {"bands": 4, "dtype": "uint16"},
    "sar_before.tif": {"bands": 2, "dtype": "float32"},
}

print("Starting validation...")
all_passed = True
hashes = []

base_crs = None
base_transform = None
base_width = None
base_height = None

for fname, specs in files_to_check.items():
    fpath = os.path.join(out_dir, fname)
    if not os.path.exists(fpath):
        print(f"FAIL: {fname} does not exist.")
        all_passed = False
        continue
    
    size_mb = os.path.getsize(fpath) / (1024*1024)
    file_hash = get_sha256(fpath)
    hashes.append(f"{file_hash}  {fname}")
    
    try:
        with rasterio.open(fpath) as src:
            bands = src.count
            dtype = src.dtypes[0]
            width, height = src.width, src.height
            crs = str(src.crs)
            transform = src.transform
            
            # Read first band for stats
            data = src.read(1)
            valid_pixels = (data != src.nodata).sum() if src.nodata is not None else (data > 0).sum()
            min_val, max_val = data.min(), data.max()
            
            if base_crs is None:
                base_crs = crs
                base_transform = transform
                base_width = width
                base_height = height
                
            print(f"\n--- {fname} ---")
            print(f"Size: {size_mb:.2f} MB")
            print(f"Bands: {bands} (Expected: {specs['bands']})")
            print(f"Dtype: {dtype} (Expected: {specs['dtype']})")
            print(f"Dimensions: {width}x{height}")
            print(f"CRS: {crs}")
            print(f"Valid Pixels: {valid_pixels} ({valid_pixels/(width*height)*100:.1f}%)")
            print(f"Min: {min_val}, Max: {max_val}")
            
            if bands != specs["bands"] or dtype != specs["dtype"]:
                print("FAIL: Band count or dtype mismatch.")
                all_passed = False
            if width != base_width or height != base_height:
                print("FAIL: Dimension mismatch.")
                all_passed = False
            if crs != base_crs:
                print("FAIL: CRS mismatch.")
                all_passed = False
            if transform != base_transform:
                print("FAIL: Transform (Grid) mismatch.")
                all_passed = False
            if valid_pixels == 0:
                print("FAIL: File is empty or completely nodata/zero-filled.")
                all_passed = False
                
    except Exception as e:
        print(f"FAIL: Could not read {fname}. Error: {e}")
        all_passed = False

with open(os.path.join(out_dir, "SHA256SUMS.txt"), "w") as f:
    f.write("\n".join(hashes) + "\n")

if all_passed:
    print("\nFINAL STATUS: PASS")
else:
    print("\nFINAL STATUS: FAIL")
