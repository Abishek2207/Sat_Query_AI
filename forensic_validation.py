import os
import hashlib
import numpy as np
import rasterio

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def inspect_geotiff(filepath, name):
    print(f"\n--- {name} ---")
    print(f"Path: {os.path.abspath(filepath)}")
    if not os.path.exists(filepath):
        print("FILE NOT FOUND.")
        return None
    print(f"Size: {os.path.getsize(filepath)} bytes")
    print(f"SHA256: {sha256_file(filepath)}")
    
    with rasterio.open(filepath) as src:
        print(f"Driver: {src.driver}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Dimensions: {src.width}x{src.height}")
        print(f"Transform: {src.transform}")
        print(f"Pixel Resolution: {src.res}")
        print(f"Dtype: {src.dtypes[0]}")
        print(f"Band Count: {src.count}")
        print(f"NoData: {src.nodatavals}")
        print(f"Tags/Metadata: {src.tags()}")
        
        data = src.read(1)
        print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean():.2f}, Std: {data.std():.2f}")
        return {
            "crs": src.crs,
            "bounds": src.bounds,
            "transform": src.transform,
            "res": src.res,
            "shape": (src.width, src.height)
        }

def main():
    print("==================================================")
    print("FORENSIC VALIDATION STARTING")
    print("==================================================")
    
    p1 = "datasets/sentinel2/optical_before.tif"
    p2 = "datasets/sentinel2/optical_after.tif"
    p3 = "datasets/sentinel1/sar_before.tif"
    
    meta1 = inspect_geotiff(p1, "OPTICAL BEFORE")
    meta2 = inspect_geotiff(p2, "OPTICAL AFTER")
    meta3 = inspect_geotiff(p3, "SAR")
    
    print("\n==================================================")
    print("CO-REGISTRATION VALIDATION")
    print("==================================================")
    if meta1 and meta2:
        print(f"Temporal overlap exact bounds match: {meta1['bounds'] == meta2['bounds']}")
        print(f"Temporal CRS match: {meta1['crs'] == meta2['crs']}")
    if meta1 and meta3:
        print(f"SAR-Optical overlap exact bounds match: {meta1['bounds'] == meta3['bounds']}")
        print(f"SAR-Optical CRS match: {meta1['crs'] == meta3['crs']}")
        
    print("\n==================================================")
    print("REAL BI-TEMPORAL CHANGE BASELINE")
    print("==================================================")
    from backend.app.change_map import compute_change_baseline
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        res = compute_change_baseline(f1.read(), f2.read())
        print(f"Before Date: 2023-07-09")
        print(f"After Date: 2023-07-31")
        print(f"Computed Output: {res.get('answer')}")
        print(f"Evidence: {res.get('evidence')}")
        
    print("\n==================================================")
    print("REAL OPTICAL-SAR MULTIMODAL BASELINE")
    print("==================================================")
    if not os.path.exists(p3):
        print("[DATA_UNAVAILABLE] Valid Sentinel-1 raster unavailable. SAR file is missing.")
    else:
        from backend.app.optical_sar import verify_optical_sar_pair
        with open(p1, "rb") as f1, open(p3, "rb") as f3:
            res2 = verify_optical_sar_pair(f1.read(), f3.read())
            print(f"Computed Output: {res2.get('answer')}")
            for ev in res2.get('evidence', []):
                print(f"Evidence: {ev.get('claim')} -> {ev.get('evidence')}")

if __name__ == "__main__":
    main()
