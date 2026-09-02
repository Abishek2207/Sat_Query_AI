import os
import rasterio
from rasterio.windows import Window
import rasterio.warp
import numpy as np
import planetary_computer as pc
from pystac_client import Client

S2_DIR = "datasets/sentinel2"
S1_DIR = "datasets/sentinel1"
os.makedirs(S2_DIR, exist_ok=True)
os.makedirs(S1_DIR, exist_ok=True)

def download_optical(url, out_path):
    print(f"Downloading Optical {url} to {out_path}...")
    href = pc.sign(url)
    with rasterio.open(href) as src:
        w, h = src.width, src.height
        window = Window(w//2 - 112, h//2 - 112, 224, 224)
        data = src.read(window=window)
        profile = src.profile
        profile.update({
            'height': 224,
            'width': 224,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data)
    return profile

def download_sar_aligned(url, out_path, opt_profile):
    print(f"Downloading SAR {url} and aligning to Optical bounds...")
    href = pc.sign(url)
    with rasterio.open(href) as src:
        target_crs = opt_profile['crs']
        target_transform = opt_profile['transform']
        target_height = opt_profile['height']
        target_width = opt_profile['width']
        
        data = np.zeros((1, target_height, target_width), dtype=src.dtypes[0])
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=rasterio.warp.Resampling.bilinear
        )
        
        profile = src.profile
        profile.update({
            'crs': target_crs,
            'transform': target_transform,
            'height': target_height,
            'width': target_width,
            'count': 1
        })
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data)
    return np.count_nonzero(data) > 0

def main():
    print("Querying Planetary Computer STAC...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    bbox = [-122.27, 47.54, -122.17, 47.64]
    
    print("Searching for Sentinel-1 RTC...")
    search_s1 = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime="2023-01-01/2023-12-31", max_items=1)
    s1_items = list(search_s1.items())
    if not s1_items: return
        
    print("Searching for Sentinel-2 L2A...")
    search_s2 = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime="2023-01-01/2023-12-31", query={"eo:cloud_cover": {"lt": 10}}, max_items=2)
    s2_items = list(search_s2.items())
    if len(s2_items) < 2: return
    s2_items.sort(key=lambda x: x.datetime)
    
    prof = download_optical(s2_items[0].assets["B04"].href, os.path.join(S2_DIR, "optical_before.tif"))
    download_optical(s2_items[1].assets["B04"].href, os.path.join(S2_DIR, "optical_after.tif"))
    
    if download_sar_aligned(s1_items[0].assets["vv"].href, os.path.join(S1_DIR, "sar_before.tif"), prof):
        print("All downloads successful and verified non-zero.")

if __name__ == "__main__":
    main()
