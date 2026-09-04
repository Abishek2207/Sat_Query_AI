import os
import json
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import pystac_client
import planetary_computer
import numpy as np

# Coordinates for a region in California (Sacramento area, often has agricultural changes)
lon, lat = -121.5, 38.5

# Time ranges
time_before = "2023-05-01/2023-05-30"
time_after = "2024-05-01/2024-05-30"

out_dir = "datasets/paired_multimodal"
os.makedirs(out_dir, exist_ok=True)

# 1. Search Planetary Computer for Sentinel-2
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search S2 Before
search_s2_before = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime=time_before,
    query={"eo:cloud_cover": {"lt": 5}}
)
items_s2_before = list(search_s2_before.items())
if not items_s2_before:
    print("No S2 Before items found.")
    exit(1)
s2_item_b = items_s2_before[0]

# Search S2 After
search_s2_after = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime=time_after,
    query={"eo:cloud_cover": {"lt": 5}}
)
items_s2_after = list(search_s2_after.items())
if not items_s2_after:
    print("No S2 After items found.")
    exit(1)
s2_item_a = items_s2_after[0]

# 2. Search Planetary Computer for Sentinel-1 RTC
search_s1_before = catalog.search(
    collections=["sentinel-1-rtc"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime=time_before
)
items_s1_before = list(search_s1_before.items())
if not items_s1_before:
    print("No S1 Before items found.")
    exit(1)
s1_item_b = items_s1_before[0]

search_s1_after = catalog.search(
    collections=["sentinel-1-rtc"],
    intersects={"type": "Point", "coordinates": [lon, lat]},
    datetime=time_after
)
items_s1_after = list(search_s1_after.items())
if not items_s1_after:
    print("No S1 After items found.")
    exit(1)
s1_item_a = items_s1_after[0]

print(f"Selected S2 Before: {s2_item_b.id}")
print(f"Selected S2 After: {s2_item_a.id}")
print(f"Selected S1 Before: {s1_item_b.id}")
print(f"Selected S1 After: {s1_item_a.id}")

# 3. Define standard grid (256x256 patch, 10m res)
# We will use the CRS of the S2 Before image.
# We center the patch at the item's center to ensure data coverage.
b04_url_b = s2_item_b.assets["B04"].href
with rasterio.open(b04_url_b) as src:
    s2_crs = src.crs
    # Center of the S2 image
    cx, cy = src.xy(src.height // 2, src.width // 2)
    # Define a 2560m x 2560m bounding box
    half_size = 256 * 10 / 2
    left, right = cx - half_size, cx + half_size
    bottom, top = cy - half_size, cy + half_size
    
    # Calculate window
    window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
    # Round to integer window
    col_off, row_off = int(window.col_off), int(window.row_off)
    width, height = 256, 256
    int_window = Window(col_off, row_off, width, height)
    
    # The new transform for this 256x256 window
    out_transform = src.window_transform(int_window)

# 4. Download S2 files (B02, B03, B04, B08)
bands_s2 = ["B02", "B03", "B04", "B08"]

def download_s2(item, out_path):
    print(f"Downloading S2 {item.id} to {out_path}")
    data = np.zeros((4, height, width), dtype=np.uint16)
    for i, band in enumerate(bands_s2):
        with rasterio.open(item.assets[band].href) as src:
            # We warp just in case the S2 After is in a different tile/grid,
            # though usually it's the same if it's the same tile ID. We will use WarpedVRT to be safe.
            with WarpedVRT(src, crs=s2_crs, transform=out_transform, width=width, height=height, resampling=Resampling.nearest) as vrt:
                data[i] = vrt.read(1)
                
    # Save 4-band TIFF
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'nodata': 0,
        'width': width,
        'height': height,
        'count': 4,
        'crs': s2_crs,
        'transform': out_transform,
        'compress': 'deflate',
        'tiled': True
    }
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, 'B02')
        dst.set_band_description(2, 'B03')
        dst.set_band_description(3, 'B04')
        dst.set_band_description(4, 'B08')
        
# 5. Download S1 files (VV, VH)
bands_s1 = ["vv", "vh"]
def download_s1(item, out_path):
    print(f"Downloading S1 {item.id} to {out_path}")
    data = np.zeros((2, height, width), dtype=np.float32)
    for i, band in enumerate(bands_s1):
        if band in item.assets:
            url = item.assets[band].href
        else:
            # Try uppercase
            url = item.assets[band.upper()].href
            
        with rasterio.open(url) as src:
            # S1 is often in a different grid, so we strictly warp it to S2's grid
            with WarpedVRT(src, crs=s2_crs, transform=out_transform, width=width, height=height, resampling=Resampling.bilinear) as vrt:
                data[i] = vrt.read(1)
                
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': 0.0,
        'width': width,
        'height': height,
        'count': 2,
        'crs': s2_crs,
        'transform': out_transform,
        'compress': 'deflate',
        'tiled': True
    }
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, 'VV')
        dst.set_band_description(2, 'VH')

download_s2(s2_item_b, os.path.join(out_dir, "optical_before.tif"))
download_s2(s2_item_a, os.path.join(out_dir, "optical_after.tif"))
download_s1(s1_item_b, os.path.join(out_dir, "sar_before.tif"))
download_s1(s1_item_a, os.path.join(out_dir, "sar_after.tif"))

# Create Metadata JSON
metadata = {
  "dataset_name": "Sacramento_Temporal_Optical_SAR",
  "source": "Microsoft Planetary Computer",
  "license": "Open Data",
  "location": f"Lon {lon}, Lat {lat}",
  "optical": {
    "sensor": "Sentinel-2",
    "product": "L2A",
    "bands": ["B02", "B03", "B04", "B08"],
    "resolution_m": 10,
    "before_date": s2_item_b.datetime.isoformat(),
    "after_date": s2_item_a.datetime.isoformat()
  },
  "sar": {
    "sensor": "Sentinel-1",
    "product": "RTC",
    "polarizations": ["VV", "VH"],
    "resolution_m": 10,
    "before_date": s1_item_b.datetime.isoformat(),
    "after_date": s1_item_a.datetime.isoformat()
  },
  "coregistration": {
    "crs": str(s2_crs),
    "grid": f"{width}x{height}, 10m",
    "method": "rasterio WarpedVRT onto optical_before grid (nearest for S2, bilinear for S1)"
  }
}
with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Dataset successfully acquired!")
