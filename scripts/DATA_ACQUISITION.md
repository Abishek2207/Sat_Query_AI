# REAL DATA ACQUISITION PROTOCOL

As per the strict zero-hallucination mandate of PS 26167, the system cannot fake the required multi-band geospatial arrays. Because Copernicus Data Space Ecosystem requires secure user authentication, automated edge-node scraping is blocked.

You must manually download the three required genuine GeoTIFF files to unlock the Rasterio deterministic baselines.

## STEP 1: AUTHENTICATE
1. **Website:** `https://dataspace.copernicus.eu/`
2. **Action:** Log in with your registered account credentials.
3. **Navigate:** Open the Copernicus Browser / Data Explorer.

## STEP 2: TEMPORAL / OPTICAL ACQUISITION
1. **Dataset/Product:** Sentinel-2 L2A (Bottom of Atmosphere)
2. **Sensor:** MSI (Multispectral Instrument)
3. **Bands Required:** B02, B03, B04, B08 (10m resolution arrays).
4. **Acquisition 1 (Before):** Select a specific date. Download the GeoTIFF/SAFE file.
   - **Extract To:** `c:\Users\Admin\OneDrive\Desktop\SIH_project\datasets\sentinel2\optical_before.tif`
5. **Acquisition 2 (After):** Select a *different* date over the exact same bounding box. Download the GeoTIFF/SAFE file.
   - **Extract To:** `c:\Users\Admin\OneDrive\Desktop\SIH_project\datasets\sentinel2\optical_after.tif`
6. **Approximate Size:** ~10-50MB per extracted 10m GeoTIFF.

## STEP 3: SAR ACQUISITION
1. **Dataset/Product:** Sentinel-1 C-SAR GRD (Ground Range Detected)
2. **Sensor:** SAR
3. **Bands Required:** VV and VH Polarization arrays.
4. **Acquisition Date:** Select a date matching (or as close as possible to) the `optical_before.tif` footprint.
5. **Extract To:** `c:\Users\Admin\OneDrive\Desktop\SIH_project\datasets\sentinel1\sar_before.tif`
6. **Approximate Size:** ~50MB per extracted subset GeoTIFF.

## STEP 4: SYSTEM HANDOFF
Once the three true `.tif` files are placed precisely in those destination paths, the backend Rasterio pipeline will automatically ingest them, extract their true spatial grids (CRS, bounds, resolution), reproject them if necessary, and compute the `[PASS]` statuses on `verify_real_multimodal.py`.
