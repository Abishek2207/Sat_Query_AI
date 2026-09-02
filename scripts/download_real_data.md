# MANUAL DOWNLOAD INSTRUCTIONS: REAL MULTIMODAL DATA

Because Copernicus Open Access Hub mandates secure user authentication, automated scraping of gigabyte-scale Sentinel data is physically blocked for this edge node. As per the strict zero-hallucination anti-fake mandate, you **must** download these exact files manually before the pipeline will clear `DATA_UNAVAILABLE`.

## 1. SOURCE
**Copernicus Data Space Ecosystem:** `https://dataspace.copernicus.eu/`

## 2. AOI REQUIREMENT
You must select a single bounding box (e.g., an airport, port, or urban center) and maintain that EXACT geographic footprint across all three downloads to ensure spatial overlap validation passes.

## 3. FILE 1: OPTICAL BEFORE (Sentinel-2)
- **Product Type:** Sentinel-2 L2A (Bottom of Atmosphere)
- **Required Bands:** B02, B03, B04, B08 (10m resolution)
- **Expected File Format:** GeoTIFF / `.tif`
- **Extraction Instructions:** Unzip the `.SAFE` folder, locate the primary 10m TCI/multispectral `.tif` array.
- **Destination Path:** `datasets/sentinel2/optical_before.tif`

## 4. FILE 2: OPTICAL AFTER (Sentinel-2)
- **Product Type:** Sentinel-2 L2A (Bottom of Atmosphere)
- **Required Dates:** A different acquisition date than FILE 1.
- **Expected File Format:** GeoTIFF / `.tif`
- **Destination Path:** `datasets/sentinel2/optical_after.tif`

## 5. FILE 3: SAR (Sentinel-1)
- **Product Type:** Sentinel-1 C-SAR GRD (Ground Range Detected)
- **Required Bands:** VV + VH Polarization
- **Required Dates:** As close to FILE 1 as possible.
- **Expected File Format:** GeoTIFF / `.tif`
- **Destination Path:** `datasets/sentinel1/sar_before.tif`
