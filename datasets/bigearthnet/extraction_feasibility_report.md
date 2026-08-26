# BigEarthNet Image Extraction Feasibility Report

## 1. Requested BigEarthNet Identifiers
Extracted directly from `datasets/bigearthnet/adaptation_manifest.jsonl`:
- **ID 6253:** 
  - Optical: `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_77`
  - SAR: `S1B_IW_GRDH_1SDV_20170612T165809_33UUP_40_77`
- **ID 4685:** 
  - Optical: `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_38_72`
  - SAR: `S1B_IW_GRDH_1SDV_20170612T165809_33UUP_38_72`
- **ID 1732:** 
  - Optical: `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_33_59`
  - SAR: `S1B_IW_GRDH_1SDV_20170612T165809_33UUP_33_59`

## 2. Official Source & Distribution Mechanism
The official, canonical BigEarthNet V1.0 datasets are hosted on **Zenodo** (Record 12687186) and TU Berlin.
- **Optical (Sentinel-2) Archive URL:** `https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz`
- **SAR (Sentinel-1) Archive URL:** `https://zenodo.org/records/12687186/files/BigEarthNet-S1-v1.0.tar.gz`

## 3. Storage Analysis
- **100 Verified Pairs:** ~600 MB of raw `.tif` tiles.
- **500 Verified Pairs:** ~3.0 GB of raw `.tif` tiles.
- **Current Available Disk Space:** ~13.0 GB on C: drive.
*(While storing 500 pairs would easily fit the host drive, obtaining them is the technical blocker, detailed below).*

## 4. Feasibility of Selective Download
**Selective download is IMPOSSIBLE.**
- **Technical Evidence:** The official dataset is distributed as monolithic `.tar.gz` (Gzip compressed Tarball) files, measuring **65.3 GB (S2)** and **59.0 GB (S1)**. 
- While Zenodo HTTP servers support `Accept-Ranges: bytes`, **Gzip compression does not mathematically support random byte access.** Decompressing a `.tar.gz` to search for a specific `S2A_MSIL2A_...` folder requires sequentially streaming and discarding network bytes from the start of the file. 
- Finding a patch near the end of the archive would require downloading up to 65 GB over the network. Repeating this sequential network stream for 500 patches is technologically unviable and would effectively download terabytes of wasted stream data.
- **Alternative Open Distributions:** Unofficial mirrors on Hugging Face (e.g. `danielz01/BigEarthNet-S2-v1.0`) encapsulate the raw patches into monolithic `.parquet` data blobs, which also prevents raw file extraction without downloading or loading large shard partitions. 

## 5. Conclusion
**STATUS: DATA_REQUIRED**
I have successfully identified the real IDs and mapped the exact official URLs, but I have forcefully stopped the pipeline. 

Because we cannot bypass the gzip random-access limitation, and because downloading the full 124 GB tarballs exceeds the 13 GB drive capacity, no actual image patches can be verifiably acquired right now. 

I will **not** fabricate missing samples, synthesize data, or substitute unrelated imagery. The system remains locked in a fully truthful, scientifically valid wait state until the required image folders are manually provided to `datasets/bigearthnet/images/`.
