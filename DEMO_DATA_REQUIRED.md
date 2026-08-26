# DEMO_DATA_REQUIRED

To fully demonstrate all capabilities of the SatQuery AI system, please provide the following real data and weights during the final presentation/demo. 

The application strictly enforces a "No Fabrication" rule. Features will remain securely locked and display `MODEL_UNAVAILABLE` or `DATASET_NOT_AVAILABLE` until the legitimate data is provided.

## 1. Imagery Data
- **Optical/Multispectral GeoTIFF:** Required to demonstrate VQA, Grounding, and Captioning routing. Must contain valid CRS and affine transforms.
- **SAR GeoTIFF:** Required alongside the optical GeoTIFF to demonstrate the deterministic Optical-SAR cross-modal analysis.
- **Bi-temporal Optical Pair:** Two GeoTIFF files covering the exact same spatial extent at two different times. Required to demonstrate the Change Analysis baseline.

## 2. Models
Because the host environment (Snapdragon X, limited RAM/Storage) physically rejected the 1.5GB downloads during initialization, the system actively returns `MODEL_UNAVAILABLE`.
To demonstrate *actual* local VQA and Captioning:
- Download the weights for `Salesforce/blip-vqa-base` into your local Hugging Face cache.
- Download the weights for `Salesforce/blip-image-captioning-base` into your local Hugging Face cache.

## 3. BigEarthNet Images
The `datasets/bigearthnet/BigEarthNet.txt.parquet` metadata dataset successfully parsed into `adaptation_manifest.jsonl`, but the raw satellite patches are absent.
- Provide the actual `datasets/bigearthnet/images` folder containing the patch `.tif` files mapped in the manifest to unlock the LoRA/PEFT adaptation module.

## 4. Benchmark Datasets
The evaluation framework is primed but currently reports `DATASET_NOT_AVAILABLE`.
- **VRSBench:** Place the raw dataset into `datasets/vrsbench`
- **RSVQA:** Place the raw dataset into `datasets/rsvqa`
- **CDVQA:** Place the raw dataset into `datasets/cdvqa`
*(Note: ISRO/SAC evaluation data is legitimately withheld by organizers and will correctly display `ORGANIZER_EVALUATION_PENDING`)*.
