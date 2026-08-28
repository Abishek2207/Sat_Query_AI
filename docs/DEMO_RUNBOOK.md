# SATQUERY AI - DEMO RUNBOOK

This runbook provides the exact commands and queries to execute the 8 real-world demonstration scenarios required for the SIH 26167 evaluation.

## Prerequisites
1. Backend running: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000`
2. Frontend running: `cd frontend && npm run dev`
3. Test images available in `datasets/test_images/`

---

## Scenarios

### 1. Single Image VQA
* **Input**: `image_1.jpg` (Optical)
* **Query**: `"What objects are present in this image?"`
* **Expected Task Route**: `VQA`
* **Expected Output**: Comma-separated list (e.g., "buildings, roads, trees") via `blip-vqa-base`.

### 2. Captioning / Scene Description
* **Input**: `image_2.jpg` (Optical)
* **Query**: `"Describe this satellite image."`
* **Expected Task Route**: `CAPTIONING`
* **Expected Output**: Remote-sensing adapted description (e.g., "An aerial view of a dense residential neighborhood.") via RSICD-adapted LoRA.

### 3. Text-Guided Grounding
* **Input**: `image_1.jpg` (Optical)
* **Query**: `"Highlight the water body."` (or "Highlight the buildings.")
* **Expected Task Route**: `GROUNDING`
* **Expected Output**: Explicit bounding boxes `[xmin, ymin, xmax, ymax]` rendered in the Satellite Viewer over the image via `grounding-dino`.

### 4. Land-Cover Classification
* **Input**: `image_1.jpg`
* **Query**: `"What is the dominant land cover?"`
* **Expected Task Route**: `LAND_COVER_CLASSIFICATION`
* **Expected Output**: EuroSAT categorization (e.g., "Residential" or "Forest") with float confidence.

### 5. Change Detection (Bi-Temporal)
* **Input**: `image_a.tif` and `image_b.tif` (Co-registered optical pair)
* **Query**: `"What changed between these two images?"`
* **Expected Task Route**: `CHANGE_ANALYSIS`
* **Expected Output**: Changed area percentage + a Base64-encoded visual change mask rendered in the UI overlay.

### 6. Change-Based VQA
* **Input**: `image_a.tif` and `image_b.tif`
* **Query**: `"Has the built-up area increased?"`
* **Expected Task Route**: `CHANGE_ANALYSIS`
* **Expected Output**: If the heuristic change mask provides insufficient semantic context to definitively claim "built-up" expansion, the system will explicitly return `UNCERTAIN` to prevent hallucination.

### 7. Cross-Modal Optical + SAR
* **Input**: `optical.tif` and `sar.tif`
* **Query**: `"Use the optical and SAR images together to identify built-up and water-covered regions."`
* **Expected Task Route**: `OPTICAL_SAR`
* **Expected Output**: If dimensions or geospatial metadata conflict, it will yield `CONFLICT DETECTED`. If co-registered perfectly, it will output a joint baseline finding.

### 8. Download Evidence Report
* **Action**: Click the `DOWNLOAD MISSION REPORT` button after any of the above successful queries.
* **Expected Output**: A PDF file containing the timestamp, mission ID, selected task, exact evidence, and execution trace. Base64 masks and bounding boxes will be embedded directly in the PDF.
