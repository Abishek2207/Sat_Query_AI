# SIH 26167 FINAL COMPLIANCE MATRIX

This matrix maps every SIH requirement to its physical code implementation and execution trace.

| Requirement | Implementation | File | Model | Dataset | Input | Output | Test | Status | Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VQA** | Agent routes image/query to VLM endpoint. | `agent.py` | `blip-vqa-base` | RSICD | Optical Image | String Answer + Conf | `test_gateway.py::test_vqa_routing` | IMPLEMENTED | Token limit (VRAM) |
| **Captioning** | Custom LoRA weights generating RS-specific descriptions. | `adapters.py` | `blip-image-captioning-base` + PEFT | RSICD | Optical Image | String Caption | `test_gateway.py::test_captioning_routing` | IMPLEMENTED | None |
| **Grounding** | Zero-shot bounding box extraction on RS vocabulary. | `adapters.py` | `grounding-dino-base` | General | Image + Query | `[x,y,w,h]` bounding box | `test_gateway.py::test_grounding_routing` | IMPLEMENTED | None |
| **Land Cover** | Targeted EuroSAT categorization. | `adapters.py` | `convnext-tiny-finetuned-eurosat` | EuroSAT | Optical Image | Label String | `test_gateway.py::test_land_cover_routing` | IMPLEMENTED | 43-class limit |
| **Change Det.** | Pixel-wise baseline spatial derivation. | `local_specialists.py` | Heuristic Baseline | Any | Co-registered pair | Mask Base64 PNG | `test_baselines.py::test_change_map_visual` | IMPLEMENTED | Not Deep Learning |
| **Change VQA** | Combines mask stats and base image into text query. | `agent.py` | Baseline + VLM | Any | Co-registered pair | VQA Result | `test_gateway.py::test_change_analysis` | IMPLEMENTED | Hallucination risks mitigated by abstention |
| **Optical+SAR** | Validates metadata bounds and yields explicit evidence. | `local_specialists.py` | Heuristic Baseline | Any | Optical + SAR | Joint Evidence | `test_gateway.py::test_optical_sar` | IMPLEMENTED | True deep fusion weights missing |
| **LangGraph** | E2E strict deterministic graph validation. | `agent.py` | N/A | N/A | Graph State | `AnalysisResponse` | `test_gateway.py` | IMPLEMENTED | None |
| **GeoTIFF** | Uses GDAL/rasterio for band and transform extraction. | `validator.py` | N/A | Any | `.tif` File | ValidationResult | `test_gateway.py::test_synthetic_geotiff` | IMPLEMENTED | Fails if missing Affine |
| **Kaggle Train** | Remote-Sensing scripts exported to Jupyter environment. | `train_remote_sensing.py` | ConvNeXt | BigEarthNet | BigEarthNet images | Weights | N/A | PARTIAL | Awaiting physical image data to run |
| **Dataset Reg.** | FS-level scanning marking missing items as NOT_AVAILABLE. | `dataset_registry.py` | N/A | All | OS Path | JSON Registry | `test_dataset.py::test_lazy_initialization` | IMPLEMENTED | None |
| **Model Reg.** | Loads artifacts dynamically and reports `health`. | `model_registry.py` | All | N/A | System State | JSON Status | API Health Endpoint | IMPLEMENTED | None |
| **No-Fabrication**| Safely traps synthetic coords, scores, or false datasets. | `evidence_policy.py` | N/A | N/A | `EvidenceItem` | Exception / `UNCERTAIN` | Graph Unit Tests | IMPLEMENTED | None |
| **UI Design** | Vite/React dashboard with aerospace aesthetic (3-zone). | `App.jsx` | N/A | N/A | HTTP / DOM | Render | Manual Demo | IMPLEMENTED | None |
| **Report PDF** | Base64-injected ReportLab dynamic generation. | `report.py` | N/A | N/A | `AnalysisResponse`| PDF Blob | E2E Endpoint | IMPLEMENTED | None |
| **ISRO Data** | Custom optical-SAR validation hooks prepared. | `adapters.py` | TBD | ISRO/SAC | TBD | NOT_AVAILABLE | Manual Review | MISSING | Restricted public access |
