# SIH 2026 Readiness Report: SatQuery AI

## 1. Problem Addressed
Accessing, querying, and interpreting remote-sensing data traditionally requires deep geospatial expertise, specialized GIS software, and manual analysis. Natural-language interfaces for satellite imagery have lagged behind standard multimodal LLMs due to domain-specific features.

## 2. Existing Gap
Current Vision-Language Models (VLMs) like BLIP or LLaVA struggle with remote-sensing imagery because they are trained on natural scene datasets (e.g., COCO). They misinterpret satellite viewpoints, fail to detect specialized land cover, and hallucinate object sizes. 

## 3. Proposed Solution
SatQuery AI: A multimodal platform that adapts foundational VLMs to the geospatial domain using parameter-efficient fine-tuning (LoRA). It offers a unified API and frontend to handle geospatial images and natural-language queries seamlessly.

## 4. Architecture
The architecture comprises a React + Vite frontend and a FastAPI backend with dynamic capability routing. 
User Image &rarr; Image Validation &rarr; BLIP Processor &rarr; BLIP Vision Encoder &rarr; BLIP Text Decoder + Trained LoRA &rarr; Remote-Sensing Caption &rarr; Backend API &rarr; Frontend.

## 5. Model Architecture
Base Model: `Salesforce/blip-image-captioning-base`. This comprises a ViT vision encoder and a transformer-based text decoder.

## 6. LoRA Adaptation
A custom native PyTorch LoRA injection script explicitly modifies the attention `query` and `value` projection layers in the model, bypassing standard HF PEFT `inputs_embeds` bugs while keeping the base weights frozen.
- Target Modules: Query, Value
- Rank ($r$): 8
- Alpha ($\alpha$): 16
- Trainable Parameters: 589,824

## 7. Dataset
Dataset: `arampacha/rsicd` (Remote Sensing Image Captioning Dataset). Comprises aerial imagery mapped to human-annotated descriptive captions.

## 8. Training Configuration
- Training Samples: 2000
- Validation Samples: 500
- Epochs: 3
- Hardware: Tesla T4 GPU

## 9. Training Evidence
Actual Training Loss by Epoch:
- Epoch 1: 6.8226
- Epoch 2: 5.8473
- Epoch 3: 5.6354

Actual Validation Loss by Epoch:
- Epoch 1: 6.2482
- Epoch 2: 5.8645
- Epoch 3: 5.7490

## 10. Evaluation
Historical metrics computed on the Kaggle GPU run:
- Baseline BLEU: 0.000000 | Adapted BLEU: 0.323397
- Baseline ROUGE-L: 0.110373 | Adapted ROUGE-L: 0.586496

## 11. Baseline Comparison
Locally verified inference comparison on 20 validation images confirms a 100% change rate in captions. The adapted model consistently replaces generic predictions ("an aerial view of...") with detailed spatial descriptions ("many aircraft are parked in an airport near some buildings...").

## 12. Supported Image Formats
- JPEG / JPG
- PNG
- GeoTIFF / TIFF (Extracts CRS, transforms, and metadata before converting down to RGB for model ingestion).

## 13. Remote-Sensing Capabilities
The system exposes explicit capabilities for optical and SAR imagery, providing accurate provenance data and safely rejecting malformed or unsupported geospatial combinations.

## 14. Query Capabilities
- **Captioning**: (VERIFIED) Full remote-sensing image captioning powered by the RSICD LoRA adapter.
- **VQA / Change Analysis / Grounding**: (IMPLEMENTED BUT NOT BENCHMARKED) Routed correctly by the backend, but awaiting dedicated fine-tuned specialist weights.

## 15. Backend
Built on FastAPI. Implements a capability registry, robust memory management (manual cache clearing after inference), validation middleware, and a unified `/analyze` API endpoint.

## 16. Frontend
A React + Vite workspace explicitly labeled as "RSICD BLIP LORA CAPTIONING". Handles file uploads, dynamically renders JSON provenance, displays confidence meters, and safely manages fallback/offline states.

## 17. Testing
36/36 Backend Pytest suite passing. End-to-end Demo Test passing. Frontend building without errors.

## 18. Limitations
Inference is currently running on a CPU in the local environment, resulting in ~10 second generation times per image. Some advanced tasks (Grounding, Change Detection) currently fallback to placeholders or base unadapted models until their respective LoRA adapters are trained.

## 19. Novelty
Injects LoRA into BLIP entirely natively, avoiding the massive bloat of external PEFT libraries while retaining full compatibility with standard PyTorch formats.

## 20. Future Work
Train and integrate additional LoRA adapters for Change Detection and Visual Question Answering using the BigEarthNet dataset.
