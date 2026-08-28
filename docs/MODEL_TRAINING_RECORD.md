# SATQUERY AI - MODEL TRAINING RECORD

Tracks Kaggle-based fine-tuning and adaptation of Vision-Language Models for Remote Sensing.

## 1. Captioning Adaptation (RSICD)
* **Base Model**: `Salesforce/blip-image-captioning-base`
* **Dataset**: `RSICD`
* **Method**: PEFT / LoRA (Rank=16, Target=`qkv`)
* **Artifact Status**: **AVAILABLE** (Loaded in `backend/app/adapters.py`)
* **Provenance**: Verified via checksum against remote-sensing dictionary metrics.
* **Capabilities**: Generates domain-specific terminology (e.g., "industrial area," "dense residential") rather than generic COCO-style captions.

## 2. Deep Multispectral Adaptation (BigEarthNet)
* **Base Model**: ResNet / ConvNeXt backbone targeting BigEarthNet classification.
* **Dataset**: `BigEarthNet`
* **Artifact Status**: **AWAITING EXECUTION**
* **Reasoning**: The pipeline scripts (`kaggle/train_remote_sensing.py`) are fully written and staged. However, because the physical BigEarthNet image patches (60+ GB) are not present on this evaluation server (only the metadata parquet), the training script cannot be executed without violating the NO-FABRICATION policy. 
* **Action Required**: Mount physical BigEarthNet Sentinel patches to `/kaggle/input` and execute the staged pipeline to generate the artifact.

## 3. Land-Cover Specialist
* **Base Model**: `nielsr/convnext-tiny-finetuned-eurosat`
* **Artifact Status**: **AVAILABLE** (Pre-trained weights downloaded directly from HuggingFace).

## 4. Grounding Specialist
* **Base Model**: `IDEA-Research/grounding-dino-base`
* **Artifact Status**: **AVAILABLE** (Zero-shot evaluation mode utilized for spatial coordinate extraction).
