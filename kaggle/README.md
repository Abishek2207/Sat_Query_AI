# SatQuery AI - Kaggle Training Pipeline

This directory contains the reproducible training pipelines for adapting foundational Vision-Language Models (VLMs) to Remote Sensing data using Kaggle GPUs (T4/P100 x2).

## Principle
Due to compute and bandwidth constraints, we do not perform full end-to-end training on local evaluation environments. Instead, these scripts are designed to be executed inside a Kaggle Notebook where datasets like BigEarthNet (60+ GB) or RSICD are mounted.

## Contents
- `train_captioning.py`: LoRA adaptation of BLIP for RSICD scene description.
- `train_vqa.py`: Fine-tuning script for RSVQA / VRSBench datasets.
- `train_remote_sensing.py`: Deep multispectral/SAR baseline adaptation scripts (BigEarthNet).
- `train_multimodal.py`: Cross-modal fusion script for Optical + SAR (e.g. Cartosat + RISAT).

## Workflow
1. Upload `/kaggle` folder to a Kaggle Notebook.
2. Mount the required dataset (e.g., BigEarthNet, RSICD).
3. Execute `python train_*.py`
4. Download the generated `adapter_model.bin` and `provenance.json` from `/kaggle/working/models/adapters/`
5. Place them locally in `backend/models/adapters/`

*Note: Per the strict NO_FABRICATION policy, the backend will only load these adapters if the `provenance.json` exists and matches the local adapter checksum. We do NOT claim training is complete until these artifacts are physically generated and placed in the project.*
