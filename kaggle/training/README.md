# Kaggle Training Pipeline for SatQuery AI

This pipeline reproduces the remote sensing adaptation workflows required by SIH 26167 using Kaggle GPUs (T4x2 or P100).

## Datasets Supported
- **BigEarthNet** (Multispectral / SAR Baseline)
- **RSICD** (Captioning / LoRA adaptation)
- **RSVQA / VRSBench** (VQA Fine-tuning)

## Training Process
1. Upload this `/kaggle` folder directly to your Kaggle Workspace.
2. Add the corresponding dataset (e.g., `BigEarthNet.txt`) via "Add Data".
3. Install dependencies: `pip install -r requirements.txt`
4. Run the target script (e.g., `python train_remote_sensing.py`).

## Mixed Precision & GPU Scaling
- Scripts explicitly use `fp16=True` via `accelerate` and `Trainer` API to maximize T4 GPU throughput.
- We utilize `peft` for LoRA adapters rather than full fine-tuning on large Vision-Language models.

## Export & Deployment
- Checkpoints are saved to `/kaggle/working/models/`.
- Download the generated artifact `adapter_model.bin` (or equivalent) alongside its `provenance.json`.
- Place them in `Sat_Query_AI/backend/models/`.
- The `model_registry.py` will dynamically load and mark them as `READY` in the UI upon startup.

**No-Fabrication Policy:**
These scripts refuse to run if the physical dataset is missing. Do not bypass the data check to simulate completion.
