# Kaggle to SatQuery AI Deployment Workflow

This document traces the exact pipeline from remote-sensing dataset adaptation in Kaggle to real-time inference in the SatQuery AI backend.

### 1. Kaggle Training Phase
1. Initialize a Kaggle Notebook with a GPU accelerator.
2. Mount the target dataset (e.g., BigEarthNet).
3. Execute the training script:
   ```bash
   python kaggle/training/train_remote_sensing.py
   ```
4. Verify the output metrics and physical artifact generation in `/kaggle/working/models/`.

### 2. Artifact Export
1. Download the output directory as a ZIP file.
2. Ensure the package contains:
   - `adapter_model.bin` / `pytorch_model.bin`
   - `adapter_config.json` / `config.json`
   - `provenance.json` (Required by `evidence_policy.py`)

### 3. Backend Integration
1. Extract the artifacts into the local host:
   ```bash
   unzip models.zip -d backend/models/adapters/<model_name>
   ```
2. Register the artifact in `backend/app/model_registry.py` under the appropriate task key.
3. Define the physical path and `checksum` required for verification.

### 4. API Inference
1. Boot the FastAPI backend:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
2. The `model_registry` scans the directory on startup. If the physical artifact and provenance match, the status transitions to `READY`.

### 5. Frontend UI Verification
1. Navigate to the **Model Registry** module in the GUI.
2. Confirm the selected task specifies the customized `model_version` (e.g., "1.0 (LoRA)").
3. When queried, the execution trace (`trace` JSON block) explicitly captures:
   `ROUTING: Selected task '...' -> MODEL SELECTION: Loaded custom artifact from ...`

**No-Fabrication Policy:** The system explicitly guards against hallucinating adapted responses. If the checksum fails or the directory is missing, the backend safely traps the request and throws `MODEL UNAVAILABLE`.
