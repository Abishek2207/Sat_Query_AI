# SATQUERY AI - EVALUATION RECORD

## Benchmark Framework
The system implements a rigorous, file-backed evaluation framework under `backend/app/evaluations/runner.py`.

### 1. RSICD Captioning Evaluation
* **Status**: **EXECUTED**
* **Evaluated Samples**: 500
* **Metric**: Exact Match / Diagnostic Overlap
* **Outcome**: Achieved significant overlap matching domain-specific remote sensing vocabulary (e.g., "airport", "commercial area", "dense vegetation").
* **Score Record**: Authenticated against `evalResult` object inside frontend UI.

### 2. BigEarthNet Classification
* **Status**: **NOT_EVALUATED (PARTIAL)**
* **Reason**: While `.parquet` manifests exist, the pipeline halted evaluation because raw `images/` directory is missing. The system accurately rendered this as `PARTIAL - MISSING IMAGE SAMPLES`.

### 3. VRSBench / RSVQA
* **Status**: **NOT_AVAILABLE**
* **Reason**: No physical ground truth or evaluation imagery present locally.

### Methodology Guardrails
- **No placeholder scores**: A missing dataset returns a JSON object with `metric: null` and `score: null`.
- **Reproducibility**: Evaluations loop explicitly through local manifests, feed them individually through the standard inference graph (`call_specialist_model`), and programmatically aggregate metric averages.
