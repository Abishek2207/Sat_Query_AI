# Evaluation Framework — SatQuery AI
# SIH Problem Statement 26167

## Overview

This directory contains the benchmark evaluation structure for SatQuery AI.

| Benchmark | Task | Status |
|---|---|---|
| RSVQA | Remote-Sensing Visual Question Answering | NOT RUN — dataset not downloaded |
| VRSBench | VQA on diverse RS scenes | NOT RUN — dataset not downloaded |
| CDVQA | Change Detection VQA | NOT RUN — dataset not downloaded |
| ISRO/SAC | Custom Indian RS evaluation dataset | NOT RUN — awaiting data |

## How to Run (when datasets are available)

```bash
python evaluation/rsvqa/evaluate.py --model_endpoint http://localhost:8001/predict
python evaluation/vrsbench/evaluate.py --model_endpoint http://localhost:8001/predict
python evaluation/cdvqa/evaluate.py --model_endpoint http://localhost:8001/predict
```

## Important Rules

- Do NOT fabricate benchmark scores.
- Do NOT mark any evaluation as PASSED unless the script actually executes.
- All metric files must contain real computed values.
