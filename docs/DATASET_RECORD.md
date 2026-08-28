# SATQUERY AI - DATASET RECORD

This log accurately tracks the physical presence and preparation state of datasets required by SIH Problem Statement 26167.

| Dataset | Modality | Local Status | Reason | Physical Image Count | Manifests Present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BigEarthNet** | Multispectral/SAR | `PARTIAL` | IMAGE PATCHES REQUIRED | 0 | Yes (`.parquet`) |
| **VRSBench** | Optical | `NOT_AVAILABLE` | MANUAL DOWNLOAD REQUIRED | 0 | No |
| **RSVQA-LR** | Optical | `NOT_AVAILABLE` | MANUAL DOWNLOAD REQUIRED | 0 | No |
| **RSVQA-HR** | Optical | `NOT_AVAILABLE` | MANUAL DOWNLOAD REQUIRED | 0 | No |
| **CDVQA** | Optical Pair | `NOT_AVAILABLE` | MANUAL DOWNLOAD REQUIRED | 0 | No |
| **ISRO/SAC** | Cartosat/RISAT | `NOT_AVAILABLE` | RESTRICTED TARGET | 0 | No |
| **RSICD** | Optical | `READY` | FULLY LOADED | 500+ | Yes |

### Workflow Note
We explicitly forbid the generation of synthetic dummy images to trick the system into a `READY` state. Datasets remain `NOT_AVAILABLE` until the physical `.jpg`/`.tif` binaries are mounted to `datasets/<id>/images/`.
