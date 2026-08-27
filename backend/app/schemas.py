from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class Provenance(BaseModel):
    model: str
    model_version: str
    adaptation_dataset: Optional[str] = None
    adaptation_method: Optional[str] = None
    remote_sensing_adapted: bool
    inference_timestamp: str
    input_filenames: List[str]
    input_modalities: List[str]
    crs: Optional[str] = None
    geospatial_evidence_generated: bool
    device: Optional[str] = None

class ValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = None
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    bands: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    modality: str = "unknown"
    acquisition_time: Optional[str] = None
    file_format: Optional[str] = None

class EvidenceItem(BaseModel):
    claim: str
    evidence: str
    region: Optional[List[float]] = None # bounding box [xmin, ymin, xmax, ymax]
    timestamp: Optional[str] = None
    modality: str
    confidence: Optional[float] = None
    confidence_type: Optional[str] = None # e.g., "softmax", "rule-based", "model-intrinsic"
    status: Literal["VERIFIED", "PARTIALLY_VERIFIED", "UNCERTAIN", "DATA_UNAVAILABLE"]
    source: str
    model: str
    model_version: str

class AnalysisResponse(BaseModel):
    task: str
    answer: str
    status: Literal["VERIFIED", "PARTIALLY_VERIFIED", "UNCERTAIN", "DATA_UNAVAILABLE", "INVALID_INPUT", "ERROR", "EVIDENCE_UNAVAILABLE", "MODEL_UNAVAILABLE", "SUCCESS"]
    confidence: Optional[float] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)
    visual_output: Optional[str] = None # base64 image or similar
    execution_trace: List[str] = Field(default_factory=list)
    provenance: Optional[Provenance] = None
    validation: Optional[List[ValidationResult]] = None
    conflict: bool = False
    abstention_reason: Optional[str] = None
