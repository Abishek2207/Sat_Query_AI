from pydantic import BaseModel, Field
from typing import List, Optional

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

class AnalysisResponse(BaseModel):
    task: str
    answer: str
    status: str
    confidence: Optional[float] = None
    evidence: List[str] = Field(default_factory=list)
    visual_output: Optional[str] = None
    execution_trace: List[str] = Field(default_factory=list)
    provenance: Optional[Provenance] = None
    validation: Optional[List[ValidationResult]] = None
