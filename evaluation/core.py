import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Optional, Any

class EvaluationResult(BaseModel):
    benchmark_name: str
    status: str
    metrics: Optional[Dict[str, float]] = None
    expected_format: str
    dataset_path: str
    description: str

class BenchmarkEvaluator:
    def __init__(self, name: str, dataset_env_var: str, expected_format: str, description: str):
        self.name = name
        self.dataset_path = os.getenv(dataset_env_var, f"datasets/{name.lower()}")
        self.expected_format = expected_format
        self.description = description
        self.results_file = Path(f"evaluation/{name.lower()}/results.json")
        
    def is_data_available(self) -> bool:
        return Path(self.dataset_path).exists()
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Override this method with actual metric calculation when data is present."""
        raise NotImplementedError("Metrics calculation requires actual data and predictions.")
        
    def run(self) -> EvaluationResult:
        if not self.is_data_available():
            status = "DATASET_NOT_AVAILABLE"
            metrics = None
        else:
            # When data is present, normally we would run inference and calculate_metrics().
            # For strict compliance: do not run fake training/evaluation.
            status = "NOT_EVALUATED"
            metrics = None
            
        result = EvaluationResult(
            benchmark_name=self.name,
            status=status,
            metrics=metrics,
            expected_format=self.expected_format,
            dataset_path=self.dataset_path,
            description=self.description
        )
        
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
            
        print(f"[{self.name}] Status: {status}")
        return result
