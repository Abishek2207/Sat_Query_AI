import pytest
import os
import json
from pathlib import Path

from evaluation.core import BenchmarkEvaluator, EvaluationResult

def test_evaluator_missing_data(tmp_path, monkeypatch):
    # Set a dataset path that doesn't exist
    dataset_dir = tmp_path / "missing_dataset"
    monkeypatch.setenv("DUMMY_BENCH_PATH", str(dataset_dir))
    
    # We must patch the evaluator so it writes to tmp_path, else it writes to the real evaluation/ directory
    class DummyEvaluator(BenchmarkEvaluator):
        def __init__(self):
            super().__init__(
                name="DUMMY",
                dataset_env_var="DUMMY_BENCH_PATH",
                expected_format="Format X",
                description="Dummy Desc"
            )
            self.results_file = tmp_path / "results.json"
            
    evaluator = DummyEvaluator()
    res = evaluator.run()
    
    assert res.status == "DATASET_NOT_AVAILABLE"
    assert res.benchmark_name == "DUMMY"
    
    # Check JSON output
    assert evaluator.results_file.exists()
    with open(evaluator.results_file, "r") as f:
        data = json.load(f)
        assert data["status"] == "DATASET_NOT_AVAILABLE"

def test_evaluator_data_present_but_not_evaluated(tmp_path, monkeypatch):
    # Set a dataset path that DOES exist
    dataset_dir = tmp_path / "exists_dataset"
    dataset_dir.mkdir()
    monkeypatch.setenv("DUMMY_BENCH_PATH", str(dataset_dir))
    
    class DummyEvaluator(BenchmarkEvaluator):
        def __init__(self):
            super().__init__(
                name="DUMMY2",
                dataset_env_var="DUMMY_BENCH_PATH",
                expected_format="Format Y",
                description="Dummy Desc"
            )
            self.results_file = tmp_path / "results.json"
            
    evaluator = DummyEvaluator()
    res = evaluator.run()
    
    # Strictly do NOT fabricate scores.
    assert res.status == "NOT_EVALUATED"
