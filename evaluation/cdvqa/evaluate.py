"""
CDVQA Evaluator
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluation.core import BenchmarkEvaluator

class CDVQAEvaluator(BenchmarkEvaluator):
    def __init__(self):
        super().__init__(
            name="CDVQA",
            dataset_env_var="CDVQA_PATH",
            expected_format="Bi-temporal image pairs (T1, T2) with change-detection Q/A pairs.",
            description="Change Detection Visual Question Answering."
        )

if __name__ == "__main__":
    evaluator = CDVQAEvaluator()
    evaluator.run()
