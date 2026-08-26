"""
RSVQA Evaluator
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluation.core import BenchmarkEvaluator

class RSVQAEvaluator(BenchmarkEvaluator):
    def __init__(self):
        super().__init__(
            name="RSVQA",
            dataset_env_var="RSVQA_PATH",
            expected_format="GeoTIFF tiles with JSON Q/A pairs (Low/High resolution splits).",
            description="Baseline Remote-Sensing Visual Question Answering."
        )

if __name__ == "__main__":
    evaluator = RSVQAEvaluator()
    evaluator.run()
