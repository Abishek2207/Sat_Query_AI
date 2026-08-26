"""
VRSBench Evaluator
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluation.core import BenchmarkEvaluator

class VRSBenchEvaluator(BenchmarkEvaluator):
    def __init__(self):
        super().__init__(
            name="VRSBench",
            dataset_env_var="VRSBENCH_PATH",
            expected_format="JSON annotations mapping to GeoTIFF/PNG satellite images.",
            description="Visual Question Answering on diverse remote sensing scenes."
        )

if __name__ == "__main__":
    evaluator = VRSBenchEvaluator()
    evaluator.run()
