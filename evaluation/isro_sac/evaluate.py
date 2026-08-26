"""
ISRO/SAC Evaluator
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluation.core import BenchmarkEvaluator

class ISROSACEvaluator(BenchmarkEvaluator):
    def __init__(self):
        super().__init__(
            name="ISRO_SAC",
            dataset_env_var="ISRO_SAC_PATH",
            expected_format="ISRO optical/SAR payload data with domain-specific expert QA annotations.",
            description="Custom Indian Space Research Organisation (SAC) Evaluation Dataset."
        )
        
    def run(self):
        # The ISRO dataset annotations are deliberately hidden by organizers
        status = "ORGANIZER_EVALUATION_PENDING"
        
        result = self._save_result(status, None)
        print(f"[{self.name}] Status: {status}")
        return result

    def _save_result(self, status, metrics):
        from evaluation.core import EvaluationResult
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
        return result

if __name__ == "__main__":
    evaluator = ISROSACEvaluator()
    evaluator.run()
