"""
Remote Sensing Evaluation Module.
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, IoU, and Dice score.
"""

from .metrics import (
    calculate_classification_metrics,
    calculate_segmentation_metrics,
    print_metrics_summary,
)
from .evaluate import evaluate_model_checkpoint

__all__ = [
    "calculate_classification_metrics",
    "calculate_segmentation_metrics",
    "print_metrics_summary",
    "evaluate_model_checkpoint",
]
