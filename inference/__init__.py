"""
Remote Sensing Inference Module.
Provides standalone prediction API and CLI for satellite imagery analysis.
"""

from .predict import predict_image, load_inference_model

__all__ = [
    "predict_image",
    "load_inference_model",
]
