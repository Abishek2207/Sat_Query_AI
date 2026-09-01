"""
Remote Sensing Inference and API Interface.

Provides:
- Standalone function `predict_image(image_path)` ready for API integration
- Full command-line inference tool with multi-panel visualization
- Automatic modality detection (Optical RGB, Multi-band GeoTIFF, SAR)
- Spectral indices (NDVI/NDWI) and explainable feature extraction
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, Union, Tuple

import torch
import numpy as np

from preprocessing.preprocess import RemoteSensingPreprocessor, ModalityType
from models.classifier import RemoteSensingClassifier
from models.backbone import auto_select_device
from data.dataset import DEFAULT_CLASSES
from utils.visualization import plot_prediction

# Global cache for inference model to enable fast repeated API calls
_GLOBAL_MODEL_CACHE: Dict[str, Tuple[RemoteSensingClassifier, RemoteSensingPreprocessor, torch.device]] = {}


def load_inference_model(
    checkpoint_path: Optional[str] = "checkpoints/best_model.pth",
    device: Optional[torch.device] = None,
    backbone: str = "resnet18",
    num_classes: int = 10,
    in_channels: int = 3
) -> Tuple[RemoteSensingClassifier, RemoteSensingPreprocessor, torch.device]:
    """
    Loads and caches model and preprocessor for low-latency inference.
    """
    global _GLOBAL_MODEL_CACHE

    if device is None:
        device = auto_select_device()

    cache_key = f"{checkpoint_path}_{device}"
    if cache_key in _GLOBAL_MODEL_CACHE:
        return _GLOBAL_MODEL_CACHE[cache_key]

    class_names = DEFAULT_CLASSES[:num_classes]
    image_size = (224, 224)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        class_names = checkpoint.get("class_names", class_names)
        backbone = checkpoint.get("backbone", backbone)
        in_channels = checkpoint.get("in_channels", in_channels)
        num_classes = checkpoint.get("num_classes", len(class_names))
        img_sz = checkpoint.get("config", {}).get("data", {}).get("image_size", [224, 224])
        image_size = tuple(img_sz)

        model = RemoteSensingClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=False,
            in_channels=in_channels,
            class_names=class_names
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Pretrained baseline fallback if no checkpoint is passed
        model = RemoteSensingClassifier(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=True,
            in_channels=in_channels,
            class_names=class_names
        ).to(device)

    model.eval()

    preprocessor = RemoteSensingPreprocessor(
        image_size=image_size,
        target_channels=in_channels,
        normalize_method="imagenet"
    )

    _GLOBAL_MODEL_CACHE[cache_key] = (model, preprocessor, device)
    return model, preprocessor, device


def predict_image(
    image_path: str,
    checkpoint_path: Optional[str] = "checkpoints/best_model.pth",
    model: Optional[RemoteSensingClassifier] = None,
    preprocessor: Optional[RemoteSensingPreprocessor] = None,
    device: Optional[torch.device] = None,
    visualize: bool = False,
    save_plot_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main Production API function for Remote Sensing Image Inference.

    Input:
        image_path (str): Path to satellite image (.jpg, .png, .tif, GeoTIFF, SAR)
        checkpoint_path (str, optional): Path to trained model checkpoint
        model (optional): Pre-loaded model instance
        preprocessor (optional): Pre-loaded preprocessor instance
        device (optional): Target torch.device
        visualize (bool): Whether to generate an inspection plot
        save_plot_path (str, optional): Destination for visualization image

    Returns:
        Structured dictionary:
        {
            "prediction": "Residential",
            "confidence": 0.9421,
            "features": [
                "Urban and suburban housing",
                "High density of buildings and rooftops",
                "Interspersed street networks"
            ],
            "top_k": [...],
            "modality": "optical_rgb",
            "spatial_info": {...},
            "spectral_indices": {...},
            "explanation": "Human-readable explanation..."
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Satellite image not found: {image_path}")

    # Load model and preprocessor if not provided
    if model is None or preprocessor is None or device is None:
        model, preprocessor, device = load_inference_model(
            checkpoint_path=checkpoint_path,
            device=device
        )

    # 1. Preprocess satellite image
    processed_arr, metadata = preprocessor(image_path)
    tensor = torch.from_numpy(processed_arr).unsqueeze(0).to(device)  # (1, C, H, W)

    # 2. Run inference with explanation
    result = model.predict_with_explanation(tensor, metadata=metadata)
    result["image_path"] = os.path.abspath(image_path)

    # 3. Optional Visualization
    if visualize or save_plot_path:
        plot_out = save_plot_path or os.path.splitext(image_path)[0] + "_prediction_analysis.png"
        plot_prediction(image_path, result, save_path=plot_out)
        result["visualization_path"] = os.path.abspath(plot_out)

    return result


def print_prediction_cli(result: Dict[str, Any]) -> None:
    """Prints a clear and formatted CLI inference summary."""
    print("\n" + "=" * 60)
    print("        REMOTE SENSING AI: SATELLITE IMAGE ANALYSIS")
    print("=" * 60)
    print(f"  Input Image : {result.get('image_path', 'N/A')}")
    print(f"  Modality    : {result.get('modality', 'N/A').upper()}")
    print("-" * 60)
    print(f"  PREDICTION  : {result.get('prediction', 'Unknown')}")
    print(f"  CONFIDENCE  : {result.get('confidence', 0.0) * 100:.2f}%")
    print("-" * 60)
    print("  DETECTED REMOTE SENSING FEATURES:")
    for feat in result.get("features", []):
        print(f"    - {feat}")
        
    indices = result.get("spectral_indices", {})
    if indices:
        print("\n  SPECTRAL INDICES:")
        for k, v in indices.items():
            print(f"    - {k.upper()}: {v:.3f}")
            
    print("\n  TOP PREDICTIONS:")
    for item in result.get("top_k", []):
        print(f"    - {item['class']:<22} : {item['confidence']*100:>6.2f}%")
        
    print("\n  EXPLANATION:")
    print(f"    {result.get('explanation', 'N/A')}")
    
    if "visualization_path" in result:
        print(f"\n  ANALYSIS PLOT SAVED TO:")
        print(f"    {result['visualization_path']}")
        
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on Satellite Images")
    parser.add_argument("--image", type=str, required=True, help="Path to satellite image (.tif, .png, .jpg)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Trained checkpoint path")
    parser.add_argument("--visualize", action="store_true", help="Generate multi-panel analysis plot")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save analysis plot")
    args = parser.parse_args()

    res = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        visualize=args.visualize,
        save_plot_path=args.save_plot
    )
    print_prediction_cli(res)
