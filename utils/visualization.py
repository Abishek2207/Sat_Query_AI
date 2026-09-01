"""
Remote Sensing Visualization Suite.

Provides:
- Multi-panel prediction inspection (Original image, Confidence bars, Detected features, Spectral indices)
- Confusion Matrix Heatmap
- NDVI / NDWI False-color and Index heatmaps
- Semantic Segmentation Mask Overlays
"""

import os
from typing import Dict, Any, List, Optional, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all servers/environments
import matplotlib.pyplot as plt

from preprocessing.preprocess import load_remote_sensing_image, compute_spectral_indices, ModalityType


def plot_prediction(
    image_path_or_array: Union[str, np.ndarray],
    prediction_result: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Plots a multi-panel visual prediction report:
    1. Satellite Image (Natural RGB, False-Color, or SAR backscatter)
    2. Confidence Bar Chart for Top-K predictions
    3. Spectral Index / SAR Analysis (if available)
    4. Detected Geospatial Features summary
    
    Args:
        image_path_or_array: Path to image or (C, H, W) numpy array
        prediction_result: Output dictionary from predict_image()
        save_path: Optional path to save figure PNG
    """
    if isinstance(image_path_or_array, str):
        data, modality, meta = load_remote_sensing_image(image_path_or_array)
    else:
        data = image_path_or_array
        modality = ModalityType.OPTICAL_RGB

    c, h, w = data.shape
    pred_class = prediction_result.get("prediction", "Unknown")
    confidence = prediction_result.get("confidence", 0.0)
    top_k = prediction_result.get("top_k", [])
    features = prediction_result.get("features", [])

    fig = plt.figure(figsize=(14, 5))

    # 1. Satellite Image View
    ax1 = fig.add_subplot(1, 3, 1)
    if c >= 3:
        # Normalize first 3 bands for display
        rgb_disp = data[:3].copy()
        for i in range(3):
            p2, p98 = np.percentile(rgb_disp[i], (2, 98))
            if p98 > p2:
                rgb_disp[i] = np.clip((rgb_disp[i] - p2) / (p98 - p2), 0.0, 1.0)
            else:
                rgb_disp[i] = 0.0
        rgb_disp = np.transpose(rgb_disp, (1, 2, 0))
        ax1.imshow(rgb_disp)
        ax1.set_title(f"Satellite Input ({modality.value.upper()})", fontsize=11, fontweight="bold")
    else:
        # Single band / SAR
        ch = data[0]
        p2, p98 = np.percentile(ch, (2, 98))
        norm_ch = np.clip((ch - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        ax1.imshow(norm_ch, cmap="gray")
        ax1.set_title(f"SAR / Single Band Input", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # 2. Confidence Distribution Bar Chart
    ax2 = fig.add_subplot(1, 3, 2)
    if top_k:
        classes = [item["class"] for item in top_k][::-1]
        confs = [item["confidence"] * 100 for item in top_k][::-1]
        colors = ["#4C72B0", "#55A868", "#C44E52"][:len(classes)]
        bars = ax2.barh(classes, confs, color=colors, height=0.55)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Confidence (%)", fontsize=10, fontweight="bold")
        ax2.set_title("Classification Confidence", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                     ha="left", va="center", fontsize=9, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, f"Prediction: {pred_class}\nConfidence: {confidence*100:.1f}%",
                 ha="center", va="center", fontsize=12)
        ax2.axis("off")

    # 3. Detected Features & Explanation Box
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")

    feature_text = f"PREDICTION: {pred_class.upper()}\n"
    feature_text += f"Confidence: {confidence * 100:.1f}%\n"
    feature_text += "-" * 32 + "\n"
    feature_text += "DETECTED REMOTE SENSING FEATURES:\n"
    for feat in features:
        feature_text += f"  * {feat}\n"

    # Spectral indices if available
    indices = prediction_result.get("spectral_indices", {})
    if indices:
        feature_text += "-" * 32 + "\n"
        feature_text += "SPECTRAL INDICES:\n"
        if "ndvi_mean" in indices:
            feature_text += f"  * Mean NDVI: {indices['ndvi_mean']:.3f}\n"
        if "ndwi_mean" in indices:
            feature_text += f"  * Mean NDWI: {indices['ndwi_mean']:.3f}\n"

    ax3.text(
        0.05, 0.95, feature_text,
        transform=ax3.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F0F4F8", edgecolor="#B0C4DE", alpha=0.9)
    )

    plt.suptitle(f"Remote Sensing AI Analysis: {pred_class} ({confidence*100:.1f}%)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.close()


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plots a normalized confusion matrix heatmap.
    """
    cm_arr = np.array(cm, dtype=np.float32)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_arr / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Remote Sensing Classification Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            raw_val = int(cm_arr[i, j])
            norm_val = cm_norm[i, j]
            text = f"{raw_val}\n({norm_val:.0%})" if raw_val > 0 else "0"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if norm_val > thresh else "black",
                    fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.close()


def plot_segmentation_result(
    image_path_or_array: Union[str, np.ndarray],
    pred_mask: np.ndarray,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Plots original satellite image alongside the predicted semantic segmentation mask.
    """
    if isinstance(image_path_or_array, str):
        data, _, _ = load_remote_sensing_image(image_path_or_array)
    else:
        data = image_path_or_array

    c = data.shape[0]
    if c >= 3:
        rgb = np.transpose(data[:3], (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    else:
        rgb = np.repeat(data[0, :, :, np.newaxis], 3, axis=2)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original
    axes[0].imshow(rgb)
    axes[0].set_title("Input Satellite Image", fontweight="bold")
    axes[0].axis("off")

    # 2. Predicted Mask
    axes[1].imshow(pred_mask, cmap="tab10")
    axes[1].set_title("Predicted Segmentation Mask", fontweight="bold")
    axes[1].axis("off")

    # 3. Blended Overlay
    axes[2].imshow(rgb)
    axes[2].imshow(pred_mask, cmap="tab10", alpha=0.45)
    axes[2].set_title("Mask Overlay", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.close()
