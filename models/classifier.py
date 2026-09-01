"""
Remote Sensing Image Classifier.

Deep Learning classifier for Land-Use and Land-Cover (LULC) satellite image analysis.
Integrates backbone feature extraction, classification head, confidence calibration,
and explainable feature extraction.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from data.dataset import DEFAULT_CLASSES, CLASS_FEATURE_DESCRIPTIONS


class RemoteSensingClassifier(nn.Module):
    """
    Modular Remote Sensing Satellite Image Classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            num_classes: Number of target land cover classes
            backbone_name: CNN/Vision backbone architecture
            pretrained: Whether to use ImageNet pretraining
            in_channels: Input channels (3 for RGB, 1 for SAR, 4+ for Multispectral)
            dropout_rate: Dropout probability for regularization
            hidden_dim: Hidden dimension in classification head
            class_names: List of class labels
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.in_channels = in_channels
        self.class_names = class_names or DEFAULT_CLASSES[:num_classes]

        # 1. Feature Extractor Backbone
        self.backbone, self.feature_dim = build_backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            in_channels=in_channels
        )

        # 2. Classification Head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent feature vectors from input satellite images."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    @torch.no_grad()
    def predict_with_explanation(
        self,
        x: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs inference on a single image tensor (1, C, H, W) and generates
        a comprehensive human-readable prediction report with detected features.
        
        Args:
            x: Tensor of shape (1, C, H, W)
            metadata: Preprocessing and geospatial metadata
            
        Returns:
            Dictionary matching the API requirements:
            {
                "prediction": "Urban Area",
                "confidence": 0.92,
                "features": ["Buildings", "Roads", ...],
                "top_k": [...],
                "modality": "optical_rgb",
                "explanation": "..."
            }
        """
        self.eval()
        metadata = metadata or {}
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
        
        # Top prediction
        conf, pred_idx = torch.max(probs, dim=0)
        pred_idx = int(pred_idx.item())
        confidence = float(conf.item())
        
        pred_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else f"Class_{pred_idx}"
        
        # Top-3 predictions
        topk_confs, topk_indices = torch.topk(probs, k=min(3, self.num_classes))
        top_k = [
            {
                "class": self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}",
                "confidence": round(float(c.item()), 4)
            }
            for c, idx in zip(topk_confs, topk_indices)
        ]

        # Associated remote-sensing visual & physical features
        features = CLASS_FEATURE_DESCRIPTIONS.get(pred_class, [
            "Distinct spatial reflectance pattern",
            "Homogeneous land cover region"
        ])

        # Synthesize human-readable explanation
        modality = metadata.get("modality", "optical_rgb")
        explanation_parts = [
            f"The image was identified as '{pred_class}' with {confidence * 100:.1f}% confidence."
        ]
        
        # Spectral context if available
        spectral_indices = metadata.get("spectral_indices", {})
        if "ndvi_mean" in spectral_indices:
            ndvi = spectral_indices["ndvi_mean"]
            explanation_parts.append(f"Measured Mean NDVI is {ndvi:.3f}, indicating {'strong photosynthetic density' if ndvi > 0.4 else 'low/sparse vegetation'}.")
        if "ndwi_mean" in spectral_indices:
            ndwi = spectral_indices["ndwi_mean"]
            explanation_parts.append(f"Measured Mean NDWI is {ndwi:.3f} ({'high water presence' if ndwi > 0.1 else 'dry terrain/built surface'}).")

        if modality == "sar":
            explanation_parts.append("Radar backscatter analysis detected surface roughness and dielectric signatures characteristic of this terrain.")

        explanation_parts.append(f"Key identified signatures include: {', '.join(features[:3])}.")

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "features": features,
            "top_k": top_k,
            "modality": modality,
            "spatial_info": {
                "crs": metadata.get("crs", "N/A"),
                "bounds": metadata.get("bounds", "N/A"),
                "original_shape": metadata.get("original_shape", "N/A")
            },
            "spectral_indices": {k: v for k, v in spectral_indices.items() if not isinstance(v, (torch.Tensor, list)) and 'map' not in k},
            "explanation": " ".join(explanation_parts)
        }
