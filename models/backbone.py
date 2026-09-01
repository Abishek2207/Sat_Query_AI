"""
Modular Feature Extractor Backbones for Remote Sensing Imagery.

Supports:
- ResNet-18 / ResNet-50 (Transfer Learning)
- MobileNetV3 (Lightweight for CPU / edge deployment)
- EfficientNet-B0 (High-efficiency feature extraction)
- Custom Lightweight RemoteSensingCNN (Zero-dependency fast baseline)
- Multi-channel Adapter (Adapts 1-ch SAR or 4+ ch Multispectral to standard backbones)
"""

from typing import Tuple
import torch
import torch.nn as nn

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def auto_select_device() -> torch.device:
    """
    Automatically selects the best available compute hardware:
    1. NVIDIA CUDA GPU
    2. Apple Silicon MPS
    3. CPU fallback
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[Hardware] Using NVIDIA GPU: {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Hardware] Using Apple Silicon MPS GPU")
    else:
        device = torch.device("cpu")
        print("[Hardware] Using CPU (No GPU detected or CUDA unavailable)")
    return device


class ConvBlock(nn.Module):
    """Standard Convolution-BatchNorm-ReLU block with optional Residual shortcut."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class RemoteSensingCustomCNN(nn.Module):
    """
    Lightweight, high-performance custom CNN backbone designed specifically
    for Earth Observation and Remote Sensing tasks without requiring heavy downloads.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = ConvBlock(32, 64, stride=2)   # H/2, W/2
        self.layer2 = ConvBlock(64, 128, stride=2)  # H/4, W/4
        self.layer3 = ConvBlock(128, 256, stride=2) # H/8, W/8
        self.layer4 = ConvBlock(256, feature_dim, stride=2) # H/16, W/16
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def adapt_first_conv_layer(conv_layer: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Adapts a pretrained 3-channel Conv2d layer to accept arbitrary channel counts (e.g. 1-ch SAR or 4-ch Multispectral)
    while preserving pretrained ImageNet filter weights.
    """
    if conv_layer.in_channels == in_channels:
        return conv_layer

    old_weights = conv_layer.weight.data  # shape: (out_c, 3, k_h, k_w)
    out_c, _, k_h, k_w = old_weights.shape

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_c,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=(conv_layer.bias is not None)
    )

    with torch.no_grad():
        if in_channels == 1:
            # Average the 3 RGB channel weights for 1-channel (SAR / Grayscale)
            new_conv.weight.data = old_weights.mean(dim=1, keepdim=True)
        elif in_channels > 3:
            # Copy first 3 channels and initialize remaining channels with average
            new_conv.weight.data[:, :3, :, :] = old_weights
            avg_weight = old_weights.mean(dim=1, keepdim=True)
            for extra_c in range(3, in_channels):
                new_conv.weight.data[:, extra_c:extra_c+1, :, :] = avg_weight
        else:
            new_conv.weight.data[:, :in_channels, :, :] = old_weights[:, :in_channels, :, :]

        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data

    return new_conv


def build_backbone(
    model_name: str = "resnet18",
    pretrained: bool = True,
    in_channels: int = 3
) -> Tuple[nn.Module, int]:
    """
    Builds and returns a feature extraction backbone.
    
    Args:
        model_name: 'resnet18', 'resnet50', 'mobilenet_v3_small', 'efficientnet_b0', or 'custom_cnn'
        pretrained: Whether to load ImageNet pretrained weights
        in_channels: Input channel count (1 for SAR, 3 for RGB, 4+ for Multispectral)
        
    Returns:
        Tuple of (backbone_module, feature_dimension)
    """
    model_name = model_name.lower()

    if not HAS_TORCHVISION or model_name == "custom_cnn":
        backbone = RemoteSensingCustomCNN(in_channels=in_channels, feature_dim=256)
        return backbone, 256

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        if in_channels != 3:
            base.conv1 = adapt_first_conv_layer(base.conv1, in_channels)
        feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        return base, feature_dim

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        if in_channels != 3:
            base.conv1 = adapt_first_conv_layer(base.conv1, in_channels)
        feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        return base, feature_dim

    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        if in_channels != 3:
            base.features[0][0] = adapt_first_conv_layer(base.features[0][0], in_channels)
        feature_dim = base.classifier[0].in_features
        base.classifier = nn.Identity()
        return base, feature_dim

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        if in_channels != 3:
            base.features[0][0] = adapt_first_conv_layer(base.features[0][0], in_channels)
        feature_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
        return base, feature_dim

    else:
        # Fallback to custom CNN
        backbone = RemoteSensingCustomCNN(in_channels=in_channels, feature_dim=256)
        return backbone, 256


def get_model_summary(model: nn.Module) -> str:
    """Returns a readable summary of total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"Total Parameters: {total_params:,} | "
        f"Trainable Parameters: {trainable_params:,} "
        f"({trainable_params / (total_params + 1e-9) * 100:.1f}%)"
    )
