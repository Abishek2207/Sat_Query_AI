"""
Remote Sensing Segmentation Model (U-Net).

Implements a full U-Net architecture for pixel-level semantic segmentation
in satellite and Earth observation imagery (e.g. water bodies, forest canopy,
urban footprints, roads, agricultural fields).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 block with padding=1."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class UpBlock(nn.Module):
    """Upscaling then DoubleConv with skip connection concatenation."""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle potential shape mismatch due to odd dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class RemoteSensingUNet(nn.Module):
    """
    Standard U-Net Architecture for Remote Sensing Pixel Segmentation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_features: int = 32,
        bilinear: bool = True
    ):
        """
        Args:
            in_channels: Input channels (3 for RGB, 1 for SAR, 4 for Multispectral)
            num_classes: Number of segmentation categories (e.g. 2 for Water/Non-Water or N for LULC)
            base_features: Initial channel width (32 or 64)
            bilinear: Whether to use bilinear upsampling (lighter) or Transposed Conv
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        f = base_features  # 32

        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, f)        # f = 32
        self.down1 = DownBlock(f, f * 2)             # 64
        self.down2 = DownBlock(f * 2, f * 4)         # 128
        self.down3 = DownBlock(f * 4, f * 8)         # 256
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(f * 8, (f * 16) // factor) # 256 or 512

        # Decoder (Upsampling with Skip Connections)
        self.up1 = UpBlock(f * 16, (f * 8) // factor, bilinear)
        self.up2 = UpBlock(f * 8, (f * 4) // factor, bilinear)
        self.up3 = UpBlock(f * 4, (f * 2) // factor, bilinear)
        self.up4 = UpBlock(f * 2, f, bilinear)

        # Output 1x1 Convolution
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates discrete class mask (B, H, W) via argmax.
        """
        self.eval()
        logits = self.forward(x)
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            return (probs > 0.5).long().squeeze(1)
        else:
            return torch.argmax(logits, dim=1)
