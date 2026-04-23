"""Ultra-Lightweight DS-CNN (A7) — sub-10 K parameter EEG classifier.

Targets edge deployment on wearable hardware.  Combines channel
attention, dilated depthwise-separable convolutions, and aggressive
pooling to stay under 10 K parameters.

Inspired by:
  Paper 08 (Xie et al. — TSS3D): 6,540 params after pruning, SGD + warmup + step-LR
  Paper 20 (Wang Y. et al. — CAM-CNN): ~255 params, dilated conv, GAP → Dense

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  ChannelAttention1D(C)
  Conv1d(C→16, k=5) + BN + ReLU + MaxPool(4)
  Conv1d(16→32, k=3, dilation=2, pad=same) + BN + ReLU + MaxPool(4)
  AdaptiveAvgPool1d(1) → Dropout → Dense(32→num_classes)
  ≈ 3–5 K parameters
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor

from auras.models.base import BaseSeizureModel
from auras.models.modules import ChannelAttention1D


class UltraLightCNN(BaseSeizureModel):
    """Sub-10 K ultra-lightweight CNN for edge-deployable seizure prediction."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        self.cam = ChannelAttention1D(num_channels)

        self.features = nn.SequentialCell(
            # Block 1: basic Conv + BN + ReLU + MaxPool(4)
            nn.Conv1d(num_channels, 16, 5, pad_mode="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # Block 2: dilated Conv (receptive field ×2) + BN + ReLU + MaxPool(4)
            nn.Conv1d(16, 32, 3, dilation=2, pad_mode="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(32, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.cam(x)                    # (B, C, T) — channel-wise rescaling
        x = self.features(x)               # (B, 32, T'')
        x = self.pool(x).squeeze(-1)       # (B, 32)
        return self.classifier(x)
