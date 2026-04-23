"""CNN Baseline (A1) — simple 3-layer 1-D CNN.

Serves as the ablation baseline for comparing against hybrid and
attention-augmented architectures.  Adapted from Paper 14 (Manzouri et al. — EE-Implantable) and Paper 20 (Wang Y. et al. — CAM-CNN).

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  Conv1d(C→32, k=7) + BN + ReLU + MaxPool(2)
  Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)
  Conv1d(64→128, k=3) + BN + ReLU + MaxPool(2)
  AdaptiveAvgPool1d(1) → Dropout → Dense(128→num_classes)
"""

from __future__ import annotations

from typing import Tuple

import mindspore.nn as nn
from mindspore import Tensor

from auras.models.base import BaseSeizureModel


class CNNBaseline(BaseSeizureModel):
    """3-layer 1-D CNN with batch-norm and global average pooling."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        kernels: Tuple[int, int, int] = (7, 5, 3),
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        layers = []
        in_ch = num_channels
        for out_ch, k in zip(conv_channels, kernels):
            layers += [
                nn.Conv1d(in_ch, out_ch, k, pad_mode="same"),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            in_ch = out_ch

        self.features = nn.SequentialCell(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(in_ch, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.features(x)           # (B, 128, T')
        x = self.pool(x).squeeze(-1)   # (B, 128)
        return self.classifier(x)
