"""ResNet adapted for 1-D temporal signals (EEG classification).

Uses 1-D convolutions instead of 2-D, with residual skip connections.
Input: (B, C, T) → Output: (B, num_classes)

Reference 2-D implementation available in mindcv:
    ``mindcv.models.resnet`` (pip install mindcv)
    https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet
"""

from __future__ import annotations

from typing import List

import mindspore.nn as nn
from mindspore import Tensor, ops

from auras.models.base import BaseSeizureModel


class ResidualBlock1D(nn.Cell):
    """Basic residual block with two Conv1d layers."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, pad_mode="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, stride=1, pad_mode="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.SequentialCell(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, pad_mode="valid"),
                nn.BatchNorm1d(out_channels),
            )

    def construct(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(BaseSeizureModel):
    """1-D ResNet for multi-channel time-series classification."""

    def __init__(
        self,
        num_channels: int = 4,
        num_blocks: List[int] = (2, 2, 2, 2),
        base_channels: int = 64,
        dropout: float = 0.2,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        self.stem = nn.SequentialCell(
            nn.Conv1d(num_channels, base_channels, 15, stride=2, pad_mode="same"),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        layers = []
        in_ch = base_channels
        for i, n in enumerate(num_blocks):
            out_ch = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2
            for j in range(n):
                layers.append(ResidualBlock1D(in_ch, out_ch, stride if j == 0 else 1))
                in_ch = out_ch
        self.blocks = nn.SequentialCell(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Dense(in_ch, num_classes)

    def construct(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
