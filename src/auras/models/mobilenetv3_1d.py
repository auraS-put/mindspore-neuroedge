"""MobileNetV3-Small adapted for 1-D EEG classification.

Lightweight inverted-residual blocks with squeeze-and-excitation (SE)
attention, ported from 2-D image classification to 1-D temporal signals.
Input: (B, C, T) → Output: (B, num_classes)

Reference 2-D implementation available in mindcv:
    ``mindcv.models.mobilenetv3`` (pip install mindcv)
    https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor, ops

from auras.models.base import BaseSeizureModel


class _SE1D(nn.Cell):
    """Squeeze-and-Excitation block for 1-D."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(channels, mid),
            nn.ReLU(),
            nn.Dense(mid, channels),
            nn.HSigmoid(),
        )

    def construct(self, x: Tensor) -> Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class _InvertedResidual1D(nn.Cell):
    """MobileNetV3 inverted residual block for 1-D."""

    def __init__(self, inp, hidden, oup, kernel, stride, use_se, activation):
        super().__init__()
        self.use_res = stride == 1 and inp == oup
        act = nn.HSwish() if activation == "HS" else nn.ReLU()

        layers = []
        if inp != hidden:
            layers.extend([nn.Conv1d(inp, hidden, 1, pad_mode="valid"), nn.BatchNorm1d(hidden), act])
        layers.extend([
            nn.Conv1d(hidden, hidden, kernel, stride=stride, pad_mode="same", group=hidden),
            nn.BatchNorm1d(hidden),
            act,
        ])
        if use_se:
            layers.append(_SE1D(hidden))
        layers.extend([nn.Conv1d(hidden, oup, 1, pad_mode="valid"), nn.BatchNorm1d(oup)])
        self.block = nn.SequentialCell(*layers)

    def construct(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return out + x if self.use_res else out


class MobileNetV3_1D(BaseSeizureModel):
    """MobileNetV3-Small for 1-D multi-channel time-series."""

    def __init__(
        self,
        num_channels: int = 4,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        def _c(x):
            return max(int(x * width_mult), 8)

        # MobileNetV3-Small config: [kernel, exp, out, SE, activation, stride]
        cfgs = [
            [3, 16, 16, True, "RE", 2],
            [3, 72, 24, False, "RE", 2],
            [3, 88, 24, False, "RE", 1],
            [5, 96, 40, True, "HS", 2],
            [5, 240, 40, True, "HS", 1],
            [5, 240, 40, True, "HS", 1],
            [5, 120, 48, True, "HS", 1],
            [5, 144, 48, True, "HS", 1],
            [5, 288, 96, True, "HS", 2],
            [5, 576, 96, True, "HS", 1],
            [5, 576, 96, True, "HS", 1],
        ]

        self.stem = nn.SequentialCell(
            nn.Conv1d(num_channels, _c(16), 3, stride=2, pad_mode="same"),
            nn.BatchNorm1d(_c(16)),
            nn.HSwish(),
        )

        blocks = []
        inp = _c(16)
        for k, exp, out, se, act, s in cfgs:
            blocks.append(_InvertedResidual1D(inp, _c(exp), _c(out), k, s, se, act))
            inp = _c(out)
        self.blocks = nn.SequentialCell(*blocks)

        last_ch = _c(576)
        self.head_conv = nn.SequentialCell(
            nn.Conv1d(inp, last_ch, 1, pad_mode="valid"),
            nn.BatchNorm1d(last_ch),
            nn.HSwish(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.SequentialCell(
            nn.Dense(last_ch, 1024),
            nn.HSwish(),
            nn.Dropout(p=dropout),
            nn.Dense(1024, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
