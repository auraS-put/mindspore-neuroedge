"""GhostNet adapted for 1-D EEG classification.

GhostNet (Huawei Noah's Ark Lab, CVPR 2020) generates more feature maps
from cheap linear operations, yielding a compact yet powerful model.
Input: (B, C, T) → Output: (B, num_classes)

Reference 2-D implementation available in mindcv:
    ``mindcv.models.ghostnet`` (pip install mindcv)
    https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet
"""

from __future__ import annotations

import math

import mindspore.nn as nn
from mindspore import Tensor, ops

from auras.models.base import BaseSeizureModel


class _GhostModule1D(nn.Cell):
    """Ghost module: intrinsic features + cheap ghost features."""

    def __init__(self, inp: int, oup: int, kernel: int = 1, ratio: int = 2, stride: int = 1):
        super().__init__()
        init_channels = math.ceil(oup / ratio)
        ghost_channels = init_channels * (ratio - 1)

        self.primary = nn.SequentialCell(
            nn.Conv1d(inp, init_channels, kernel, stride=stride, pad_mode="same"),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(),
        )
        self.cheap = nn.SequentialCell(
            nn.Conv1d(init_channels, ghost_channels, 3, stride=1, pad_mode="same", group=init_channels),
            nn.BatchNorm1d(ghost_channels),
            nn.ReLU(),
        )
        self.oup = oup

    def construct(self, x: Tensor) -> Tensor:
        p = self.primary(x)
        g = self.cheap(p)
        out = ops.concat((p, g), axis=1)
        return out[:, : self.oup, :]


class _GhostBottleneck1D(nn.Cell):
    """Ghost bottleneck with optional SE and stride."""

    def __init__(self, inp, hidden, oup, kernel, stride, use_se=False):
        super().__init__()
        self.use_res = stride == 1 and inp == oup

        self.ghost1 = _GhostModule1D(inp, hidden, kernel=1)
        self.dw = nn.SequentialCell(
            nn.Conv1d(hidden, hidden, kernel, stride=stride, pad_mode="same", group=hidden),
            nn.BatchNorm1d(hidden),
        ) if stride > 1 else nn.SequentialCell()
        self.ghost2 = _GhostModule1D(hidden, oup, kernel=1)

        if not self.use_res:
            self.shortcut = nn.SequentialCell(
                nn.Conv1d(inp, inp, kernel, stride=stride, pad_mode="same", group=inp),
                nn.BatchNorm1d(inp),
                nn.Conv1d(inp, oup, 1, pad_mode="valid"),
                nn.BatchNorm1d(oup),
            )

    def construct(self, x: Tensor) -> Tensor:
        out = self.ghost1(x)
        out = self.dw(out)
        out = self.ghost2(out)
        if self.use_res:
            return out + x
        return out + self.shortcut(x)


class GhostNet1D(BaseSeizureModel):
    """GhostNet for 1-D multi-channel time-series classification."""

    def __init__(
        self,
        num_channels: int = 4,
        width_mult: float = 1.0,
        ghost_ratio: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        def _c(x):
            return max(int(x * width_mult), 8)

        # Simplified GhostNet config: [kernel, exp, out, stride]
        cfgs = [
            [3, 16, 16, 1],
            [3, 48, 24, 2],
            [3, 72, 24, 1],
            [5, 72, 40, 2],
            [5, 120, 40, 1],
            [3, 240, 80, 2],
            [3, 200, 80, 1],
            [5, 184, 112, 1],
            [5, 480, 160, 2],
            [5, 672, 160, 1],
        ]

        self.stem = nn.SequentialCell(
            nn.Conv1d(num_channels, _c(16), 3, stride=2, pad_mode="same"),
            nn.BatchNorm1d(_c(16)),
            nn.ReLU(),
        )

        blocks = []
        inp = _c(16)
        for k, exp, out, s in cfgs:
            blocks.append(_GhostBottleneck1D(inp, _c(exp), _c(out), k, s))
            inp = _c(out)
        self.blocks = nn.SequentialCell(*blocks)

        last_ch = _c(960)
        self.head = nn.SequentialCell(
            nn.Conv1d(inp, last_ch, 1, pad_mode="valid"),
            nn.BatchNorm1d(last_ch),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.SequentialCell(
            nn.Dense(last_ch, 1280),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(1280, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
