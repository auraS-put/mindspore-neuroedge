"""Multi-Scale Dilated CNN (A7) — Temporal multi-scale + dilated convolutions.

Based on Paper [12] (Gao et al. 2022):
  "Pediatric Seizure Prediction in Scalp EEG Using a Multi-Scale Neural
   Network With Dilated Convolutions"
  IEEE J. Transl. Eng. Health Med. 10, 1–9.

Results: 93.3% sensitivity, 0.007 FPR/h on CHB-MIT (LOSO).

Architecture (adapted for 4 channels, 8s windows @ 256 Hz = 2048 samples):
  Temporal Multi-Scale Stage:
    3 branches with Conv1d kernels k=32, k=64, k=128 (captures ~125ms, ~250ms, ~500ms)
    Each branch: Conv1d → BN → ReLU → MaxPool(4)
    Then DilatedConvBlock: 3 parallel dilated Conv1d (rates 1,2,5) + attention fusion
  Concat branches → (B, 96, T')
  Classification:
    Conv1d(96→64, k=3) + BN + ReLU + MaxPool(4)
    Conv1d(64→64, k=3) + BN + ReLU
    AdaptiveAvgPool1d(1) → Dropout → Dense(64→2)

Paper [12] spatial multi-scale stage omitted (meaningless for 4 channels).

Input: (B, C=4, T=2048) → Output: (B, num_classes)
"""

from __future__ import annotations

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from auras.models.base import BaseSeizureModel


class DilatedConvBlock(nn.Cell):
    """Parallel dilated convolutions with attention-weighted fusion.

    Paper [12] Section II-B: 3 parallel dilated Conv1d with different dilation
    rates, fused via learned attention weights (GAP → Dense → Softmax).

    Effectively expands the receptive field without adding parameters.
    With k=3 and dilations (1, 2, 5), effective receptive fields are 3, 5, 11.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple = (1, 2, 5)):
        super().__init__()
        self.n_branches = len(dilations)

        self.branches = nn.CellList([
            nn.SequentialCell(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, pad_mode="same"),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            )
            for d in dilations
        ])

        # Attention: concat GAPs → Dense → softmax weights over branches
        self.attn_fc = nn.Dense(channels * self.n_branches, self.n_branches)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        outs = [branch(x) for branch in self.branches]  # list of (B, C, T)

        # Global average pool per branch → attention weights
        gaps = [o.mean(axis=-1) for o in outs]  # list of (B, C)
        gaps_cat = ops.concat(gaps, axis=-1)    # (B, C * n_branches)
        weights = self.softmax(self.attn_fc(gaps_cat))  # (B, n_branches)

        # Weighted sum of branch outputs
        stacked = ops.stack(outs, axis=1)             # (B, n_branches, C, T)
        w = weights.unsqueeze(-1).unsqueeze(-1)       # (B, n_branches, 1, 1)
        return (stacked * w).sum(axis=1)              # (B, C, T)


class TemporalBranch(nn.Cell):
    """Single temporal-scale branch: Conv1d(k) + BN + ReLU + Pool + DilatedConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 4):
        super().__init__()
        self.conv = nn.SequentialCell(
            nn.Conv1d(in_channels, out_channels, kernel_size, pad_mode="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        )
        self.dilated_block = DilatedConvBlock(out_channels, kernel_size=3, dilations=(1, 2, 5))

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)             # (B, out_channels, T//pool_size)
        return self.dilated_block(x)  # (B, out_channels, T//pool_size)


class MultiScaleCNN(BaseSeizureModel):
    """Multi-Scale Dilated CNN for seizure prediction (Paper [12]).

    ~80-100K params with default settings.
    """

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        branch_channels: int = 32,
        temporal_kernels: tuple = (32, 64, 128),
        pool_size: int = 4,
        cls_channels: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        n_branches = len(temporal_kernels)
        concat_dim = branch_channels * n_branches  # 96 by default

        # Temporal multi-scale: 3 branches with different kernel sizes
        self.branches = nn.CellList([
            TemporalBranch(num_channels, branch_channels, k, pool_size)
            for k in temporal_kernels
        ])

        # Classification stage
        self.classifier_conv = nn.SequentialCell(
            nn.Conv1d(concat_dim, cls_channels, 3, pad_mode="same"),
            nn.BatchNorm1d(cls_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Conv1d(cls_channels, cls_channels, 3, pad_mode="same"),
            nn.BatchNorm1d(cls_channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(cls_channels, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        branch_outs = [branch(x) for branch in self.branches]  # list of (B, 32, T')
        x = ops.concat(branch_outs, axis=1)                    # (B, 96, T')
        x = self.classifier_conv(x)                            # (B, 64, T'')
        x = self.pool(x).squeeze(-1)                           # (B, 64)
        return self.head(x)                                    # (B, num_classes)
