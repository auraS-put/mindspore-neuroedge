"""CAM-CNN-BiLSTM (A4) — our novel channel-attention hybrid (Sprint 2 proposal).

Introduces an explicit Channel Attention Module (CAM) to dynamically
weight the four EEG channels before feature extraction, then uses a
residual DS-Conv stem + BiLSTM head.

Inspired by:
  Paper 20 (Wang Y. et al. — CAM-CNN): GAP→Conv1d(k=3)→Sigmoid channel selection
  Paper 19 (El-Dajani et al. — BTE-CNN-LSTM): BiLSTM(40), patient-independent, Adam LR=1e-3
  Paper 18 (Mehrabi et al. — ConvSNN): ResDSBlock + global average pool, AdamW + OneCycleLR

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  ChannelAttention1D(C)                        ← reweight channels
  Conv1d(C→64, k=7, pad=same) + BN + GELU
  2× ResDSBlock(64, k=7, dropout)
  BiLSTM(64 → hidden=40, bidirectional)        ← 80-dim output
  Mean-pool over time → Dropout → Dense(80→num_classes)
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor

from auras.models.base import BaseSeizureModel
from auras.models.modules import ChannelAttention1D, ResDSBlock


class CAMCNNBiLSTM(BaseSeizureModel):
    """Channel-Attention + CNN + BiLSTM — paper-novel architecture."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        stem_channels: int = 64,
        lstm_hidden: int = 40,
        dropout: float = 0.25,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        self.cam = ChannelAttention1D(num_channels)
        self.stem = nn.SequentialCell(
            nn.Conv1d(num_channels, stem_channels, 7, pad_mode="same"),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(approximate=False),
        )
        self.res_blocks = nn.SequentialCell(
            ResDSBlock(stem_channels, kernel_size=7, dropout=dropout),
            ResDSBlock(stem_channels, kernel_size=7, dropout=dropout),
        )
        self.lstm = nn.LSTM(
            input_size=stem_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fc_in = lstm_hidden * 2  # 80
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(fc_in, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.cam(x)                      # (B, C, T) — reweighted channels
        x = self.stem(x)                     # (B, 64, T)
        x = self.res_blocks(x)               # (B, 64, T)
        x = x.transpose(0, 2, 1)            # (B, T, 64)
        output, _ = self.lstm(x)             # (B, T, 80)
        x = output.mean(axis=1)              # (B, 80)
        return self.classifier(x)
