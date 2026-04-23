"""CNN-BiLSTM-Attention (A3) — DS-Conv residual stem + BiLSTM + MHSA.

Integrates depthwise-separable residual convolutions (Paper 18 (Mehrabi et al. — ConvSNN) pattern),
bidirectional LSTM for temporal context, and multi-head self-attention
over the LSTM output sequence.

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  Conv1d(C→64, k=7) + BN + GELU
  2× ResDSBlock(64, k=9, dropout)
  BiLSTM(64 → hidden=64, bidirectional) → (B, T', 128)
  _TransformerBlock1D(128, 4 heads) — pre-norm MHSA
  Mean-pool over sequence → Dense(128→num_classes)

Paper references:
  Paper 18 (Mehrabi et al. — ConvSNN): ResDS-Conv + temporal MHSA + AdamW + OneCycleLR
  Paper 19 (El-Dajani et al. — BTE-CNN-LSTM): BiLSTM(40) + AvgPool, Adam LR=1e-3 batch=32
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor

from auras.models.base import BaseSeizureModel
from auras.models.mobilevit_1d import _TransformerBlock1D
from auras.models.modules import ResDSBlock


class CNNBiLSTMAttn(BaseSeizureModel):
    """DS-Conv + BiLSTM + Multi-head Self-Attention classifier."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        stem_channels: int = 64,
        lstm_hidden: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        attn_dim = lstm_hidden * 2  # bidirectional LSTM output size

        self.stem = nn.SequentialCell(
            nn.Conv1d(num_channels, stem_channels, 7, pad_mode="same"),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(approximate=False),
        )
        self.res_blocks = nn.SequentialCell(
            ResDSBlock(stem_channels, kernel_size=9, dropout=dropout),
            ResDSBlock(stem_channels, kernel_size=9, dropout=dropout),
        )
        self.lstm = nn.LSTM(
            input_size=stem_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.transformer = _TransformerBlock1D(
            dim=attn_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(attn_dim, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.stem(x)                        # (B, 64, T)
        x = self.res_blocks(x)                  # (B, 64, T)
        x = x.transpose(0, 2, 1)               # (B, T, 64)
        output, _ = self.lstm(x)                # (B, T, 128)
        # _TransformerBlock1D expects (T, B, dim)
        x = output.transpose(1, 0, 2)           # (T, B, 128)
        x = self.transformer(x)                 # (T, B, 128)
        x = x.mean(axis=0)                      # (B, 128) — mean-pool over time
        return self.classifier(x)
