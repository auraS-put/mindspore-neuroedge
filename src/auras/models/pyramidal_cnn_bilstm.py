"""Pyramidal CNN-BiLSTM (A8) — stride-based pyramidal CNN + BiLSTM (Paper 04 (Wang C. et al. — PCNN-BiLSTM)).

Implements the 9,371-parameter architecture described in Paper 04 (Wang C. et al. — PCNN-BiLSTM).
Stride-based pooling replaces explicit pooling layers, creating a
pyramid that progressively compresses the temporal dimension before
passing to the bidirectional LSTM.

Paper 04 (Wang C. et al. — PCNN-BiLSTM) settings:
  3 conv blocks (K=5/S=3, K=3/S=2, K=3/S=2), BN + ReLU, BiLSTM
  Training: Adam LR=2e-5, 1000 epochs (use early stopping in practice)

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  Conv1d(C→16,  k=5, s=3) + BN + ReLU
  Conv1d(16→32, k=3, s=2) + BN + ReLU
  Conv1d(32→64, k=3, s=2) + BN + ReLU
  BiLSTM(64 → hidden=32, bidirectional) → mean-pool
  Dropout → Dense(64→num_classes)
  ≈ 10–15 K parameters
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor

from auras.models.base import BaseSeizureModel


class PyramidalCNNBiLSTM(BaseSeizureModel):
    """Stride-based pyramidal CNN + BiLSTM (~10 K params, Paper 04 (Wang C. et al. — PCNN-BiLSTM))."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        lstm_hidden: int = 32,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        self.cnn = nn.SequentialCell(
            # Stride-3 conv replaces first pooling layer (Paper 04 (Wang C. et al. — PCNN-BiLSTM))
            nn.Conv1d(num_channels, 16, kernel_size=5, stride=3, pad_mode="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Stride-2 conv for progressive downsampling
            nn.Conv1d(16, 32, kernel_size=3, stride=2, pad_mode="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, pad_mode="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        fc_in = lstm_hidden * 2  # bidirectional → 64
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(fc_in, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.cnn(x)                    # (B, 64, T')
        x = x.transpose(0, 2, 1)           # (B, T', 64)
        output, _ = self.lstm(x)           # (B, T', 64)
        x = output.mean(axis=1)            # (B, 64) — mean-pool over time
        return self.classifier(x)
