"""CNN-BiLSTM (A2) — convolutional feature extractor + bidirectional LSTM.

Combines a 3-stage CNN stem for local feature extraction with a
stacked bidirectional LSTM for temporal modelling.  Best results
adapted from Paper 09 (Ahmad et al. — CNN-BiLSTM) and Paper 19 (El-Dajani et al. — BTE-CNN-LSTM) (patient-independent settings).

Input: (B, C, T) → Output: (B, num_classes)

Architecture:
  Conv1d(C→32, k=7) + BN + ReLU
  Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)
  Conv1d(64→128, k=3) + BN + ReLU
  BiLSTM(128 → hidden=64, bidirectional) → mean-pool over time
  Dropout → Dense(128→num_classes)
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor

from auras.models.base import BaseSeizureModel


class CNNBiLSTM(BaseSeizureModel):
    """1-D CNN stem followed by bidirectional LSTM classifier."""

    def __init__(
        self,
        num_channels: int = 4,
        num_classes: int = 2,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        conv_channels: tuple = (32, 64, 128),
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        c1, c2, c3 = conv_channels
        self.cnn = nn.SequentialCell(
            nn.Conv1d(num_channels, c1, 7, pad_mode="same"),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, 5, pad_mode="same"),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c2, c3, 3, pad_mode="same"),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        fc_in = hidden_size * 2  # bidirectional
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=dropout),
            nn.Dense(fc_in, num_classes),
        )

    def construct(self, x: Tensor) -> Tensor:
        x = self.cnn(x)                   # (B, 128, T')
        x = x.transpose(0, 2, 1)          # (B, T', 128)
        output, _ = self.lstm(x)           # (B, T', 2*hidden)
        x = output.mean(axis=1)            # (B, 2*hidden) — mean-pool over time
        return self.classifier(x)
