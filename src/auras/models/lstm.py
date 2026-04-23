"""LSTM model for EEG seizure classification.

Uses the built-in ``mindspore.nn.LSTM`` (supports Ascend/GPU/CPU).
Treats the temporal dimension as a sequence and channels as features.
Input: (B, C, T) → Output: (B, num_classes)

API reference: https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LSTM.html
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor, ops

from auras.models.base import BaseSeizureModel


class LSTMModel(BaseSeizureModel):
    """Vanilla LSTM classifier for 1-D multi-channel time-series."""

    def __init__(
        self,
        num_channels: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Dense(fc_in, num_classes)

    def construct(self, x: Tensor) -> Tensor:
        # x: (B, C, T) → (B, T, C) for LSTM
        x = x.transpose(0, 2, 1)
        output, _ = self.lstm(x)
        # Take last time-step
        last = output[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)
