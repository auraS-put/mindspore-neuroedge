"""Bidirectional LSTM for EEG seizure classification.

A thin wrapper that re-uses the LSTM model with ``bidirectional=True``.
"""

from __future__ import annotations

from auras.models.lstm import LSTMModel


class BiLSTMModel(LSTMModel):
    """Bidirectional LSTM — uses both forward and backward context."""

    def __init__(self, **kwargs):
        kwargs.setdefault("bidirectional", True)
        super().__init__(**kwargs)
