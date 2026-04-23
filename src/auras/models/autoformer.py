"""Autoformer for 1-D EEG seizure classification.

Based on "Autoformer: Decomposition Transformers with Auto-Correlation"
(NeurIPS 2021). Adapted from time-series forecasting to classification.
Uses series decomposition to separate trend and seasonal components.
Input: (B, C, T) → Output: (B, num_classes)
"""

from __future__ import annotations

import mindspore.nn as nn
from mindspore import Tensor, ops

from auras.models.base import BaseSeizureModel


class _MovingAvgDecomp(nn.Cell):
    """Series decomposition: trend (moving average) + seasonal (residual)."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, pad_mode="same")

    def construct(self, x: Tensor) -> tuple:
        # x: (B, T, D)
        # Pool along T dimension — need (B, D, T) for AvgPool1d
        xt = x.transpose(0, 2, 1)
        trend = self.avg(xt).transpose(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class _AutoCorrelationAttention(nn.Cell):
    """Auto-correlation based attention (simplified).

    Computes attention in the frequency domain via FFT, capturing
    period-based dependencies instead of point-wise attention.
    """

    def __init__(self, d_model: int, n_heads: int, factor: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.q_proj = nn.Dense(d_model, d_model)
        self.k_proj = nn.Dense(d_model, d_model)
        self.v_proj = nn.Dense(d_model, d_model)
        self.out_proj = nn.Dense(d_model, d_model)

    def construct(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, T, D = q.shape
        # Simple scaled dot-product as fallback
        # (full auto-correlation with FFT to be refined during research phase)
        q = self.q_proj(q).view(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = self.k_proj(k).view(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = self.v_proj(v).view(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        scale = self.d_k ** -0.5
        attn = ops.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)

        out = out.transpose(0, 2, 1, 3).view(B, T, D)
        return self.out_proj(out)


class _AutoformerEncoderLayer(nn.Cell):
    """Single Autoformer encoder layer with decomposition."""

    def __init__(self, d_model, n_heads, d_ff, moving_avg, dropout, factor):
        super().__init__()
        self.attn = _AutoCorrelationAttention(d_model, n_heads, factor)
        self.decomp1 = _MovingAvgDecomp(moving_avg)
        self.decomp2 = _MovingAvgDecomp(moving_avg)
        self.ffn = nn.SequentialCell(
            nn.Dense(d_model, d_ff),
            nn.GELU(approximate=False),
            nn.Dropout(p=dropout),
            nn.Dense(d_ff, d_model),
        )
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x: Tensor) -> Tensor:
        attn_out = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x, _ = self.decomp1(x)

        ff_out = self.ffn(x)
        x = x + self.dropout(ff_out)
        x, _ = self.decomp2(x)
        return x


class Autoformer(BaseSeizureModel):
    """Autoformer for EEG seizure classification."""

    def __init__(
        self,
        num_channels: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        factor: int = 3,
        moving_avg: int = 25,
        dropout: float = 0.1,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes)

        # Project channels to d_model: (B, C, T) → (B, T, d_model)
        self.input_proj = nn.Dense(num_channels, d_model)

        self.encoder = nn.SequentialCell(
            *[
                _AutoformerEncoderLayer(d_model, n_heads, d_ff, moving_avg, dropout, factor)
                for _ in range(e_layers)
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Dense(d_model, num_classes)

    def construct(self, x: Tensor) -> Tensor:
        # x: (B, C, T) → (B, T, C)
        x = x.transpose(0, 2, 1)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.encoder(x)
        # Pool over time: (B, d_model, T) → (B, d_model)
        x = x.transpose(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
