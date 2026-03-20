"""Shared building-block modules for Sprint-2 EEG seizure models.

All classes follow the MindSpore ``nn.Cell`` contract and operate on
float32 tensors.  Shape conventions:
  - CNN-domain : (B, C, T)  — batch, channels, time-steps
  - Sequence-domain : (B, T, d)  — batch, time-steps, feature-dim
  - Transformer-domain (legacy): (T, B, d) — used by _TransformerBlock1D

Exported:
  DepthwiseSeparableConv1D
  ResDSBlock
  ChannelAttention1D
  ProbSparseAttention
  AttentionDistilling
  InformerEncoderLayer
"""

from __future__ import annotations

import math

import mindspore.nn as nn
from mindspore import Tensor, ops

from auras.models.base import BaseSeizureModel  # noqa: F401 – keep import order


# ── Depthwise-Separable Conv ─────────────────────────────────────

class DepthwiseSeparableConv1D(nn.Cell):
    """Depthwise-separable 1-D convolution (MobileNet factorisation).

    Depthwise conv (group=in_ch) followed by a 1×1 pointwise conv.
    Uses ``pad_mode="same"`` so that the time dimension is preserved
    when ``stride=1``.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            stride=stride, pad_mode="same", group=in_ch,
        )
        self.pw = nn.Conv1d(in_ch, out_ch, 1, pad_mode="valid")

    def construct(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


# ── Residual DS Block ────────────────────────────────────────────

class ResDSBlock(nn.Cell):
    """Residual depthwise-separable block (Paper 18 (Mehrabi et al. — ConvSNN): ConvSNN architecture).

    DS-Conv → BN → GELU → Dropout + identity skip.
    Input and output channel sizes are identical (no projection needed).
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.SequentialCell(
            DepthwiseSeparableConv1D(channels, channels, kernel_size),
            nn.BatchNorm1d(channels),
            nn.GELU(approximate=False),
        )
        self.drop = nn.Dropout(p=dropout)

    def construct(self, x: Tensor) -> Tensor:
        return x + self.drop(self.conv(x))


# ── Channel Attention ────────────────────────────────────────────

class ChannelAttention1D(nn.Cell):
    """Squeeze-and-Excitation channel attention for 1-D signals.

    Based on Paper 20 (Wang Y. et al. — CAM-CNN): global average pool over time →
    dense projection → sigmoid → channel-wise scaling.

    Input / Output: (B, C, T).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.fc = nn.Dense(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x: Tensor) -> Tensor:
        gap = x.mean(axis=-1)           # (B, C)
        w = self.sigmoid(self.fc(gap))  # (B, C)
        return x * w.unsqueeze(-1)      # (B, C, T)


# ── ProbSparse Attention ─────────────────────────────────────────

class ProbSparseAttention(nn.Cell):
    """ProbSparse self-attention from Informer (Zhou et al., NeurIPS 2021).

    Reduces the quadratic complexity of standard attention to
    O(L log L) by selecting only the top-u "dominant" queries
    (u = factor × ln(L)).  The remaining queries are filled with
    the context mean of V.

    Input / Output: (B, T, d_model)  — batch-first.

    Parameters
    ----------
    d_model : int
        Model dimension (must be divisible by n_heads).
    n_heads : int
        Number of attention heads.
    factor : int
        ProbSparse sampling factor c.  Paper 10 (Li et al. — CNN-Informer) found c=3 optimal.
    max_len : int
        Largest sequence length expected (used to pre-compute n_top
        so that TopK uses a compile-time constant k).
    dropout : float
        Attention and output dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 3,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.factor = factor
        # Pre-compute k for TopK (compile-time constant → graph-mode safe)
        self.n_top = max(1, int(factor * math.log(max_len + 1)))

        self.q_proj = nn.Dense(d_model, d_model)
        self.k_proj = nn.Dense(d_model, d_model)
        self.v_proj = nn.Dense(d_model, d_model)
        self.out_proj = nn.Dense(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.topk = ops.TopK(sorted=False)

    def construct(self, x: Tensor) -> Tensor:
        B, T, d = x.shape[0], x.shape[1], x.shape[2]
        H, d_k = self.n_heads, self.d_k

        Q = self.q_proj(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # (B,H,T,d_k)
        K = self.k_proj(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        # Full attention scores (used for sparsity measure)
        scores = ops.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale  # (B,H,T,T)

        # Sparsity measure: M(qi) = max_j(score) − mean_j(score)
        M = scores.max(axis=-1) - scores.mean(axis=-1)  # (B,H,T)

        # Clamp n_top to actual sequence length (PyNative: T is a Python int)
        n_top = min(self.n_top, T)

        # Select top-u dominant queries per (batch × head)
        M_flat = M.reshape(B * H, T)                              # (B*H, T)
        topk_vals, _ = self.topk(M_flat, n_top)                   # (B*H, n_top)
        threshold = topk_vals.min(axis=-1).reshape(B, H, 1)       # (B, H, 1)

        # Binary mask: dominant queries use real attention; rest use mean-V
        mask = (M >= threshold).astype(scores.dtype).unsqueeze(-1)  # (B,H,T,1)

        attn = self.softmax(scores)                                 # (B,H,T,T)
        v_mean = V.mean(axis=-2, keep_dims=True).broadcast_to(V.shape)  # (B,H,T,d_k)
        attn_out = ops.matmul(attn, V)                             # (B,H,T,d_k)
        out = mask * attn_out + (1.0 - mask) * v_mean             # (B,H,T,d_k)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, d)
        return self.out_proj(self.dropout(out))


# ── Attention Distilling ─────────────────────────────────────────

class AttentionDistilling(nn.Cell):
    """Self-attention distilling from Informer (halves the sequence length).

    Conv1d(k=3, pad=same) + ELU + MaxPool(2) applied in the temporal
    dimension.  Operates on (B, T, d_model) — batch-first.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, pad_mode="same")
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model) → need (B, d_model, T) for Conv1d/MaxPool1d
        x = x.transpose(0, 2, 1)               # (B, d_model, T)
        x = self.pool(self.act(self.conv(x)))   # (B, d_model, T//2)
        return x.transpose(0, 2, 1)            # (B, T//2, d_model)


# ── Informer Encoder Layer ───────────────────────────────────────

class InformerEncoderLayer(nn.Cell):
    """Single Informer encoder layer.

    ProbSparse Self-Attention → Add & LN → FFN(ELU) → Add & LN →
    AttentionDistilling (T → T//2).

    Parameters
    ----------
    d_model : int
        Feature dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension (typically 4 × d_model).
    factor : int
        ProbSparse sampling factor (Paper 10 (Li et al. — CNN-Informer): c=3 optimal).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor, dropout=dropout)
        self.norm1 = nn.LayerNorm([d_model])
        self.ffn = nn.SequentialCell(
            nn.Dense(d_model, d_ff),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Dense(d_ff, d_model),
            nn.Dropout(p=dropout),
        )
        self.norm2 = nn.LayerNorm([d_model])
        self.distilling = AttentionDistilling(d_model)

    def construct(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        x = self.norm1(x + self.attn(x))    # post-norm (original Informer)
        x = self.norm2(x + self.ffn(x))
        return self.distilling(x)            # (B, T//2, d_model)
