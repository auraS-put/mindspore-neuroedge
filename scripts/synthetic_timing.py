#!/usr/bin/env python
"""Self-contained GPU timing benchmark using synthetic data.

Measures wall-clock time per training step for each model architecture
using random tensors of the correct shape. Since GPU compute time depends
only on tensor shapes and batch size (not data values), this gives
accurate per-step timing that can be multiplied by the real step count.

Usage (on the GPU instance):
    pip install mindspore==2.8.0  # if not already installed
    python synthetic_timing.py

No real data needed — generates random (4, 1024) tensors.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np


def install_mindspore():
    """Try to ensure mindspore is available."""
    try:
        import mindspore
        print(f"MindSpore version: {mindspore.__version__}")
        return True
    except ImportError:
        print("MindSpore not found, attempting install...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mindspore==2.8.0"])
        return True


def setup():
    import mindspore as ms
    ms.set_context(mode=ms.PYNATIVE_MODE)
    device = ms.get_context("device_target")
    print(f"Device: {device}")
    if device == "CPU":
        print("WARNING: Running on CPU. Results won't represent GPU timing.")
        print("Set device_target to GPU: ms.set_context(device_target='GPU')")
    return device


# ─── Model definitions (self-contained, no external imports) ─────────────

def build_cnn_bilstm_attn():
    """Paper 18 — DS-Conv + BiLSTM + Transformer attention. ~210K params."""
    import mindspore.nn as nn
    from mindspore import Tensor, ops

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            # Stem: 3 DS-Conv blocks
            self.stem = nn.SequentialCell(
                nn.Conv1d(4, 64, 9, pad_mode="same"),
                nn.BatchNorm1d(64),
                nn.GELU(approximate=False),
                nn.Conv1d(64, 64, 7, pad_mode="same", group=64),
                nn.Conv1d(64, 64, 1),
                nn.BatchNorm1d(64),
                nn.GELU(approximate=False),
                nn.Conv1d(64, 64, 5, pad_mode="same", group=64),
                nn.Conv1d(64, 64, 1),
                nn.BatchNorm1d(64),
                nn.GELU(approximate=False),
            )
            self.pool = nn.AdaptiveAvgPool1d(64)
            self.lstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
            self.attn_norm = nn.LayerNorm([128])
            self.attn = nn.MultiheadAttention(128, 4, dropout=0.1)
            self.head = nn.SequentialCell(
                nn.Dense(128, 64),
                nn.GELU(approximate=False),
                nn.Dropout(p=0.1),
                nn.Dense(64, 2),
            )

        def construct(self, x):
            x = self.stem(x)
            x = self.pool(x)
            x = x.transpose(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.transpose(1, 0, 2)
            normed = self.attn_norm(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            x = x.mean(axis=0)
            return self.head(x)

    return Model()


def build_eegformer():
    """Paper 02 — Patch Transformer. ~280K params."""
    import mindspore.nn as nn
    from mindspore import Tensor, ops

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.SequentialCell(
                nn.Conv1d(4, 64, 5, stride=5, pad_mode="valid"),
                nn.GELU(approximate=False),
                nn.Conv1d(64, 128, 5, stride=5, pad_mode="valid"),
                nn.GELU(approximate=False),
            )
            self.pos_embed = nn.Embedding(512, 128)
            self.norm1 = nn.LayerNorm([128])
            self.attn = nn.MultiheadAttention(128, 8, dropout=0.1)
            self.norm2 = nn.LayerNorm([128])
            self.ffn = nn.SequentialCell(
                nn.Dense(128, 256),
                nn.GELU(approximate=False),
                nn.Dropout(p=0.1),
                nn.Dense(256, 128),
                nn.Dropout(p=0.1),
            )
            self.head = nn.Dense(128, 2)

        def construct(self, x):
            x = self.patch_embed(x)  # (B, 128, seq_len)
            x = x.transpose(0, 2, 1)  # (B, seq_len, 128)
            B, T, D = x.shape
            pos = self.pos_embed(ops.arange(T).astype("int32"))
            x = x + pos.unsqueeze(0)
            x = x.transpose(1, 0, 2)  # (T, B, 128)
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            x = x.mean(axis=0)  # (B, 128)
            return self.head(x)

    return Model()


def build_cnn_informer():
    """Paper 10 — CNN + ProbSparse Informer. ~196K params."""
    import mindspore.nn as nn
    from mindspore import Tensor, ops

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.cnn = nn.SequentialCell(
                nn.Conv1d(4, 32, 7, pad_mode="same"),
                nn.BatchNorm1d(32),
                nn.GELU(approximate=False),
                nn.Conv1d(32, 64, 5, pad_mode="same"),
                nn.BatchNorm1d(64),
                nn.GELU(approximate=False),
                nn.Conv1d(64, 64, 3, pad_mode="same"),
                nn.BatchNorm1d(64),
                nn.GELU(approximate=False),
            )
            self.pool = nn.AdaptiveAvgPool1d(128)
            # 3 encoder layers with distilling (halves seq each time)
            self.enc1_attn = nn.MultiheadAttention(64, 4, dropout=0.1)
            self.enc1_norm = nn.LayerNorm([64])
            self.enc1_ffn = nn.SequentialCell(nn.Dense(64, 256), nn.ELU(), nn.Dense(256, 64))
            self.enc1_pool = nn.MaxPool1d(2, 2)
            self.enc2_attn = nn.MultiheadAttention(64, 4, dropout=0.1)
            self.enc2_norm = nn.LayerNorm([64])
            self.enc2_ffn = nn.SequentialCell(nn.Dense(64, 256), nn.ELU(), nn.Dense(256, 64))
            self.enc2_pool = nn.MaxPool1d(2, 2)
            self.enc3_attn = nn.MultiheadAttention(64, 4, dropout=0.1)
            self.enc3_norm = nn.LayerNorm([64])
            self.enc3_ffn = nn.SequentialCell(nn.Dense(64, 256), nn.ELU(), nn.Dense(256, 64))
            self.enc3_pool = nn.MaxPool1d(2, 2)
            self.head = nn.Dense(64, 2)

        def _enc_layer(self, x, attn, norm, ffn, pool):
            # x: (B, T, 64)
            xt = x.transpose(1, 0, 2)  # (T, B, 64)
            a, _ = attn(xt, xt, xt)
            x = norm(x + a.transpose(1, 0, 2))
            x = x + ffn(x)
            # Distilling: pool in time
            x = pool(x.transpose(0, 2, 1)).transpose(0, 2, 1)
            return x

        def construct(self, x):
            x = self.cnn(x)  # (B, 64, 1024)
            x = self.pool(x)  # (B, 64, 128)
            x = x.transpose(0, 2, 1)  # (B, 128, 64)
            x = self._enc_layer(x, self.enc1_attn, self.enc1_norm, self.enc1_ffn, self.enc1_pool)
            x = self._enc_layer(x, self.enc2_attn, self.enc2_norm, self.enc2_ffn, self.enc2_pool)
            x = self._enc_layer(x, self.enc3_attn, self.enc3_norm, self.enc3_ffn, self.enc3_pool)
            x = x.mean(axis=1)  # (B, 64)
            return self.head(x)

    return Model()


def build_pyramidal_cnn_bilstm():
    """Paper 04 — Pyramidal stride CNN + BiLSTM. ~33K params."""
    import mindspore.nn as nn
    from mindspore import Tensor, ops

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(4, 8, 7, stride=4, pad_mode="same")
            self.conv2 = nn.Conv1d(8, 16, 5, stride=4, pad_mode="same")
            self.conv3 = nn.Conv1d(16, 32, 3, stride=4, pad_mode="same")
            self.bn1 = nn.BatchNorm1d(8)
            self.bn2 = nn.BatchNorm1d(16)
            self.bn3 = nn.BatchNorm1d(32)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(32, 32, bidirectional=True, batch_first=True)
            self.head = nn.Dense(64, 2)

        def construct(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = x.transpose(0, 2, 1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            return self.head(x)

    return Model()


def build_cam_cnn_bilstm():
    """Paper 19/20 — Channel Attention + CNN + BiLSTM. ~45K params."""
    import mindspore.nn as nn
    from mindspore import Tensor, ops

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            # Channel attention (SE block)
            self.ca_fc = nn.Dense(4, 4)
            self.ca_sigmoid = nn.Sigmoid()
            # CNN stem
            self.conv1 = nn.Conv1d(4, 64, 7, pad_mode="same")
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 64, 5, pad_mode="same")
            self.bn2 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(64)
            self.lstm = nn.LSTM(64, 40, bidirectional=True, batch_first=True)
            self.head = nn.Dense(80, 2)

        def construct(self, x):
            # Channel attention
            gap = x.mean(axis=-1)  # (B, 4)
            w = self.ca_sigmoid(self.ca_fc(gap)).unsqueeze(-1)
            x = x * w
            # CNN
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.transpose(0, 2, 1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            return self.head(x)

    return Model()


# ─── Timing logic ─────────────────────────────────────────────────────

MODELS = {
    "cnn_bilstm_attn":      (build_cnn_bilstm_attn, 64, 100),    # (builder, batch_size, full_epochs)
    "eegformer":            (build_eegformer, 16, 150),
    "cnn_informer":         (build_cnn_informer, 32, 100),
    "pyramidal_cnn_bilstm": (build_pyramidal_cnn_bilstm, 32, 300),
    "cam_cnn_bilstm":       (build_cam_cnn_bilstm, 32, 40),
}

FULL_DATASET_WINDOWS = 451344  # total windows in siena_sop_merged.npz


def time_model(name: str, builder, batch_size: int, n_steps: int = 200):
    """Time forward+backward for n_steps and return avg time/step."""
    import mindspore as ms
    import mindspore.nn as nn
    from mindspore import Tensor, ops

    print(f"\n  Building {name} (batch={batch_size})...")
    model = builder()
    model.set_train(True)

    # Count params
    n_params = sum(p.size for p in model.trainable_params())
    print(f"  Params: {n_params:,}")

    # Loss + optimizer (to include backward pass in timing)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)

    # Forward + backward function
    def forward_fn(x, y):
        logits = model(x)
        return loss_fn(logits, y)

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(x, y):
        loss, grads = grad_fn(x, y)
        optimizer(grads)
        return loss

    # Generate random data
    x_batch = Tensor(np.random.randn(batch_size, 4, 1024).astype(np.float32))
    y_batch = Tensor(np.random.randint(0, 2, (batch_size,)).astype(np.int32))

    # Warmup (first few steps are slower due to JIT compilation)
    print(f"  Warmup (10 steps)...")
    for _ in range(10):
        train_step(x_batch, y_batch)

    # Timed run
    print(f"  Timing ({n_steps} steps)...")
    t0 = time.time()
    for _ in range(n_steps):
        train_step(x_batch, y_batch)
    elapsed = time.time() - t0
    time_per_step = elapsed / n_steps

    print(f"  ✓ {elapsed:.2f}s total, {time_per_step*1000:.2f} ms/step")
    return {
        "time_per_step_ms": round(time_per_step * 1000, 2),
        "params": n_params,
        "batch_size": batch_size,
    }


def main():
    install_mindspore()
    device = setup()

    print(f"\n{'═'*60}")
    print(f"  Synthetic Timing Benchmark")
    print(f"  Device: {device}")
    print(f"  Full dataset: {FULL_DATASET_WINDOWS:,} windows")
    print(f"{'═'*60}")

    results = {}
    for name, (builder, batch_size, full_epochs) in MODELS.items():
        try:
            r = time_model(name, builder, batch_size, n_steps=200)
            steps_per_epoch = FULL_DATASET_WINDOWS // batch_size
            total_steps = steps_per_epoch * full_epochs
            total_hours = (r["time_per_step_ms"] / 1000) * total_steps / 3600

            r.update({
                "steps_per_epoch": steps_per_epoch,
                "full_epochs": full_epochs,
                "total_steps": total_steps,
                "estimated_hours": round(total_hours, 2),
            })
            results[name] = r
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    # Summary
    print(f"\n{'═'*70}")
    print(f"  TIMING RESULTS — {device}")
    print(f"{'═'*70}")
    print(f"  {'Model':<25} {'Params':>8} {'ms/step':>8} {'steps/ep':>9} {'epochs':>7} {'Est.hours':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*9} {'─'*7} {'─'*10}")
    total_h = 0
    for name, r in results.items():
        if "error" not in r:
            print(f"  {name:<25} {r['params']:>8,} {r['time_per_step_ms']:>8.2f}"
                  f" {r['steps_per_epoch']:>9,} {r['full_epochs']:>7}"
                  f" {r['estimated_hours']:>10.1f}")
            total_h += r["estimated_hours"]
        else:
            print(f"  {name:<25} {'ERROR':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*9} {'─'*7} {'─'*10}")
    print(f"  {'TOTAL (sequential)':<53} {total_h:>10.1f}h")
    print(f"  {'TOTAL (5 parallel jobs)':<53} {max(r.get('estimated_hours',0) for r in results.values()):>10.1f}h")
    print(f"{'═'*70}\n")

    # Save JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
