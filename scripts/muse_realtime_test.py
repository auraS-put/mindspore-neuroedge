#!/usr/bin/env python
"""Muse headband real-time inference test.

This script:
  1. Receives EEG data from a Muse headband via OSC (streamed from the Muse app)
  2. Buffers 4-second windows (1024 samples @ 256 Hz)
  3. Applies preprocessing (bandpass, notch, z-score)
  4. Runs inference through a MindSpore Lite model (or dummy model for testing)
  5. Reports prediction results

Setup:
  1. Connect Muse to phone via Muse app
  2. In Muse app: Settings → OSC Output → set IP to this computer, port 5000
  3. Enable streaming
  4. Run this script:  python scripts/muse_realtime_test.py

For testing WITHOUT a headband (dummy data):
  python scripts/muse_realtime_test.py --simulate

OSC channel mapping (Muse 2/S):
  /muse/eeg → [TP9, AF7, AF8, TP10, AUX] (5 values per packet)
  We reorder to match our model: [AF7, AF8, TP9, TP10] → model indices [0,1,2,3]
"""
from __future__ import annotations

import argparse
import collections
import threading
import time
from typing import Optional

import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────

SAMPLE_RATE = 256          # Muse 2/S native EEG sampling rate
WINDOW_SECONDS = 4         # 4-second window
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS  # 1024
STRIDE_SAMPLES = SAMPLE_RATE  # 1-second stride (75% overlap)
N_CHANNELS = 4

# Muse OSC delivers: [TP9, AF7, AF8, TP10, AUX_RIGHT]
# Our model expects:  [AF7, AF8, TP9, TP10] (indices 0,1,2,3)
# Reorder: Muse index [1, 2, 0, 3] → model index [0, 1, 2, 3]
MUSE_TO_MODEL_INDICES = [1, 2, 0, 3]

# Preprocessing params (matching configs/data/siena_sop5.yaml)
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
NOTCH_HZ = 50.0


# ─── Signal processing ───────────────────────────────────────────────────

def design_bandpass(low: float, high: float, fs: float, order: int = 2):
    """Design bandpass filter coefficients (Butterworth IIR)."""
    from scipy.signal import butter
    nyq = fs / 2
    return butter(order, [low / nyq, high / nyq], btype='band')


def design_notch(freq: float, fs: float, Q: float = 30.0):
    """Design notch filter coefficients."""
    from scipy.signal import iirnotch
    return iirnotch(freq, Q, fs)


class Preprocessor:
    """Real-time signal preprocessor matching training pipeline."""

    def __init__(self, fs: float = SAMPLE_RATE):
        self.fs = fs
        self.bp_b, self.bp_a = design_bandpass(BANDPASS_LOW, BANDPASS_HIGH, fs)
        self.notch_b, self.notch_a = design_notch(NOTCH_HZ, fs)

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Process a (4, 1024) window: bandpass → notch → zscore."""
        from scipy.signal import filtfilt

        x = window.copy().astype(np.float64)

        for ch in range(x.shape[0]):
            # Bandpass 0.5-45 Hz
            x[ch] = filtfilt(self.bp_b, self.bp_a, x[ch])
            # Notch 50 Hz
            x[ch] = filtfilt(self.notch_b, self.notch_a, x[ch])

        # Z-score per channel
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std[std < 1e-8] = 1.0  # avoid division by zero
        x = (x - mean) / std

        return x.astype(np.float32)


# ─── Model inference ─────────────────────────────────────────────────────

class DummyModel:
    """Dummy model for testing pipeline without MindSpore Lite."""

    def predict(self, x: np.ndarray) -> dict:
        """Simulate inference: random prediction with timing."""
        t0 = time.time()
        # Simulate some compute (hash-based pseudo-prediction from input stats)
        energy = np.sum(x ** 2, axis=(1, 2))
        prob_preictal = 1.0 / (1.0 + np.exp(-0.001 * (energy[0] - 1000)))
        elapsed_ms = (time.time() - t0) * 1000
        return {
            "prob_preictal": float(prob_preictal),
            "prob_interictal": float(1.0 - prob_preictal),
            "latency_ms": round(elapsed_ms, 3),
        }


class MindSporeLiteModel:
    """Inference via MindSpore Lite (for exported .ms models)."""

    def __init__(self, model_path: str):
        import mindspore_lite as mslite
        context = mslite.Context()
        context.target = ["cpu"]
        context.cpu.thread_num = 2
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR_LITE, context)
        self.inputs = self.model.get_inputs()

    def predict(self, x: np.ndarray) -> dict:
        t0 = time.time()
        self.inputs[0].set_data_from_numpy(x)
        outputs = self.model.predict(self.inputs)
        logits = outputs[0].get_data_to_numpy()
        elapsed_ms = (time.time() - t0) * 1000

        # Sigmoid for probabilities
        probs = 1.0 / (1.0 + np.exp(-logits[0]))
        return {
            "prob_interictal": float(probs[0]),
            "prob_preictal": float(probs[1]),
            "latency_ms": round(elapsed_ms, 3),
        }


# ─── EEG Buffer ──────────────────────────────────────────────────────────

class EEGBuffer:
    """Thread-safe ring buffer for EEG samples."""

    def __init__(self, n_channels: int = N_CHANNELS, window_size: int = WINDOW_SAMPLES):
        self.n_channels = n_channels
        self.window_size = window_size
        self.buffer = np.zeros((n_channels, window_size * 2), dtype=np.float32)
        self.write_pos = 0
        self.samples_received = 0
        self.lock = threading.Lock()

    def add_sample(self, channels: list[float]):
        """Add a single multi-channel sample."""
        with self.lock:
            pos = self.write_pos % (self.window_size * 2)
            for ch in range(self.n_channels):
                self.buffer[ch, pos] = channels[ch]
            self.write_pos += 1
            self.samples_received += 1

    def get_window(self) -> Optional[np.ndarray]:
        """Get the latest full window (4, 1024). Returns None if not enough data."""
        with self.lock:
            if self.samples_received < self.window_size:
                return None
            end = self.write_pos % (self.window_size * 2)
            if end >= self.window_size:
                return self.buffer[:, end - self.window_size:end].copy()
            else:
                # Wrap around
                part1 = self.buffer[:, -(self.window_size - end):]
                part2 = self.buffer[:, :end]
                return np.hstack([part1, part2])


# ─── OSC Receiver ────────────────────────────────────────────────────────

def start_osc_server(buffer: EEGBuffer, port: int = 5000):
    """Start OSC server to receive Muse EEG data."""
    from pythonosc import dispatcher, osc_server

    disp = dispatcher.Dispatcher()

    def eeg_handler(address, *args):
        """Handle /muse/eeg messages: [TP9, AF7, AF8, TP10, AUX]"""
        if len(args) >= 4:
            # Reorder to model order: [AF7, AF8, TP9, TP10]
            reordered = [args[i] for i in MUSE_TO_MODEL_INDICES]
            buffer.add_sample(reordered)

    disp.map("/muse/eeg", eeg_handler)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", port), disp)
    print(f"OSC server listening on port {port}")
    print(f"  Expecting: /muse/eeg [TP9, AF7, AF8, TP10, AUX]")
    print(f"  Reordering to model: [AF7, AF8, TP9, TP10]")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


# ─── Simulator ───────────────────────────────────────────────────────────

def start_simulator(buffer: EEGBuffer):
    """Generate synthetic EEG-like data at 256 Hz for testing without headband."""
    def _sim_loop():
        t = 0
        dt = 1.0 / SAMPLE_RATE
        while True:
            # Simulate EEG: alpha (10 Hz) + noise
            sample = []
            for ch in range(N_CHANNELS):
                # Mix of alpha, theta, and noise
                alpha = 20 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
                theta = 10 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
                noise = np.random.randn() * 5
                sample.append(alpha + theta + noise)
            buffer.add_sample(sample)
            t += dt
            time.sleep(dt)

    thread = threading.Thread(target=_sim_loop, daemon=True)
    thread.start()
    print("Simulator started: generating synthetic EEG at 256 Hz")
    print("  4 channels, alpha (10 Hz) + theta (6 Hz) + noise")


# ─── Main loop ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Muse real-time inference test")
    parser.add_argument("--port", type=int, default=5000, help="OSC UDP port")
    parser.add_argument("--simulate", action="store_true", help="Use simulated EEG (no headband needed)")
    parser.add_argument("--model", type=str, default=None, help="Path to .ms model file (uses dummy if not provided)")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory of .ms models (runs ALL models per window)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Alert threshold for preictal probability")
    args = parser.parse_args()

    # Initialize components
    buffer = EEGBuffer()
    preprocessor = Preprocessor()

    # Load model(s)
    models: dict[str, object] = {}
    if args.model_dir:
        from pathlib import Path
        model_dir = Path(args.model_dir)
        ms_files = sorted(model_dir.glob("*.ms"))
        if not ms_files:
            print(f"No .ms files found in {model_dir}")
            return
        for f in ms_files:
            print(f"Loading: {f.name}")
            models[f.stem] = MindSporeLiteModel(str(f))
        print(f"Loaded {len(models)} models")
    elif args.model:
        print(f"Loading model: {args.model}")
        models["model"] = MindSporeLiteModel(args.model)
    else:
        print("Using DUMMY model (for pipeline testing)")
        models["dummy"] = DummyModel()

    # Start data source
    if args.simulate:
        start_simulator(buffer)
    else:
        start_osc_server(buffer, args.port)

    # Inference loop
    multi = len(models) > 1
    print(f"\n{'='*60}")
    print(f"  Real-time Seizure Prediction Pipeline")
    print(f"  Models: {', '.join(models.keys())}")
    print(f"  Window: {WINDOW_SECONDS}s ({WINDOW_SAMPLES} samples)")
    print(f"  Stride: {STRIDE_SAMPLES} samples (1s)")
    print(f"  Preprocessing: bandpass {BANDPASS_LOW}-{BANDPASS_HIGH} Hz, notch {NOTCH_HZ} Hz, zscore")
    print(f"  Threshold: {args.threshold}")
    print(f"{'='*60}\n")

    last_inference_sample = 0
    inference_count = 0

    try:
        while True:
            # Wait for enough new samples (stride)
            if buffer.samples_received - last_inference_sample < STRIDE_SAMPLES:
                time.sleep(0.05)
                continue

            # Get window
            window = buffer.get_window()
            if window is None:
                time.sleep(0.1)
                continue

            # Preprocess
            processed = preprocessor.process_window(window)

            # Inference (add batch dim)
            x = processed[np.newaxis, ...]  # (1, 4, 1024)

            inference_count += 1
            last_inference_sample = buffer.samples_received

            if multi:
                # Run all models, show comparison
                print(f"  [{inference_count:4d}] ", end="")
                for name, mdl in models.items():
                    result = mdl.predict(x)
                    prob = result["prob_preictal"]
                    tag = "!" if prob > args.threshold else "."
                    print(f" {name}={prob:.2f}{tag}", end="")
                print(f"  ({result['latency_ms']:.1f}ms)")
            else:
                # Single model — detailed display
                name = list(models.keys())[0]
                result = models[name].predict(x)
                prob = result["prob_preictal"]
                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                status = "⚠️  ALERT" if prob > args.threshold else "✓ Normal"
                print(f"  [{inference_count:4d}] preictal={prob:.3f} |{bar}| {status}  ({result['latency_ms']:.1f}ms)")

    except KeyboardInterrupt:
        print(f"\n\nStopped after {inference_count} inferences.")
        print(f"Total samples received: {buffer.samples_received}")
        elapsed = buffer.samples_received / SAMPLE_RATE
        print(f"Effective recording time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
