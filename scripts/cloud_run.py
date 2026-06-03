#!/usr/bin/env python
"""Multi-platform cloud runner for the timing benchmark.

Generates platform-specific, self-contained notebooks or runs locally.

Usage:
    # Generate a Colab notebook (upload manually to colab.research.google.com):
    python scripts/cloud_run.py --platform colab

    # Generate a Kaggle notebook (upload via kaggle.com/notebooks):
    python scripts/cloud_run.py --platform kaggle

    # Run directly on this machine:
    python scripts/cloud_run.py --platform local

    # (Future) Submit to ModelArts when real-name auth is done:
    python scripts/cloud_run.py --platform modelarts
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BENCHMARK_SCRIPT_MS = SCRIPT_DIR / "synthetic_timing.py"
BENCHMARK_SCRIPT_TORCH = SCRIPT_DIR / "synthetic_timing_torch.py"
OUTPUT_DIR = SCRIPT_DIR.parent / "experiments" / "notebooks"


# ─── Platform registry ───────────────────────────────────────────────────

PLATFORMS = {}


def register_platform(name: str):
    """Decorator to register a platform handler."""
    def _deco(fn):
        PLATFORMS[name] = fn
        return fn
    return _deco


# ─── Notebook generation helpers ─────────────────────────────────────────

def _make_notebook(cells: list[dict], metadata: dict | None = None) -> dict:
    """Build a Jupyter .ipynb structure."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": metadata or {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": cells,
    }


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(True), "outputs": [], "execution_count": None}


def _read_benchmark_source(framework: str = "torch") -> str:
    """Read benchmark source for the specified framework."""
    if framework == "torch":
        return BENCHMARK_SCRIPT_TORCH.read_text()
    return BENCHMARK_SCRIPT_MS.read_text()


# ─── Platform: Colab ─────────────────────────────────────────────────────

@register_platform("colab")
def generate_colab(args: argparse.Namespace) -> Path:
    """Generate a self-contained Google Colab notebook (PyTorch)."""
    cells = [
        _md_cell(
            "# Seizure Model Timing Benchmark (Colab)\n"
            "\n"
            "**Runtime → Change runtime type → T4 GPU** before running.\n"
            "\n"
            "Measures forward+backward time per training step for each model architecture\n"
            "using synthetic data (PyTorch). Results estimate full MindSpore training time\n"
            "on ModelArts (same CUDA/cuDNN kernels, ~10-15% variance).\n"
        ),
        _code_cell(
            "# Check GPU availability\n"
            "!nvidia-smi\n"
        ),
        _code_cell(
            "# PyTorch is pre-installed on Colab — verify GPU\n"
            "import torch\n"
            "print(f'PyTorch {torch.__version__}')\n"
            "print(f'CUDA available: {torch.cuda.is_available()}')\n"
            "if torch.cuda.is_available():\n"
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
        ),
        _code_cell(_read_benchmark_source("torch")),
        _code_cell(
            "# Run the benchmark\n"
            "results = main()\n"
        ),
        _code_cell(
            "# Save results JSON for download\n"
            "import json\n"
            "with open('timing_results.json', 'w') as f:\n"
            "    json.dump(results, f, indent=2)\n"
            "print('Saved to timing_results.json')\n"
            "from google.colab import files\n"
            "files.download('timing_results.json')\n"
        ),
    ]

    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    }

    nb = _make_notebook(cells, metadata)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "timing_benchmark_colab.ipynb"
    out_path.write_text(json.dumps(nb, indent=2))
    print(f"Generated: {out_path}")
    print(f"Upload to: https://colab.research.google.com/ → File → Upload notebook")
    return out_path


# ─── Platform: Kaggle ────────────────────────────────────────────────────

@register_platform("kaggle")
def generate_kaggle(args: argparse.Namespace) -> Path:
    """Generate a self-contained Kaggle notebook (PyTorch)."""
    cells = [
        _md_cell(
            "# Seizure Model Timing Benchmark (Kaggle)\n"
            "\n"
            "**Settings → Accelerator → GPU P100** before running.\n"
            "\n"
            "Measures forward+backward time per training step using PyTorch.\n"
            "Results estimate full MindSpore training time on ModelArts.\n"
        ),
        _code_cell(
            "# Check GPU\n"
            "!nvidia-smi\n"
        ),
        _code_cell(
            "# Verify PyTorch GPU\n"
            "import torch\n"
            "print(f'PyTorch {torch.__version__}')\n"
            "print(f'CUDA available: {torch.cuda.is_available()}')\n"
            "if torch.cuda.is_available():\n"
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
        ),
        _code_cell(_read_benchmark_source("torch")),
        _code_cell(
            "# Run the benchmark\n"
            "results = main()\n"
        ),
    ]

    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
        "kaggle": {"accelerator": "gpu", "isGpuEnabled": True},
    }

    nb = _make_notebook(cells, metadata)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "timing_benchmark_kaggle.ipynb"
    out_path.write_text(json.dumps(nb, indent=2))
    print(f"Generated: {out_path}")
    print(f"Upload to: https://www.kaggle.com/code → New Notebook → File → Upload")
    return out_path


# ─── Platform: Local ─────────────────────────────────────────────────────

@register_platform("local")
def run_local(args: argparse.Namespace) -> Path:
    """Run the benchmark directly on the local machine (MindSpore)."""
    print(f"Running {BENCHMARK_SCRIPT_MS} locally...")
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT_MS)],
        cwd=str(SCRIPT_DIR.parent),
    )
    sys.exit(result.returncode)


# ─── Platform: ModelArts ─────────────────────────────────────────────────

@register_platform("modelarts")
def submit_modelarts(args: argparse.Namespace) -> Path:
    """Submit to Huawei ModelArts (requires real-name auth + agency configured)."""
    print("ModelArts submission requires real-name authentication.")
    print("Use: python scripts/run_experiment.py --backend modelarts --config configs/experiment/cloud_timing.yaml")
    sys.exit(1)


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate or run timing benchmark for a cloud platform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--platform",
        choices=list(PLATFORMS.keys()),
        default="colab",
        help="Target platform.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for generated notebooks.",
    )

    args = parser.parse_args()

    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)

    handler = PLATFORMS[args.platform]
    handler(args)


if __name__ == "__main__":
    main()
