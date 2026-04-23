"""Download the Siena Scalp EEG dataset from Kaggle.

Usage:
    python scripts/download_siena.py [--output-dir data/raw/siena]

Requires:
    - KAGGLE_USERNAME and KAGGLE_KEY in .env or environment
    - pip install kaggle
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "abhishekinnvonix/epilepsy-seizure-dataset-seina-scalp-complete"


def _ensure_kaggle_creds() -> None:
    """Load .env and verify Kaggle credentials are available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print(
            "ERROR: Set KAGGLE_USERNAME and KAGGLE_KEY in .env or environment.\n"
            "  Get your key from https://www.kaggle.com/settings → API → Create New Token"
        )
        sys.exit(1)


def download(output_dir: Path) -> None:
    _ensure_kaggle_creds()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {KAGGLE_DATASET} → {output_dir}")
    subprocess.check_call(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(output_dir),
            "--unzip",
        ]
    )
    print(f"Done. Files in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Siena Scalp EEG from Kaggle")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/siena"),
        help="Directory to save the dataset",
    )
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
