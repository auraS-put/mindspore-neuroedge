"""Export best model to MindSpore Lite format.

Thin wrapper around auras.deployment.converter for CLI usage.

Usage:
    python scripts/export_lite.py --checkpoint experiments/runs/ghostnet1d/final.ckpt \
                                  --model ghostnet1d
"""

from __future__ import annotations

from auras.deployment.converter import main

if __name__ == "__main__":
    main()
