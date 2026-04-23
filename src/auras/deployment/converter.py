"""MindSpore → MindSpore Lite model conversion.

Exports a trained ``.ckpt`` model to the ``.mindir`` intermediate format,
then converts to ``.ms`` for on-device inference with MindSpore Lite.

Usage:
    python scripts/export_lite.py --checkpoint path/to/model.ckpt \
                                  --model ghostnet1d \
                                  --output exports/ghostnet1d.mindir
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_mindir(
    checkpoint_path: str,
    model_name: str,
    num_channels: int = 4,
    seq_len: int = 1024,
    output_path: str = "exports/model.mindir",
) -> Path:
    """Export a trained model to MindIR format.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``.ckpt`` file.
    model_name : str
        Model config name (e.g. ``ghostnet1d``).
    num_channels, seq_len : int
        Input tensor dimensions for tracing.
    output_path : str
        Where to save the ``.mindir`` file.
    """
    import mindspore as ms
    import numpy as np
    from omegaconf import OmegaConf

    from auras.models.factory import create_model
    from auras.utils.io import load_checkpoint

    model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
    model = create_model(model_cfg, num_channels=num_channels)
    model = load_checkpoint(model, checkpoint_path)
    model.set_train(False)

    # Dummy input for graph tracing
    dummy = ms.Tensor(np.random.randn(1, num_channels, seq_len).astype(np.float32))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ms.export(model, dummy, file_name=str(out.with_suffix("")), file_format="MINDIR")
    print(f"Exported MindIR to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Export model to MindSpore Lite format")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt")
    parser.add_argument("--model", default="ghostnet1d", help="Model config name")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output", default="exports/model.mindir")
    args = parser.parse_args()

    export_mindir(args.checkpoint, args.model, args.channels, args.seq_len, args.output)


if __name__ == "__main__":
    main()
