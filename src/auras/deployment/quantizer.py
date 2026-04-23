"""Post-training quantization and MindSpore Lite conversion for edge deployment.

Converts ``.mindir`` models to ``.ms`` format for on-device inference,
with optional INT8 quantization to reduce size and latency.

Three backends are tried in order:
1. ``mindspore_lite`` Python API (preferred — when pip package available)
2. ``converter_lite`` CLI binary (from MindSpore Lite release tarball)
3. Skip with a clear message when neither is available

Usage::

    from auras.deployment.quantizer import convert_to_ms, quantize_model

    # Plain conversion (no quantization)
    convert_to_ms("exports/model.mindir", "exports/model.ms")

    # With weight quantization
    quantize_model("exports/model.mindir", "exports/model_quant.ms",
                   quant_type="WEIGHT_QUANT")
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _has_python_api() -> bool:
    """Check if mindspore_lite Python package is importable."""
    try:
        import mindspore_lite  # noqa: F401
        return True
    except ImportError:
        return False


def _has_cli() -> bool:
    """Check if converter_lite binary is on PATH."""
    return shutil.which("converter_lite") is not None


# ── Plain .mindir → .ms conversion (no quantization) ────────────


def convert_to_ms(mindir_path: str, output_path: str) -> Path:
    """Convert a ``.mindir`` model to ``.ms`` format for MindSpore Lite.

    Parameters
    ----------
    mindir_path : str
        Path to the exported ``.mindir`` file.
    output_path : str
        Desired output path.  The ``.ms`` suffix is added automatically
        by the converter if not present.

    Returns
    -------
    Path
        The generated ``.ms`` file.
    """
    mindir = Path(mindir_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if _has_python_api():
        return _convert_python_api(mindir, out)
    if _has_cli():
        return _convert_cli(mindir, out)

    raise EnvironmentError(
        "MindSpore Lite tools not found.\n"
        "Install one of:\n"
        "  - pip install mindspore-lite  (if available for your platform)\n"
        "  - Download converter_lite from:\n"
        "    https://www.mindspore.cn/lite/docs/en/master/use/downloads.html\n"
        "    and add to PATH."
    )


def _convert_python_api(mindir: Path, out: Path) -> Path:
    """Convert via mindspore_lite.Converter Python API."""
    import mindspore_lite as mslite

    converter = mslite.Converter()
    converter.save_type = mslite.ModelType.MINDIR_LITE
    # output_file should be without .ms suffix — converter adds it
    out_stem = str(out.with_suffix(""))
    converter.converter(fmk_type=mslite.FmkType.FMK_MINDIR,
                        model_file=str(mindir),
                        output_file=out_stem)
    ms_path = Path(out_stem + ".ms")
    print(f"Converted to .ms via Python API: {ms_path}")
    return ms_path


def _convert_cli(mindir: Path, out: Path) -> Path:
    """Convert via converter_lite CLI binary."""
    out_stem = str(out.with_suffix(""))
    cmd = [
        "converter_lite",
        f"--fmk=MINDIR",
        f"--modelFile={mindir}",
        f"--outputFile={out_stem}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"converter_lite failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )
    ms_path = Path(out_stem + ".ms")
    print(f"Converted to .ms via CLI: {ms_path}")
    return ms_path


# ── Quantized conversion ─────────────────────────────────────────


def quantize_model(
    mindir_path: str,
    output_path: str,
    quant_type: str = "WEIGHT_QUANT",
) -> Path:
    """Quantize a ``.mindir`` model and output ``.ms`` for MindSpore Lite.

    Parameters
    ----------
    mindir_path : str
        Path to the exported ``.mindir`` file.
    output_path : str
        Path for the quantized ``.ms`` output.
    quant_type : str
        Quantization strategy:
        - ``WEIGHT_QUANT``: weights-only INT8 (smallest size, no calibration)
        - ``FULL_QUANT``: weights + activations INT8 (needs calibration data)
        - ``DYNAMIC_QUANT``: dynamic activation quantization

    Returns
    -------
    Path
        The generated quantized ``.ms`` file.
    """
    mindir = Path(mindir_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if _has_python_api():
        return _quantize_python_api(mindir, out, quant_type)
    if _has_cli():
        return _quantize_cli(mindir, out, quant_type)

    raise EnvironmentError(
        "MindSpore Lite tools not found for quantization.\n"
        "Install converter_lite or mindspore-lite Python package."
    )


def _quantize_python_api(mindir: Path, out: Path, quant_type: str) -> Path:
    """Quantize via mindspore_lite.Converter Python API."""
    import mindspore_lite as mslite

    converter = mslite.Converter()
    converter.save_type = mslite.ModelType.MINDIR_LITE

    # Configure quantization
    quant_cfg = {"quant_type": quant_type}
    converter.set_config_info("quantization", quant_cfg)

    out_stem = str(out.with_suffix(""))
    converter.converter(fmk_type=mslite.FmkType.FMK_MINDIR,
                        model_file=str(mindir),
                        output_file=out_stem)
    ms_path = Path(out_stem + ".ms")
    print(f"Quantized ({quant_type}) to .ms via Python API: {ms_path}")
    return ms_path


def _quantize_cli(mindir: Path, out: Path, quant_type: str) -> Path:
    """Quantize via converter_lite CLI with config file."""
    import tempfile

    # Write quantization config file
    config_content = f"[quantization]\nquant_type={quant_type}\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cfg", delete=False
    ) as f:
        f.write(config_content)
        config_path = f.name

    try:
        out_stem = str(out.with_suffix(""))
        cmd = [
            "converter_lite",
            f"--fmk=MINDIR",
            f"--modelFile={mindir}",
            f"--outputFile={out_stem}",
            f"--configFile={config_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"converter_lite quantization failed (exit {result.returncode}):\n"
                f"  stdout: {result.stdout}\n"
                f"  stderr: {result.stderr}"
            )
    finally:
        Path(config_path).unlink(missing_ok=True)

    ms_path = Path(out_stem + ".ms")
    print(f"Quantized ({quant_type}) to .ms via CLI: {ms_path}")
    return ms_path
