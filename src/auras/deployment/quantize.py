from pathlib import Path


def quantize_model(ms_path: str, out_path: str) -> Path:
    """Placeholder for post-training quantization step."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"placeholder quantized artifact from {ms_path}\n", encoding="utf-8")
    return out
