from pathlib import Path


def save_best_marker(run_dir: str, metric_name: str, value: float) -> None:
    """Write a minimal marker used by downstream scripts."""
    path = Path(run_dir) / "best_metric.txt"
    path.write_text(f"{metric_name}={value:.6f}\n", encoding="utf-8")
