from pathlib import Path

from auras.training.train import run_training


def test_training_placeholder_creates_metrics(tmp_path: Path):
    run_dir = run_training("configs/model/lstm.yaml", "configs/train/default.yaml", str(tmp_path))
    assert (run_dir / "metrics.json").exists()
