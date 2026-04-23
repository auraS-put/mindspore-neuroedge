"""Unified experiment launcher script.

Dispatches an experiment to one of three compute backends:
  local      — run in-process on the current machine (default)
  modelarts  — submit a Huawei Cloud ModelArts training job
  aws        — submit an AWS SageMaker training job

Quick start
-----------

  # Local dry run (pipeline smoke-test, < 5 min):
  python scripts/run_experiment.py \\
      --config configs/experiment/dry_run.yaml \\
      --backend local

  # ModelArts dry run (needs MA_AK / MA_SK / MA_PROJECT_ID env vars):
  python scripts/run_experiment.py \\
      --config configs/experiment/dry_run.yaml \\
      --backend modelarts

  # ModelArts — full baseline experiment:
  python scripts/run_experiment.py \\
      --config configs/experiment/baseline.yaml \\
      --backend modelarts \\
      --backend-config configs/backends/modelarts.yaml \\
      --wait

  # Only print submission payload (no actual job):
  python scripts/run_experiment.py \\
      --config configs/experiment/dry_run.yaml \\
      --backend modelarts \\
      --dry-run

  # AWS SageMaker:
  python scripts/run_experiment.py \\
      --config configs/experiment/dry_run.yaml \\
      --backend aws \\
      --backend-config configs/backends/aws.yaml

Calling from the ModelArts / SageMaker container
-------------------------------------------------
The entry script is this same file.  The cloud launcher injects:

  --config <experiment_cfg>
  --data-dir  <mounted_input_path>
  --output-dir <mounted_output_path>

so the runner writes checkpoints and metrics to the mounted OBS / S3 path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable when running as a bare script (not via pip install)
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_backend_cfg(path: str | None):
    """Load backend-specific config from YAML (or return None)."""
    if path is None:
        return None
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch an auraS experiment on any compute backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment selection ─────────────────────────────────────────
    parser.add_argument(
        "--config",
        default="configs/experiment/dry_run.yaml",
        help="Experiment YAML config (e.g. dry_run, baseline, ablation).",
    )

    # ── Backend selection ────────────────────────────────────────────
    parser.add_argument(
        "--backend",
        choices=["local", "modelarts", "aws"],
        default="local",
        help="Compute backend to use.",
    )
    parser.add_argument(
        "--backend-config",
        default=None,
        metavar="YAML",
        help=(
            "Path to backend-specific config YAML "
            "(e.g. configs/backends/modelarts.yaml).  "
            "Defaults: configs/backends/<backend>.yaml if it exists."
        ),
    )

    # ── Runtime flags ────────────────────────────────────────────────
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Block until the submitted job finishes (default: True).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit and return immediately without blocking.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the submission payload but do not actually run / submit.",
    )

    # ── Cloud path overrides (injected by launchers) ─────────────────
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Processed data directory override (used by cloud backends).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory override (checkpoints, results JSON).",
    )

    # ── Post-run actions ─────────────────────────────────────────────
    parser.add_argument(
        "--download",
        default=None,
        metavar="LOCAL_DIR",
        help="After the job finishes, download artefacts here (cloud backends only).",
    )

    args = parser.parse_args()
    wait = args.wait and not args.no_wait

    # ------------------------------------------------------------------
    # Local backend: run directly (skip launcher overhead when possible)
    # ------------------------------------------------------------------
    if args.backend == "local" and not args.dry_run:
        import datetime, os

        from auras.experiment.runner import run_experiment

        # Resolve output dir early so we can place the log there.
        from omegaconf import OmegaConf as _OC
        _exp = _OC.load(args.config)
        _out_root = args.output_dir or f"experiments/runs/{_exp.name}"
        Path(_out_root).mkdir(parents=True, exist_ok=True)

        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_path = Path(_out_root) / f"training_{_ts}.log"

        class _Tee:
            """Mirrors stdout writes to a log file."""
            def __init__(self, path):
                self._f = open(path, "w", buffering=1)
                self._orig = sys.stdout
            def write(self, data):
                self._orig.write(data)
                self._f.write(data)
            def flush(self):
                self._orig.flush()
                self._f.flush()
            def close(self):
                self._f.close()
            # Proxy any attribute the real stdout has (isatty, fileno, …)
            def __getattr__(self, name):
                return getattr(self._orig, name)

        _tee = _Tee(_log_path)
        sys.stdout = _tee
        try:
            out = run_experiment(
                args.config,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
            )
        finally:
            sys.stdout = _tee._orig
            _tee.close()

        print(f"\nDone. Results in: {out}")
        print(f"Full log saved to: {_log_path}")
        return

    # ------------------------------------------------------------------
    # Cloud / dry-run path: build launcher and submit
    # ------------------------------------------------------------------
    # Resolve backend config path
    backend_cfg_path = args.backend_config
    if backend_cfg_path is None:
        default_cfg = Path(f"configs/backends/{args.backend}.yaml")
        if default_cfg.exists():
            backend_cfg_path = str(default_cfg)

    backend_cfg = _load_backend_cfg(backend_cfg_path)

    from auras.launchers import build_launcher
    launcher = build_launcher(args.backend, backend_cfg)

    print(f"Backend : {args.backend}")
    print(f"Config  : {args.config}")
    if backend_cfg_path:
        print(f"Backend config: {backend_cfg_path}")
    if args.dry_run:
        print("Mode    : DRY RUN (no actual submission)")
    print()

    job = launcher.submit(
        args.config,
        wait=wait,
        dry_run=args.dry_run,
    )

    print(f"\nJob ID  : {job.job_id}")
    print(f"State   : {job.state}")
    if job.output_dir:
        print(f"Output  : {job.output_dir}")
    if job.error:
        print(f"Error   : {job.error}")
        sys.exit(1)

    # Optionally download
    if args.download and not args.dry_run and job.succeeded:
        launcher.download_outputs(job.job_id, args.download)


if __name__ == "__main__":
    main()
