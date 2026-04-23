"""Experiment runner — orchestrates multi-model, multi-dataset experiments.

Reads an experiment config (e.g. ``configs/experiment/dry_run.yaml``) and
runs all specified combinations of deep-learning models and classical ML
baselines, logging results via the configured monitoring backend.

The runner is intentionally stateless: each call to :func:`run_experiment`
produces a self-contained results directory and JSON summary.

Usage
-----
    # run from within the container / local machine:
    python -m auras.experiment.runner --config configs/experiment/dry_run.yaml

    # override output directory (useful on ModelArts / SageMaker):
    python -m auras.experiment.runner \\
        --config configs/experiment/dry_run.yaml \\
        --data-dir /cache/data/processed \\
        --output-dir /cache/output
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from auras.utils.reproducibility import seed_everything


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_paths(cfg, data_dir: Optional[str], output_dir: Optional[str]):
    """Override data/output directories when running on cloud backends."""
    if data_dir:
        cfg.data.processed_dir = data_dir
    if output_dir:
        cfg.output_dir = output_dir
    return cfg


def _build_run_cfg(
    exp: Any,
    dataset_name: str,
    model_name: str,
    rep: int,
    data_dir: Optional[str],
    output_dir: Optional[str],
    training_cfg_name: str = "default",
) -> Any:
    """Merge all sub-configs into one training config."""
    data_cfg = OmegaConf.load(f"configs/data/{dataset_name}.yaml")
    model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
    training_cfg = OmegaConf.load(f"configs/training/{training_cfg_name}.yaml")

    base_output = output_dir or f"experiments/runs/{exp.name}"
    run_cfg = OmegaConf.create({
        "seed": 42 + rep,
        "project_name": "auraS",
        "output_dir": f"{base_output}/{dataset_name}/{model_name}/rep_{rep}",
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
    })

    if data_dir:
        run_cfg.data.processed_dir = data_dir

    return run_cfg


def _apply_dry_run_caps(run_cfg: Any, exp: Any) -> Any:
    """Restrict data size and epochs for dry-run mode."""
    dr = exp.get("dry_run", {})
    if not dr.get("enabled", False):
        return run_cfg

    run_cfg.training.epochs = min(run_cfg.training.epochs, 2)
    run_cfg.training.batch_size = min(run_cfg.training.batch_size, 16)
    run_cfg.training.early_stopping.patience = 999  # don't stop early in dry-run

    # Cap number of windows via the dataset loader
    run_cfg.data["dry_run_max_windows"] = dr.get("max_windows", 500)
    run_cfg.data["dry_run_max_subjects"] = dr.get("max_subjects", 3)

    return run_cfg


def _build_logger(run_cfg: Any):
    """Return the appropriate monitoring logger for this run."""
    from auras.monitoring.base_logger import ConsoleLogger

    # Check for ModelArts environment
    if _is_modelarts():
        from auras.monitoring.modelarts_logger import ModelArtsLogger
        return ModelArtsLogger(run_cfg)

    # WandB if configured and available
    if run_cfg.get("wandb", False):
        try:
            from auras.monitoring.wandb_logger import WandBLogger
            return WandBLogger(run_cfg)
        except (ImportError, Exception):
            pass

    return ConsoleLogger()


def _is_modelarts() -> bool:
    """Detect if we are running inside a ModelArts training container."""
    import os
    return bool(os.environ.get("MA_INPUTS_0_URL") or os.environ.get("DLS_TASK_INDEX"))


# ---------------------------------------------------------------------------
# Deep-learning training run
# ---------------------------------------------------------------------------

def _run_deep_model(
    model_name: str,
    run_cfg: Any,
    logger,
    preloaded: dict = None,
) -> Dict[str, Any]:
    """Train one deep model and return a metrics dict."""
    import mindspore as ms
    from auras.models.factory import create_model
    from auras.training.losses import build_loss
    from auras.training.lr_schedulers import build_lr_schedule
    from auras.training.trainer import _build_datasets, _build_optimizer, _train_loop
    from auras.training.evaluator import evaluate_epoch

    import datetime
    t0 = time.time()
    seed_everything(run_cfg.seed)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] Loading data …")
    train_ds, val_ds, test_ds, meta = _build_datasets(run_cfg, preloaded=preloaded)
    print(
        f"  Data: {meta['train_samples']} train"
        f" / {meta['val_samples']} val"
        f" / {meta['test_samples']} test"
        f"  (pos={meta['train_positive']}, neg={meta['train_negative']})"
    )
    logger.log_metrics(
        {"n_train": meta["train_samples"], "n_pos": meta["train_positive"]},
        step=0,
    )

    n_channels = len(run_cfg.data.channels.selected)
    model = create_model(run_cfg.model, num_channels=n_channels)
    print(f"  Model: {model_name} ({model.count_params():,} params)")

    loss_fn = build_loss(run_cfg.training, meta["train_positive"], meta["train_negative"])
    steps_per_epoch = max(meta["train_samples"] // run_cfg.training.batch_size, 1)
    lr_schedule = build_lr_schedule(run_cfg.training, steps_per_epoch)
    optimizer = _build_optimizer(model, lr_schedule, run_cfg.training)

    def make_val_iter():
        return val_ds.create_tuple_iterator()

    print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] Training …")
    _train_loop(model, loss_fn, optimizer, train_ds, run_cfg.training,
                val_iterator=make_val_iter)

    print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] Evaluating on test set …")
    # Evaluate on test split
    test_result = evaluate_epoch(model, test_ds.create_tuple_iterator())
    metrics = test_result.segment.to_dict()
    metrics["fp_per_hour"] = test_result.fp_per_hour
    metrics["seizure_detection_rate"] = test_result.seizure_detection_rate
    metrics["elapsed_s"] = round(time.time() - t0, 1)

    logger.log_metrics(metrics)

    # Save checkpoint + per-model log
    out_dir = Path(run_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(model, str(out_dir / "model.ckpt"))

    # Persist metrics alongside the checkpoint
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(
        f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] DONE"
        f"  recall={metrics.get('recall', 0):.4f}"
        f"  f1={metrics.get('f1', 0):.4f}"
        f"  auc={metrics.get('auc_roc', 0):.4f}"
        f"  ({metrics['elapsed_s']}s)"
    )

    return {"model_name": model_name, "type": "deep", **metrics}


# ---------------------------------------------------------------------------
# Classical ML run
# ---------------------------------------------------------------------------

def _run_classical_loso(
    model_names: List[str],
    run_cfg: Any,
    logger,
    preloaded: dict = None,
) -> List[Dict[str, Any]]:
    """Run DWT feature extraction + LOSO for all classical models."""
    from auras.data.preprocess import dwt_features
    from auras.experiment.cross_validation import loso_splits
    from auras.models.classical_ml import train_and_evaluate

    npz_path = (
        Path(run_cfg.data.processed_dir) / f"{run_cfg.data.name}.npz"
    )
    if preloaded is not None:
        data = preloaded
    elif not npz_path.exists():
        print(f"  [skip classical ML] processed data not found: {npz_path}")
        return []
    else:
        data = np.load(npz_path, mmap_mode='r')
    X_raw = data["X"]
    label_key = run_cfg.data.get("target_key", "y")
    y = data[label_key]
    subjects = data["subjects"] if "subjects" in data else np.zeros(len(y), dtype="U10")

    # Dry-run cap — track absolute indices so X_dwt can be sliced identically
    abs_idx = np.arange(len(y))
    max_w = run_cfg.data.get("dry_run_max_windows", len(y))
    max_s = run_cfg.data.get("dry_run_max_subjects", None)
    if max_s is not None:
        kept = np.isin(subjects, np.unique(subjects)[:max_s])
        X_raw, y, subjects, abs_idx = X_raw[kept], y[kept], subjects[kept], abs_idx[kept]
    if len(y) > max_w:
        idx = np.random.choice(len(y), max_w, replace=False)
        idx.sort()
        X_raw, y, subjects, abs_idx = X_raw[idx], y[idx], subjects[idx], abs_idx[idx]

    print(f"  Extracting DWT features for {len(y)} windows …")
    if "X_dwt" in data:
        # Use precomputed features from the merged NPZ (fast path)
        X_feats = data["X_dwt"][abs_idx]
        print(f"  Using precomputed X_dwt {X_feats.shape}")
        print(f"  Using precomputed X_dwt {X_feats.shape}")
    else:
        X_feats = np.stack([dwt_features(x) for x in X_raw])

    all_results = []
    for name in model_names:
        t0 = time.time()
        fold_metrics: List[Dict] = []
        for train_idx, test_idx in loso_splits(subjects, y):
            n_pos = int(y[train_idx].sum())
            if n_pos == 0:
                continue
            fold = train_and_evaluate(
                name, X_feats[train_idx], y[train_idx],
                X_feats[test_idx], y[test_idx],
            )
            fold_metrics.append(fold)

        if fold_metrics:
            mean = {k: float(np.mean([f[k] for f in fold_metrics]))
                    for k in ["recall", "specificity", "f1", "auc_roc"]}
            mean["elapsed_s"] = round(time.time() - t0, 1)
            mean["n_folds"] = len(fold_metrics)
            logger.log_metrics({f"classical/{name}/{k}": v for k, v in mean.items()})
            all_results.append({"model_name": name, "type": "classical", **mean})

    return all_results


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_experiment(
    exp_cfg_path: str,
    *,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Execute an experiment plan defined in YAML.

    Parameters
    ----------
    exp_cfg_path : str
        Path to the experiment YAML (e.g. ``configs/experiment/dry_run.yaml``).
    data_dir : str, optional
        Override for processed data directory (used by cloud backends that
        mount object storage, e.g. ``/cache/data`` on ModelArts).
    output_dir : str, optional
        Override for output root (checkpoints, results JSON).

    Returns
    -------
    Path — directory where ``results_summary.json`` was written.
    """
    exp = OmegaConf.load(exp_cfg_path)
    is_dry = exp.get("dry_run", {}).get("enabled", False)
    training_cfg_name = "dry_run" if is_dry else "default"

    tag = "DRY-RUN" if is_dry else "FULL"
    print(f"\n{'═'*60}")
    print(f"  Experiment: {exp.name}  [{tag}]")
    print(f"  Datasets:   {list(exp.datasets)}")
    print(f"  DL models:  {list(exp.models)}")
    classical = list(exp.get("classical_models", []))
    if classical:
        print(f"  Classical:  {classical}")
    print(f"{'═'*60}\n")

    all_results: List[Dict] = []

    for dataset_name in exp.datasets:
        # ── Load processed data ONCE per dataset ─────────────────────────
        data_cfg = OmegaConf.load(f"configs/data/{dataset_name}.yaml")
        npz_path = Path(data_cfg.processed_dir) / f"{data_cfg.name}.npz"
        if data_dir:
            npz_path = Path(data_dir) / f"{data_cfg.name}.npz"

        preloaded = None
        if npz_path.exists():
            print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Mapping data: {npz_path.name} ({npz_path.stat().st_size / 1e9:.2f} GB) …")
            preloaded = np.load(str(npz_path), mmap_mode='r')
            label_key = data_cfg.get("target_key", "y")
            y_arr = preloaded[label_key]
            print(f"  X shape: {preloaded['X'].shape}  y ({label_key}): {y_arr.shape}  pos={int(y_arr.sum())} neg={int((y_arr==0).sum())}")
            has_subjects = "subjects" in preloaded
            if not has_subjects:
                print("  WARNING: no 'subjects' key — LOSO splits unavailable; regenerate with prepare_dataset.py")
        else:
            print(f"  WARNING: {npz_path} not found — all models will fail. Run prepare_dataset.py first.")

        for rep in range(exp.get("repetitions", 1)):

            # ── Deep learning models ─────────────────────────────────
            for model_name in exp.models:
                seed_everything(42 + rep)
                print(f"\n─── DL: {model_name} / {dataset_name} (rep {rep + 1}) ───")

                run_cfg = _build_run_cfg(
                    exp, dataset_name, model_name, rep,
                    data_dir, output_dir, training_cfg_name,
                )
                run_cfg = _apply_dry_run_caps(run_cfg, exp)
                logger = _build_logger(run_cfg)
                logger.log_config({"model": model_name, "dataset": dataset_name, "rep": rep})

                try:
                    result = _run_deep_model(model_name, run_cfg, logger, preloaded=preloaded)
                    result.update({"dataset": dataset_name, "rep": rep, "status": "ok"})
                except Exception as exc:
                    import traceback
                    print(f"  [ERROR] {model_name}: {exc}")
                    traceback.print_exc()
                    result = {
                        "model_name": model_name,
                        "dataset": dataset_name,
                        "rep": rep,
                        "status": "error",
                        "error": str(exc),
                    }
                finally:
                    logger.finish()

                all_results.append(result)

            # ── Classical ML models (once per dataset/rep) ───────────
            if classical:
                print(f"\n─── Classical ML / {dataset_name} (rep {rep + 1}) ───")
                run_cfg = _build_run_cfg(
                    exp, dataset_name, "cnn_baseline", rep,
                    data_dir, output_dir, training_cfg_name,
                )
                run_cfg = _apply_dry_run_caps(run_cfg, exp)
                logger = _build_logger(run_cfg)
                try:
                    cl_results = _run_classical_loso(classical, run_cfg, logger, preloaded=preloaded)
                    for r in cl_results:
                        r.update({"dataset": dataset_name, "rep": rep, "status": "ok"})
                    all_results.extend(cl_results)
                except Exception as exc:
                    import traceback
                    print(f"  [ERROR] classical ML: {exc}")
                    traceback.print_exc()
                finally:
                    logger.finish()

    # ── Save summary ─────────────────────────────────────────────────
    base_out = Path(output_dir or f"experiments/runs/{exp.name}")
    base_out.mkdir(parents=True, exist_ok=True)
    summary_path = base_out / "results_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))

    _print_summary(all_results)
    print(f"\nResults saved to: {summary_path}")
    return base_out


def _print_summary(results: List[Dict]) -> None:
    print(f"\n{'═'*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*70}")
    fmt = f"  {'Model':<28} {'Type':<10} {'Recall':>7} {'Spec':>7} {'F1':>7} {'AUC':>7}  {'Status'}"
    print(fmt)
    print(f"  {'-'*66}")
    for r in results:
        name = r.get("model_name", "?")[:27]
        mtype = r.get("type", "?")[:9]
        rec = f"{r.get('recall', 0):.3f}" if r.get("status") == "ok" else "  -  "
        spec = f"{r.get('specificity', 0):.3f}" if r.get("status") == "ok" else "  -  "
        f1 = f"{r.get('f1', 0):.3f}" if r.get("status") == "ok" else "  -  "
        auc = f"{r.get('auc_roc', 0):.3f}" if r.get("status") == "ok" else "  -  "
        print(f"  {name:<28} {mtype:<10} {rec:>7} {spec:>7} {f1:>7} {auc:>7}  {r.get('status', '?')}")
    print(f"{'═'*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run an auraS experiment plan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config path")
    parser.add_argument("--data-dir", default=None,
                        help="Override processed data directory (used by cloud launchers)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output root directory")
    args = parser.parse_args()
    run_experiment(args.config, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
