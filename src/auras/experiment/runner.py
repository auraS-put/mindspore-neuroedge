"""Experiment runner — orchestrates multi-model, multi-dataset experiments.

Reads an experiment config (e.g. ``configs/experiment/baseline.yaml``)
and runs all specified combinations, logging results to monitoring backends.

Usage:
    python -m auras.experiment.runner --config configs/experiment/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from auras.utils.reproducibility import seed_everything


def run_experiment(exp_cfg_path: str) -> None:
    """Execute an experiment plan defined in YAML."""
    exp = OmegaConf.load(exp_cfg_path)
    print(f"═══ Experiment: {exp.name} ═══")
    print(f"Datasets: {exp.datasets}")
    print(f"Models:   {exp.models}")
    print(f"Metrics:  {exp.metrics.track}")

    results = []

    for dataset_name in exp.datasets:
        data_cfg = OmegaConf.load(f"configs/data/{dataset_name}.yaml")

        for model_name in exp.models:
            model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")

            for rep in range(exp.get("repetitions", 1)):
                seed = 42 + rep
                seed_everything(seed)

                print(f"\n─── {model_name} on {dataset_name} (rep {rep+1}) ───")

                # Build merged config for the trainer
                run_cfg = OmegaConf.create({
                    "seed": seed,
                    "project_name": "auraS",
                    "output_dir": f"experiments/runs/{exp.name}/{dataset_name}/{model_name}/rep_{rep}",
                    "data": data_cfg,
                    "model": model_cfg,
                    "training": OmegaConf.load("configs/training/default.yaml"),
                })

                # TODO: call train() when training loop is fully operational
                #   from auras.training.trainer import train
                #   train(run_cfg)
                print(f"  → Would train {model_name} (seed={seed})")
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "rep": rep,
                    "seed": seed,
                    "status": "placeholder",
                })

    # Save results summary
    out_dir = Path(f"experiments/runs/{exp.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "results_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nExperiment summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run auraS experiment plan")
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML config")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
