"""Optuna hyperparameter search integration.

Defines the objective function and search space, creates an Optuna
study backed by SQLite, and runs trials with pruning.

Usage:
    python -m auras.experiment.optuna_search
    python -m auras.experiment.optuna_search --config configs/optuna/search.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def _create_objective(base_cfg, search_cfg):
    """Return an Optuna objective function closed over the config."""

    def objective(trial):
        import numpy as np
        from auras.utils.reproducibility import seed_everything

        seed_everything(base_cfg.seed)

        # ── Sample hyperparameters from search space ──────────────
        overrides = {}
        space = search_cfg.search_space

        for param_name, spec in space.items():
            spec = dict(spec)
            ptype = spec.pop("type")
            if ptype == "log_float":
                overrides[param_name] = trial.suggest_float(param_name, log=True, **spec)
            elif ptype == "float":
                overrides[param_name] = trial.suggest_float(param_name, **spec)
            elif ptype == "int":
                overrides[param_name] = trial.suggest_int(param_name, **spec)
            elif ptype == "categorical":
                overrides[param_name] = trial.suggest_categorical(param_name, spec["choices"])

        # ── Build run config with sampled HPs ─────────────────────
        run_cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"training": overrides}))

        # TODO: Run actual training and return validation recall
        #   from auras.training.trainer import train
        #   result = train(run_cfg)
        #   return result.val_recall

        # Placeholder: return random metric for testing the pipeline
        val_recall = np.random.uniform(0.5, 1.0)
        trial.report(val_recall, step=0)

        return val_recall

    return objective


def run_search(search_cfg_path: str, base_cfg_path: str = "configs/config.yaml") -> None:
    """Create and run an Optuna study."""
    import optuna

    search_cfg = OmegaConf.load(search_cfg_path)
    base_cfg = OmegaConf.load(base_cfg_path)

    # Resolve sub-configs
    if base_cfg.get("defaults"):
        for default in base_cfg.defaults:
            if isinstance(default, dict):
                for key, val in default.items():
                    if key != "_self_":
                        base_cfg[key] = OmegaConf.load(f"configs/{key}/{val}.yaml")

    # Setup storage
    storage_path = Path(search_cfg.storage.replace("sqlite:///", ""))
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # Pruner
    pruner_name = search_cfg.pruner.name
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=search_cfg.pruner.n_startup_trials,
            n_warmup_steps=search_cfg.pruner.n_warmup_steps,
        )
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    # Sampler
    sampler_name = search_cfg.sampler.name
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=base_cfg.seed)
    elif sampler_name == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=base_cfg.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=base_cfg.seed)

    study = optuna.create_study(
        study_name=search_cfg.study_name,
        storage=search_cfg.storage,
        direction=search_cfg.direction,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    objective = _create_objective(base_cfg, search_cfg)

    timeout = search_cfg.get("timeout_hours", None)
    timeout_sec = timeout * 3600 if timeout else None

    study.optimize(
        objective,
        n_trials=search_cfg.n_trials,
        timeout=timeout_sec,
    )

    print(f"\n═══ Best trial ═══")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")


def main():
    parser = argparse.ArgumentParser(description="Optuna HP search for auraS")
    parser.add_argument("--config", default="configs/optuna/search.yaml", help="Optuna config")
    parser.add_argument("--base-config", default="configs/config.yaml", help="Base training config")
    args = parser.parse_args()
    run_search(args.config, args.base_config)


if __name__ == "__main__":
    main()
