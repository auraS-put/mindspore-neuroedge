"""Provider-agnostic experiment launchers.

Each launcher converts a local experiment config into the appropriate
submission format for its target compute backend.

Supported backends:
    local      — run in-process on the current machine
    modelarts  — Huawei Cloud ModelArts training job
    aws        — AWS SageMaker training job

All launchers share the same interface::

    launcher = build_launcher(backend="modelarts", backend_cfg=cfg.backend)
    launcher.submit(experiment_cfg_path="configs/experiment/dry_run.yaml")

Adding a new backend
--------------------
1. Subclass :class:`BaseLauncher`.
2. Implement ``submit()`` and ``status()``.
3. Register in :func:`build_launcher`.
"""

from __future__ import annotations

from auras.launchers.base import BaseLauncher
from auras.launchers.local import LocalLauncher
from auras.launchers.modelarts import ModelArtsLauncher
from auras.launchers.aws import AWSLauncher


def build_launcher(backend: str, backend_cfg=None) -> BaseLauncher:
    """Factory — return the right launcher for *backend*.

    Parameters
    ----------
    backend : str
        ``"local"``, ``"modelarts"``, or ``"aws"``.
    backend_cfg : OmegaConf DictConfig, optional
        Backend-specific subsection of the run config (e.g. ``cfg.backend``).

    Returns
    -------
    BaseLauncher subclass ready to call ``.submit()``.
    """
    backend = backend.lower().strip()
    if backend == "local":
        return LocalLauncher(backend_cfg)
    elif backend == "modelarts":
        return ModelArtsLauncher(backend_cfg)
    elif backend in ("aws", "sagemaker"):
        return AWSLauncher(backend_cfg)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            "Supported: local, modelarts, aws"
        )


__all__ = [
    "BaseLauncher",
    "LocalLauncher",
    "ModelArtsLauncher",
    "AWSLauncher",
    "build_launcher",
]
