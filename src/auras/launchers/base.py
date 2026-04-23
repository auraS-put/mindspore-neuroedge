"""Abstract base class for compute-backend launchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JobStatus:
    """Unified job status returned by any launcher."""

    job_id: str
    backend: str
    state: str                        # "pending" | "running" | "succeeded" | "failed"
    output_dir: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.state == "succeeded"

    @property
    def failed(self) -> bool:
        return self.state == "failed"


class BaseLauncher(ABC):
    """Interface all compute backends must implement.

    Parameters
    ----------
    cfg : OmegaConf DictConfig, optional
        Backend-specific configuration block (credentials, region, …).
    """

    def __init__(self, cfg=None):
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def submit(
        self,
        experiment_cfg_path: str,
        *,
        wait: bool = True,
        dry_run: bool = False,
    ) -> JobStatus:
        """Submit an experiment and optionally block until completion.

        Parameters
        ----------
        experiment_cfg_path : str
            Path to the experiment YAML consumed by ``run_experiment()``.
        wait : bool
            If True, block until the job reaches a terminal state and
            return the final :class:`JobStatus`.  If False, return
            immediately with state ``"pending"`` or ``"running"``.
        dry_run : bool
            Print the submission payload but do not actually submit.

        Returns
        -------
        JobStatus
        """

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Query the current status of a previously submitted job.

        Parameters
        ----------
        job_id : str
            Backend-specific job identifier returned by :meth:`submit`.
        """

    # ------------------------------------------------------------------
    # Optional helpers (subclasses may override)
    # ------------------------------------------------------------------

    def cancel(self, job_id: str) -> None:
        """Cancel a running job (best-effort)."""
        raise NotImplementedError(f"{type(self).__name__} does not support job cancellation.")

    def download_outputs(self, job_id: str, local_dir: str) -> None:
        """Download output artefacts (checkpoints, metrics) to *local_dir*."""
        raise NotImplementedError(f"{type(self).__name__} does not support output download.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(cfg={self._cfg})"
