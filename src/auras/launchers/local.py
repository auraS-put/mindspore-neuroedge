"""Local backend launcher — runs the experiment in-process.

No network calls or credentials needed.  Used for development, debugging,
and CI.  Passes ``experiment_cfg_path`` directly to
:func:`auras.experiment.runner.run_experiment`.
"""

from __future__ import annotations

import traceback
import uuid

from auras.launchers.base import BaseLauncher, JobStatus


class LocalLauncher(BaseLauncher):
    """Execute experiments on the local machine.

    Config keys (all optional)
    --------------------------
    ``backend.local.verbose`` : bool
        Print extra information.  Default True.
    """

    def submit(
        self,
        experiment_cfg_path: str,
        *,
        wait: bool = True,
        dry_run: bool = False,
    ) -> JobStatus:
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        verbose = (self._cfg or {}).get("verbose", True) if self._cfg else True

        if dry_run:
            print(f"[LocalLauncher] DRY RUN — would run: {experiment_cfg_path}")
            return JobStatus(job_id=job_id, backend="local", state="pending")

        if verbose:
            print(f"[LocalLauncher] Starting job {job_id}: {experiment_cfg_path}")

        try:
            from auras.experiment.runner import run_experiment
            output_dir = run_experiment(experiment_cfg_path)
            return JobStatus(
                job_id=job_id,
                backend="local",
                state="succeeded",
                output_dir=str(output_dir),
            )
        except Exception as exc:
            err = traceback.format_exc()
            print(f"[LocalLauncher] Job {job_id} FAILED:\n{err}")
            return JobStatus(
                job_id=job_id,
                backend="local",
                state="failed",
                error=str(exc),
            )

    def status(self, job_id: str) -> JobStatus:
        # Local jobs run synchronously in submit(); status is always terminal.
        return JobStatus(job_id=job_id, backend="local", state="succeeded")
