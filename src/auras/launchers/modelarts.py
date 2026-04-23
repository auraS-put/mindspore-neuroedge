"""Huawei Cloud ModelArts training-job launcher.

Submits a custom training job via the ModelArts Python SDK
(``modelarts`` package, installed from PyPI or Huawei's pip index).

How ModelArts custom training jobs work
---------------------------------------
1. Upload the project source code and preprocessed data to OBS (Object Storage).
2. Call ``session.estimator.MindSporeEstimator`` (or the generic
   ``Estimator``) with the entry script, OBS paths, Ascend flavour, and
   hyper-parameters passed as environment variables / ``--`` args.
3. ModelArts pulls the container, mounts the OBS data bucket, runs the
   entry script, and writes checkpoints + logs back to OBS.
4. We poll the job until it reaches a terminal state and optionally
   download artefacts.

Configuration (``backend.modelarts`` in your experiment config or
``configs/backends/modelarts.yaml``)
----------------------------------------------------------------------
::

    endpoint:          https://modelarts.cn-north-4.myhuaweicloud.com
    region:            cn-north-4
    project_id:        <your Huawei Cloud project ID>
    ak:                <your Access Key>   # or set env MA_AK
    sk:                <your Secret Key>   # or set env MA_SK
    obs_bucket:        obs://auras-experiments
    obs_data_path:     obs://auras-experiments/data/processed/
    train_instance:    modelarts.vm.cpu.8u     # or ascend-snt9b for Ascend
    train_image_url:   ""                  # leave empty → use built-in MindSpore image
    log_level:         INFO
    max_wait_seconds:  7200                # 2 h timeout

Environment-variable alternatives (preferred for CI/CD)
---------------------------------------------------------
``MA_AK``, ``MA_SK``, ``MA_PROJECT_ID``, ``MA_ENDPOINT``, ``MA_REGION``
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from auras.launchers.base import BaseLauncher, JobStatus


class ModelArtsLauncher(BaseLauncher):
    """Submit and monitor ModelArts custom training jobs.

    Parameters
    ----------
    cfg : OmegaConf DictConfig, optional
        ``backend.modelarts`` config block.  Missing keys fall back to
        environment variables (see module docstring).
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, key: str, env_var: Optional[str] = None, default=None):
        """Read from cfg → env → default in that priority order."""
        if self._cfg is not None and key in self._cfg:
            return self._cfg[key]
        if env_var and os.environ.get(env_var):
            return os.environ[env_var]
        return default

    def _session(self):
        """Build and return an authenticated ModelArts Session."""
        try:
            from modelarts.session import Session
        except ImportError as exc:
            raise ImportError(
                "ModelArts SDK not installed. "
                "Run: pip install modelarts"
            ) from exc

        ak = self._get("ak", "MA_AK")
        sk = self._get("sk", "MA_SK")
        project_id = self._get("project_id", "MA_PROJECT_ID")
        endpoint = self._get("endpoint", "MA_ENDPOINT",
                             "https://modelarts.cn-north-4.myhuaweicloud.com")
        region = self._get("region", "MA_REGION", "cn-north-4")

        if not ak or not sk or not project_id:
            raise EnvironmentError(
                "ModelArts credentials missing.  "
                "Set MA_AK / MA_SK / MA_PROJECT_ID environment variables "
                "or configure backend.modelarts in your YAML."
            )

        return Session(
            access_key=ak,
            secret_key=sk,
            project_id=project_id,
            endpoint_url=endpoint,
            region_name=region,
        )

    def _sync_source_to_obs(self, obs_bucket: str) -> str:
        """Rsync project source to OBS so ModelArts can mount it.

        Returns the OBS path of the uploaded code directory.
        """
        try:
            from modelarts.session import Session
            import obsutil  # type: ignore
        except ImportError:
            pass

        # Use OBS CLI (obsutil) if available; fall back to SDK upload.
        src_root = Path(__file__).parent.parent.parent.parent  # repo root
        obs_code_path = f"{obs_bucket.rstrip('/')}/code/"

        print(f"[ModelArtsLauncher] Syncing source to {obs_code_path} …")
        ret = os.system(
            f"obsutil sync {src_root} {obs_code_path} "
            f"--exclude '.venv/*' --exclude '*.pyc' --exclude '__pycache__/*' "
            f"--exclude '.git/*' --exclude 'data/raw/*'"
        )
        if ret != 0:
            raise RuntimeError(
                "obsutil sync failed.  "
                "Ensure obsutil is installed and configured: "
                "https://support.huaweicloud.com/intl/en-us/utiltg-obs/obs_11_0001.html"
            )
        return obs_code_path

    @staticmethod
    def _map_state(ma_status: str) -> str:
        """Normalise ModelArts job status strings to our unified vocabulary."""
        mapping = {
            "Creating": "pending",
            "Pending": "pending",
            "Running": "running",
            "Succeeded": "succeeded",
            "Failed": "failed",
            "Terminated": "failed",
        }
        return mapping.get(ma_status, "running")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(
        self,
        experiment_cfg_path: str,
        *,
        wait: bool = True,
        dry_run: bool = False,
    ) -> JobStatus:
        """Submit the experiment as a ModelArts custom training job.

        The entry script ``scripts/run_experiment.py`` is called inside
        the container with ``--config <experiment_cfg_path>`` injected as
        a training hyperparameter so ModelArts passes it to the script.
        """
        job_id = f"ma-{uuid.uuid4().hex[:8]}"
        obs_bucket = self._get("obs_bucket", "MA_OBS_BUCKET", "obs://auras-experiments")
        obs_data_path = self._get("obs_data_path", default=f"{obs_bucket}/data/processed/")
        obs_output_path = self._get("obs_output_path", default=f"{obs_bucket}/runs/")
        flavour = self._get("train_instance", default="modelarts.vm.cpu.8u")
        image_url = self._get("train_image_url", default="")

        payload = {
            "job_id": job_id,
            "backend": "modelarts",
            "experiment_cfg": experiment_cfg_path,
            "obs_bucket": obs_bucket,
            "obs_data_path": obs_data_path,
            "obs_output_path": obs_output_path,
            "train_instance": flavour,
            "image_url": image_url,
        }

        if dry_run:
            print("[ModelArtsLauncher] DRY RUN — submission payload:")
            print(json.dumps(payload, indent=2))
            return JobStatus(job_id=job_id, backend="modelarts", state="pending")

        print(f"[ModelArtsLauncher] Submitting job {job_id} …")

        try:
            session = self._session()

            # Sync source code to OBS
            obs_code_path = self._sync_source_to_obs(obs_bucket)

            from modelarts.estimatorV2 import Estimator  # type: ignore

            estimator = Estimator(
                session=session,
                training_files=obs_code_path,
                outputs=[{"name": "trained_model", "obs_path": obs_output_path}],
                job_description=f"auraS experiment: {experiment_cfg_path}",
                framework_type="MindSpore",
                framework_version="2.8",
                train_instance_type=flavour,
                train_instance_count=1,
                user_image_url=image_url or None,
                # Entry script relative to obs_code_path
                entry_command=(
                    f"python scripts/run_experiment.py "
                    f"--config {experiment_cfg_path} "
                    f"--data-dir ${{MA_INPUTS_0_URL}} "
                    f"--output-dir ${{MA_OUTPUTS_0_URL}}"
                ),
                log_level=self._get("log_level", default="INFO"),
            )

            job = estimator.fit(
                inputs=[{"name": "data", "obs_path": obs_data_path}],
                job_name=f"auras-{job_id}",
                wait=False,
            )
            ma_job_id = job.job_id
            print(f"[ModelArtsLauncher] Job submitted: {ma_job_id}")

            if not wait:
                return JobStatus(
                    job_id=ma_job_id,
                    backend="modelarts",
                    state="running",
                    output_dir=obs_output_path,
                )

            return self._wait_for_completion(ma_job_id, job, obs_output_path)

        except Exception as exc:
            import traceback
            print(f"[ModelArtsLauncher] Submit failed:\n{traceback.format_exc()}")
            return JobStatus(
                job_id=job_id,
                backend="modelarts",
                state="failed",
                error=str(exc),
            )

    def _wait_for_completion(self, job_id: str, job, output_dir: str) -> JobStatus:
        """Poll until the job reaches a terminal state."""
        max_wait = self._get("max_wait_seconds", default=7200)
        poll_interval = 30
        elapsed = 0

        print(f"[ModelArtsLauncher] Waiting for job {job_id} (max {max_wait}s) …")
        while elapsed < max_wait:
            try:
                state_raw = job.get_status()
                state = self._map_state(state_raw)
                print(f"  [{elapsed}s] {state_raw}")
                if state in ("succeeded", "failed"):
                    return JobStatus(
                        job_id=job_id,
                        backend="modelarts",
                        state=state,
                        output_dir=output_dir,
                    )
            except Exception as exc:
                print(f"  [warn] status poll failed: {exc}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        return JobStatus(
            job_id=job_id,
            backend="modelarts",
            state="failed",
            error=f"Timed out after {max_wait}s",
        )

    def status(self, job_id: str) -> JobStatus:
        try:
            session = self._session()
            from modelarts.estimatorV2 import Estimator  # type: ignore
            job = Estimator.get_job(session, job_id)
            state = self._map_state(job.get_status())
            return JobStatus(job_id=job_id, backend="modelarts", state=state)
        except Exception as exc:
            return JobStatus(
                job_id=job_id, backend="modelarts", state="failed", error=str(exc)
            )

    def cancel(self, job_id: str) -> None:
        try:
            session = self._session()
            from modelarts.estimatorV2 import Estimator  # type: ignore
            job = Estimator.get_job(session, job_id)
            job.stop()
            print(f"[ModelArtsLauncher] Cancelled job {job_id}")
        except Exception as exc:
            print(f"[ModelArtsLauncher] Cancel failed: {exc}")

    def download_outputs(self, job_id: str, local_dir: str) -> None:
        """Download OBS output artefacts to *local_dir* via obsutil."""
        obs_output = self._get("obs_output_path", default="obs://auras-experiments/runs/")
        job_obs_path = f"{obs_output.rstrip('/')}/{job_id}/"
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        ret = os.system(f"obsutil cp {job_obs_path} {local_dir} -r -f")
        if ret != 0:
            raise RuntimeError(f"obsutil cp failed from {job_obs_path} → {local_dir}")
        print(f"[ModelArtsLauncher] Downloaded outputs to {local_dir}")
