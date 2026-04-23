"""AWS SageMaker training-job launcher.

Submits a SageMaker training job via the ``sagemaker`` Python SDK.

How SageMaker custom training jobs work
----------------------------------------
1. Package the entry script + dependencies into a Docker image (or use a
   pre-built PyTorch/MXNet framework image + ``source_dir``).
2. Call ``sagemaker.estimator.Estimator`` with the entry point, S3 data
   channel, instance type, and hyper-parameters.
3. SageMaker copies data from S3, runs the container, and writes
   checkpoints + TensorBoard logs back to S3.

Configuration (``backend.aws`` in your experiment config or
``configs/backends/aws.yaml``)
----------------------------------------------------------------------
::

    region:            eu-west-1
    role_arn:          arn:aws:iam::<account>:role/SageMakerExecutionRole
    s3_bucket:         s3://auras-experiments
    s3_data_prefix:    data/processed/
    instance_type:     ml.g4dn.xlarge   # GPU instance
    image_uri:         ""               # use built-in framework image if empty
    framework:         pytorch           # pytorch | mxnet | huggingface
    framework_version: "2.1"
    py_version:        py310
    max_wait_seconds:  7200

Environment-variable alternatives
-----------------------------------
``AWS_DEFAULT_REGION``, ``SAGEMAKER_ROLE_ARN``, ``SAGEMAKER_S3_BUCKET``
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from auras.launchers.base import BaseLauncher, JobStatus


class AWSLauncher(BaseLauncher):
    """Submit and monitor SageMaker training jobs.

    Parameters
    ----------
    cfg : OmegaConf DictConfig, optional
        ``backend.aws`` config block.
    """

    def _get(self, key: str, env_var: str | None = None, default=None):
        if self._cfg is not None and key in self._cfg:
            return self._cfg[key]
        if env_var and os.environ.get(env_var):
            return os.environ[env_var]
        return default

    def _sagemaker_session(self):
        try:
            import boto3
            import sagemaker
        except ImportError as exc:
            raise ImportError(
                "SageMaker SDK not installed. "
                "Run: pip install sagemaker boto3"
            ) from exc

        region = self._get("region", "AWS_DEFAULT_REGION", "eu-west-1")
        boto_session = boto3.Session(region_name=region)
        return sagemaker.Session(boto_session=boto_session)

    @staticmethod
    def _map_state(sm_status: str) -> str:
        mapping = {
            "InProgress": "running",
            "Completed": "succeeded",
            "Failed": "failed",
            "Stopped": "failed",
            "Stopping": "failed",
        }
        return mapping.get(sm_status, "running")

    def submit(
        self,
        experiment_cfg_path: str,
        *,
        wait: bool = True,
        dry_run: bool = False,
    ) -> JobStatus:
        job_id = f"sagemaker-{uuid.uuid4().hex[:8]}"
        role_arn = self._get("role_arn", "SAGEMAKER_ROLE_ARN")
        s3_bucket = self._get("s3_bucket", "SAGEMAKER_S3_BUCKET", "s3://auras-experiments")
        s3_data = self._get("s3_data_prefix", default="data/processed/")
        instance_type = self._get("instance_type", default="ml.g4dn.xlarge")
        image_uri = self._get("image_uri", default="")
        framework = self._get("framework", default="pytorch")
        framework_version = self._get("framework_version", default="2.1")
        py_version = self._get("py_version", default="py310")

        hyperparameters = {
            "config": experiment_cfg_path,
        }

        payload = {
            "job_id": job_id,
            "backend": "aws_sagemaker",
            "experiment_cfg": experiment_cfg_path,
            "role_arn": role_arn,
            "s3_bucket": s3_bucket,
            "s3_data": s3_data,
            "instance_type": instance_type,
            "hyperparameters": hyperparameters,
        }

        if dry_run:
            print("[AWSLauncher] DRY RUN — submission payload:")
            print(json.dumps(payload, indent=2))
            return JobStatus(job_id=job_id, backend="aws", state="pending")

        if not role_arn:
            raise EnvironmentError(
                "SageMaker IAM role ARN missing.  "
                "Set SAGEMAKER_ROLE_ARN or configure backend.aws.role_arn."
            )

        print(f"[AWSLauncher] Submitting job {job_id} …")
        try:
            import sagemaker
            from sagemaker.estimator import Estimator as SMEstimator

            sm_session = self._sagemaker_session()
            repo_root = str(Path(__file__).parent.parent.parent.parent)

            if image_uri:
                estimator = SMEstimator(
                    image_uri=image_uri,
                    role=role_arn,
                    instance_count=1,
                    instance_type=instance_type,
                    sagemaker_session=sm_session,
                    entry_point="scripts/run_experiment.py",
                    source_dir=repo_root,
                    hyperparameters=hyperparameters,
                    output_path=f"{s3_bucket}/runs/",
                )
            else:
                # Use built-in framework container
                EstimatorCls = _get_framework_estimator(framework)
                estimator = EstimatorCls(
                    entry_point="scripts/run_experiment.py",
                    source_dir=repo_root,
                    role=role_arn,
                    instance_count=1,
                    instance_type=instance_type,
                    framework_version=framework_version,
                    py_version=py_version,
                    sagemaker_session=sm_session,
                    hyperparameters=hyperparameters,
                    output_path=f"{s3_bucket}/runs/",
                )

            estimator.fit(
                inputs={"data": f"{s3_bucket}/{s3_data}"},
                job_name=f"auras-{job_id}",
                wait=wait,
            )

            state = "succeeded" if wait else "running"
            return JobStatus(
                job_id=estimator.latest_training_job.name,
                backend="aws",
                state=state,
                output_dir=f"{s3_bucket}/runs/{job_id}",
            )

        except Exception as exc:
            import traceback
            print(f"[AWSLauncher] Submit failed:\n{traceback.format_exc()}")
            return JobStatus(
                job_id=job_id, backend="aws", state="failed", error=str(exc)
            )

    def status(self, job_id: str) -> JobStatus:
        try:
            import boto3
            region = self._get("region", "AWS_DEFAULT_REGION", "eu-west-1")
            client = boto3.client("sagemaker", region_name=region)
            resp = client.describe_training_job(TrainingJobName=job_id)
            state = self._map_state(resp["TrainingJobStatus"])
            return JobStatus(job_id=job_id, backend="aws", state=state)
        except Exception as exc:
            return JobStatus(job_id=job_id, backend="aws", state="failed", error=str(exc))

    def cancel(self, job_id: str) -> None:
        try:
            import boto3
            region = self._get("region", "AWS_DEFAULT_REGION", "eu-west-1")
            boto3.client("sagemaker", region_name=region).stop_training_job(
                TrainingJobName=job_id
            )
            print(f"[AWSLauncher] Stop requested for job {job_id}")
        except Exception as exc:
            print(f"[AWSLauncher] Cancel failed: {exc}")

    def download_outputs(self, job_id: str, local_dir: str) -> None:
        try:
            import boto3
            s3_bucket = self._get("s3_bucket", "SAGEMAKER_S3_BUCKET")
            s3_key_prefix = f"runs/{job_id}/"
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            s3 = boto3.client("s3")
            bucket_name = s3_bucket.replace("s3://", "").split("/")[0]
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_key_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    local_path = Path(local_dir) / Path(key).relative_to(s3_key_prefix)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket_name, key, str(local_path))
            print(f"[AWSLauncher] Downloaded outputs to {local_dir}")
        except Exception as exc:
            raise RuntimeError(f"S3 download failed: {exc}") from exc


def _get_framework_estimator(framework: str):
    """Return the SageMaker estimator class for the requested framework."""
    framework = framework.lower()
    if framework == "pytorch":
        from sagemaker.pytorch import PyTorch  # type: ignore
        return PyTorch
    elif framework == "mxnet":
        from sagemaker.mxnet import MXNet  # type: ignore
        return MXNet
    elif framework in ("huggingface", "hf"):
        from sagemaker.huggingface import HuggingFace  # type: ignore
        return HuggingFace
    else:
        raise ValueError(f"Unsupported framework '{framework}'. Use: pytorch, mxnet, huggingface")
