"""AWS SageMaker training service implementation."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import time

from scripts.cloud.base import (
    JobConfig,
    JobPhase,
    JobStatus,
    ResourceMetrics,
    TrainingService,
)


class SageMakerTraining(TrainingService):
    """AWS SageMaker training job management."""

    def __init__(self, region: str, role_arn: str, image_uri: str, **kwargs):
        import boto3

        self._region = region
        self._role_arn = role_arn
        self._image_uri = image_uri  # Custom ECR image with MindSpore

        session = boto3.Session(
            aws_access_key_id=kwargs.get("aws_access_key_id"),
            aws_secret_access_key=kwargs.get("aws_secret_access_key"),
            region_name=region,
        )
        self._sm = session.client("sagemaker")

    # SageMaker flavor → instance type mapping
    FLAVOR_MAP = {
        "ml.p3.2xlarge": "ml.p3.2xlarge",       # 1x V100 16GB
        "ml.p3.8xlarge": "ml.p3.8xlarge",       # 4x V100 16GB
        "ml.g4dn.xlarge": "ml.g4dn.xlarge",     # 1x T4 16GB
        "ml.g5.xlarge": "ml.g5.xlarge",         # 1x A10G 24GB
        # Allow pass-through for direct instance types
    }

    def submit_job(self, config: JobConfig) -> str:
        job_name = f"{config.name}-{int(time.time())}"
        # SageMaker job names: alphanumeric + hyphens only, max 63 chars
        job_name = job_name.replace("_", "-")[:63]

        instance_type = self.FLAVOR_MAP.get(config.flavor, config.flavor)

        # Parse S3 paths from config
        # data_path format: s3://bucket/prefix/
        # output_path format: s3://bucket/prefix/
        # Filter out internal keys from hyperparameters
        hyperparameters = {
            k: str(v) for k, v in config.env_vars.items()
            if not k.startswith("__")
        }

        # Derive code S3 URI from data_path (sibling directory)
        # e.g., s3://bucket/data/ → s3://bucket/code/
        code_s3_uri = config.data_path.rsplit("data/", 1)[0] + "code/"

        training_params = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": self._image_uri,
                "TrainingInputMode": "File",
            },
            "RoleArn": self._role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": config.data_path,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
                {
                    "ChannelName": "code",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": code_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
            ],
            "OutputDataConfig": {"S3OutputPath": config.output_path},
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 50,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": config.max_runtime_s},
            "HyperParameters": hyperparameters,
        }

        self._sm.create_training_job(**training_params)
        return job_name  # SageMaker uses job name as ID

    def get_status(self, job_id: str) -> JobStatus:
        resp = self._sm.describe_training_job(TrainingJobName=job_id)

        sm_status = resp["TrainingJobStatus"]
        phase_map = {
            "InProgress": JobPhase.RUNNING,
            "Completed": JobPhase.COMPLETED,
            "Failed": JobPhase.FAILED,
            "Stopping": JobPhase.TERMINATED,
            "Stopped": JobPhase.TERMINATED,
        }
        phase = phase_map.get(sm_status, JobPhase.UNKNOWN)

        # Duration
        start = resp.get("TrainingStartTime")
        end = resp.get("TrainingEndTime")
        duration = 0
        if start:
            end_time = end or __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            )
            duration = int((end_time - start).total_seconds())

        # Resource info
        resource = resp.get("ResourceConfig", {})
        instance_type = resource.get("InstanceType", "")
        instance_count = resource.get("InstanceCount", 1)
        flavor_detail = f"{instance_count}x {instance_type}"

        # Metrics from CloudWatch (basic)
        metrics = None
        metric_list = resp.get("FinalMetricDataList", [])
        if metric_list:
            metrics = ResourceMetrics()
            for m in metric_list:
                name = m["MetricName"]
                val = m["Value"]
                if "gpu" in name.lower() and "util" in name.lower():
                    metrics.gpu_util_avg = val

        return JobStatus(
            job_id=job_id,
            phase=phase,
            duration_s=duration,
            flavor=instance_type,
            flavor_detail=flavor_detail,
            metrics=metrics,
            events=[],
        )

    def terminate(self, job_id: str) -> None:
        self._sm.stop_training_job(TrainingJobName=job_id)

    def list_jobs(self, limit: int = 10) -> list[JobStatus]:
        resp = self._sm.list_training_jobs(
            MaxResults=limit,
            SortBy="CreationTime",
            SortOrder="Descending",
        )

        results = []
        for j in resp.get("TrainingJobSummaries", []):
            name = j["TrainingJobName"]
            sm_status = j["TrainingJobStatus"]
            phase_map = {
                "InProgress": JobPhase.RUNNING,
                "Completed": JobPhase.COMPLETED,
                "Failed": JobPhase.FAILED,
                "Stopping": JobPhase.TERMINATED,
                "Stopped": JobPhase.TERMINATED,
            }
            phase = phase_map.get(sm_status, JobPhase.UNKNOWN)

            start = j.get("TrainingStartTime")
            end = j.get("TrainingEndTime")
            duration = 0
            if start and end:
                duration = int((end - start).total_seconds())

            results.append(
                JobStatus(job_id=name, phase=phase, duration_s=duration, flavor=name)
            )
        return results
