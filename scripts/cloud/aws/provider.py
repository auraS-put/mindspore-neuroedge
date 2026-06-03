"""AWS Cloud provider — wires together S3 + SageMaker."""

from __future__ import annotations

import os

from scripts.cloud.base import CloudProvider, ObjectStore, TrainingService
from scripts.cloud.aws.s3 import S3Store
from scripts.cloud.aws.sagemaker import SageMakerTraining
from scripts.cloud.registry import register_provider


@register_provider("aws")
class AWSProvider(CloudProvider):
    """AWS: S3 for storage, SageMaker for training."""

    def __init__(self, **kwargs):
        self._region = kwargs.get("region") or os.environ.get("AWS_REGION", "eu-west-1")
        self._bucket = kwargs.get("bucket") or os.environ["AWS_S3_BUCKET"]
        self._role_arn = kwargs.get("role_arn") or os.environ["SAGEMAKER_ROLE_ARN"]
        self._image_uri = kwargs.get("image_uri") or os.environ["SAGEMAKER_IMAGE_URI"]
        self._aws_access_key_id = kwargs.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
        self._aws_secret_access_key = kwargs.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")

        self._storage: S3Store | None = None
        self._training: SageMakerTraining | None = None

    @property
    def name(self) -> str:
        return "aws"

    def storage(self) -> ObjectStore:
        if self._storage is None:
            self._storage = S3Store(
                bucket=self._bucket,
                region=self._region,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
            )
        return self._storage

    def training(self) -> TrainingService:
        if self._training is None:
            self._training = SageMakerTraining(
                region=self._region,
                role_arn=self._role_arn,
                image_uri=self._image_uri,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
            )
        return self._training

    def close(self) -> None:
        if self._storage:
            self._storage.close()
