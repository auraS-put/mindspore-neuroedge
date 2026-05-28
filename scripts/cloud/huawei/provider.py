"""Huawei Cloud provider — wires together OBS + ModelArts."""

from __future__ import annotations

import os

from scripts.cloud.base import CloudProvider, ObjectStore, TrainingService
from scripts.cloud.huawei.modelarts import ModelArtsTraining
from scripts.cloud.huawei.obs import OBSStore
from scripts.cloud.registry import register_provider


@register_provider("huawei")
class HuaweiProvider(CloudProvider):
    """Huawei Cloud: OBS for storage, ModelArts for training."""

    def __init__(self, **kwargs):
        self._ak = kwargs.get("ak") or os.environ["HUAWEI_AK"]
        self._sk = kwargs.get("sk") or os.environ["HUAWEI_SK"]
        self._region = kwargs.get("region") or os.environ["HUAWEI_REGION"]
        self._project_id = kwargs.get("project_id") or os.environ["MODELARTS_PROJECT_ID"]
        self._bucket = kwargs.get("bucket") or os.environ.get("HUAWEI_OBS_BUCKET", "auras-experiments")

        self._storage: OBSStore | None = None
        self._training: ModelArtsTraining | None = None

    @property
    def name(self) -> str:
        return "huawei"

    def storage(self) -> ObjectStore:
        if self._storage is None:
            self._storage = OBSStore(self._ak, self._sk, self._region, self._bucket)
        return self._storage

    def training(self) -> TrainingService:
        if self._training is None:
            self._training = ModelArtsTraining(
                self._ak, self._sk, self._region, self._project_id
            )
        return self._training

    def close(self) -> None:
        if self._storage:
            self._storage.close()
