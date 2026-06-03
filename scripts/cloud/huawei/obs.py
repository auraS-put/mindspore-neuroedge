"""Huawei OBS (Object Storage Service) implementation."""

from __future__ import annotations

import os

from scripts.cloud.base import ObjectStore


class OBSStore(ObjectStore):
    """Huawei OBS object storage client."""

    def __init__(self, ak: str, sk: str, region: str, bucket: str):
        from obs import ObsClient

        self._bucket = bucket
        self._client = ObsClient(
            access_key_id=ak,
            secret_access_key=sk,
            server=f"https://obs.{region}.myhuaweicloud.com",
        )

    def upload_file(self, local_path: str, remote_key: str) -> None:
        resp = self._client.putFile(self._bucket, remote_key, local_path)
        if resp.status >= 300:
            raise RuntimeError(f"OBS upload failed: {resp.status} {resp.reason}")

    def upload_bytes(self, data: bytes, remote_key: str) -> None:
        resp = self._client.putContent(self._bucket, remote_key, content=data)
        if resp.status >= 300:
            raise RuntimeError(f"OBS upload failed: {resp.status} {resp.reason}")

    def download_file(self, remote_key: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        resp = self._client.getObject(
            self._bucket, remote_key, loadStreamInMemory=True
        )
        if resp.status != 200:
            raise RuntimeError(
                f"OBS download failed: {resp.status} {getattr(resp, 'errorMessage', '')}"
            )
        with open(local_path, "wb") as f:
            f.write(resp.body.buffer)

    def get_bytes(self, remote_key: str) -> bytes:
        resp = self._client.getObject(
            self._bucket, remote_key, loadStreamInMemory=True
        )
        if resp.status != 200:
            raise RuntimeError(
                f"OBS get failed: {resp.status} {getattr(resp, 'errorMessage', '')}"
            )
        return resp.body.buffer

    def list_objects(self, prefix: str, delimiter: str = "") -> list[str]:
        resp = self._client.listObjects(
            self._bucket, prefix=prefix, delimiter=delimiter, max_keys=1000
        )
        if resp.status != 200:
            return []
        return [obj.key for obj in (resp.body.contents or [])]

    def list_prefixes(self, prefix: str, delimiter: str = "/") -> list[str]:
        resp = self._client.listObjects(
            self._bucket, prefix=prefix, delimiter=delimiter
        )
        if resp.status != 200:
            return []
        return [cp.prefix for cp in (resp.body.commonPrefixs or [])]

    def head(self, remote_key: str) -> bool:
        resp = self._client.getObjectMetadata(self._bucket, remote_key)
        return resp.status == 200

    def close(self) -> None:
        self._client.close()
