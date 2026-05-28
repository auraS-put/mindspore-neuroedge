"""AWS S3 object storage implementation."""

from __future__ import annotations

import os

from scripts.cloud.base import ObjectStore


class S3Store(ObjectStore):
    """AWS S3 object storage client."""

    def __init__(self, bucket: str, region: str, **kwargs):
        import boto3

        self._bucket = bucket
        session = boto3.Session(
            aws_access_key_id=kwargs.get("aws_access_key_id"),
            aws_secret_access_key=kwargs.get("aws_secret_access_key"),
            region_name=region,
        )
        self._s3 = session.client("s3")

    def upload_file(self, local_path: str, remote_key: str) -> None:
        self._s3.upload_file(local_path, self._bucket, remote_key)

    def upload_bytes(self, data: bytes, remote_key: str) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=remote_key, Body=data)

    def download_file(self, remote_key: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self._s3.download_file(self._bucket, remote_key, local_path)

    def get_bytes(self, remote_key: str) -> bytes:
        resp = self._s3.get_object(Bucket=self._bucket, Key=remote_key)
        return resp["Body"].read()

    def list_objects(self, prefix: str, delimiter: str = "") -> list[str]:
        params = {"Bucket": self._bucket, "Prefix": prefix, "MaxKeys": 1000}
        if delimiter:
            params["Delimiter"] = delimiter
        resp = self._s3.list_objects_v2(**params)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def list_prefixes(self, prefix: str, delimiter: str = "/") -> list[str]:
        resp = self._s3.list_objects_v2(
            Bucket=self._bucket, Prefix=prefix, Delimiter=delimiter
        )
        return [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]

    def head(self, remote_key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=remote_key)
            return True
        except self._s3.exceptions.ClientError:
            return False

    def close(self) -> None:
        pass  # boto3 client doesn't need explicit close
