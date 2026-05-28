"""Huawei ModelArts training service implementation."""

from __future__ import annotations

import json
import tarfile
import tempfile
import time
from datetime import datetime, timezone

import requests

from scripts.cloud.base import (
    JobConfig,
    JobPhase,
    JobStatus,
    ResourceMetrics,
    TrainingService,
)


class ModelArtsTraining(TrainingService):
    """Huawei ModelArts training job management."""

    def __init__(self, ak: str, sk: str, region: str, project_id: str):
        self._ak = ak
        self._sk = sk
        self._region = region
        self._project_id = project_id
        self._base_url = f"https://modelarts.{region}.myhuaweicloud.com"
        self._token: str | None = None
        self._token_expires: float = 0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires:
            return self._token

        iam_url = f"https://iam.{self._region}.myhuaweicloud.com/v3/auth/tokens"
        payload = {
            "auth": {
                "identity": {
                    "methods": ["hw_ak_sk"],
                    "hw_ak_sk": {
                        "access": {"key": self._ak},
                        "secret": {"key": self._sk},
                    },
                },
                "scope": {"project": {"id": self._project_id}},
            }
        }
        resp = requests.post(iam_url, json=payload, timeout=15)
        resp.raise_for_status()
        self._token = resp.headers["X-Subject-Token"]
        self._token_expires = time.time() + 3500  # ~1h validity
        return self._token

    @property
    def _headers(self) -> dict:
        return {
            "X-Auth-Token": self._get_token(),
            "Content-Type": "application/json",
        }

    def submit_job(self, config: JobConfig) -> str:
        job_name = f"{config.name}-{int(time.time())}"

        algorithm = {
            "code_dir": config.env_vars.pop("__code_dir", f"/{config.output_path.split('/')[1]}/code/"),
            "boot_file": config.env_vars.pop("__boot_file", f"/{config.output_path.split('/')[1]}/code/boot.py"),
            "engine": {
                "engine_id": "mindspore_1.3.0-cuda_10.1-py_3.7-ubuntu_1804-x86_64"
            },
            "inputs": [
                {"name": "data", "remote": {"obs": {"obs_url": config.data_path}}}
            ],
            "outputs": [
                {"name": "output", "remote": {"obs": {"obs_url": config.output_path}}}
            ],
        }
        if config.env_vars:
            algorithm["environments"] = config.env_vars

        job_body = {
            "kind": "job",
            "metadata": {"name": job_name, "description": "Training job"},
            "algorithm": algorithm,
            "spec": {
                "resource": {"flavor_id": config.flavor, "node_count": 1}
            },
        }

        url = f"{self._base_url}/v2/{self._project_id}/training-jobs"
        r = requests.post(url, headers=self._headers, json=job_body, timeout=30)

        if r.status_code != 201:
            raise RuntimeError(
                f"Job submission failed ({r.status_code}): {r.text[:500]}"
            )

        result = r.json()
        return result["metadata"]["id"]

    def get_status(self, job_id: str) -> JobStatus:
        url = f"{self._base_url}/v2/{self._project_id}/training-jobs/{job_id}"
        r = requests.get(url, headers=self._headers, timeout=30)
        r.raise_for_status()
        result = r.json()

        status = result["status"]
        phase_str = status.get("phase", "Unknown")
        try:
            phase = JobPhase(phase_str)
        except ValueError:
            phase = JobPhase.UNKNOWN

        duration = status.get("duration", 0) // 1000
        spec = result.get("spec", {}).get("resource", {})
        flavor = spec.get("flavor_id", "")

        # Build flavor detail string
        flavor_info = spec.get("flavor_detail", {}).get("flavor_info", {})
        detail = ""
        if flavor_info:
            cpu = flavor_info.get("cpu", {})
            mem = flavor_info.get("memory", {})
            gpu = flavor_info.get("gpu", {})
            disk = flavor_info.get("disk", {})
            detail = (
                f"CPU: {cpu.get('core_num', '?')} cores | "
                f"RAM: {mem.get('size', '?')} GiB | "
                f"GPU: {gpu.get('unit_num', '0')}x {gpu.get('product_name', '')} "
                f"{gpu.get('memory', '')} | Disk: {disk.get('size', '?')} GB"
            )

        # Metrics
        metrics = None
        metrics_data = status.get("metrics_statistics")
        if metrics_data:
            cpu_m = metrics_data.get("cpu_usage", {})
            mem_m = metrics_data.get("mem_usage", {})
            gpu_m = metrics_data.get("gpu", {})
            metrics = ResourceMetrics(
                cpu_avg=cpu_m.get("average", -1),
                cpu_max=cpu_m.get("max", -1),
                ram_avg=mem_m.get("average", -1),
                ram_max=mem_m.get("max", -1),
                gpu_util_avg=gpu_m.get("util", {}).get("average", -1),
                gpu_util_max=gpu_m.get("util", {}).get("max", -1),
                gpu_mem_avg=gpu_m.get("mem_usage", {}).get("average", -1),
                gpu_mem_max=gpu_m.get("mem_usage", {}).get("max", -1),
            )

        # Events
        events = self._get_events(job_id)

        return JobStatus(
            job_id=job_id,
            phase=phase,
            duration_s=duration,
            flavor=flavor,
            flavor_detail=detail,
            metrics=metrics,
            events=events,
        )

    def _get_events(self, job_id: str) -> list[dict]:
        url = f"{self._base_url}/v2/{self._project_id}/training-jobs/{job_id}/events"
        try:
            r = requests.get(
                url, headers=self._headers,
                params={"order": "asc", "limit": 50},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("events", [])
        except Exception:
            pass
        return []

    def terminate(self, job_id: str) -> None:
        url = f"{self._base_url}/v2/{self._project_id}/training-jobs/{job_id}/actions"
        r = requests.post(
            url, headers=self._headers,
            json={"action_type": "terminate"},
            timeout=15,
        )
        if r.status_code not in (200, 202):
            raise RuntimeError(f"Terminate failed: {r.status_code} {r.text[:200]}")

    def list_jobs(self, limit: int = 10) -> list[JobStatus]:
        url = f"{self._base_url}/v2/{self._project_id}/training-jobs"
        try:
            r = requests.get(
                url, headers=self._headers,
                params={"limit": limit, "sort_by": "create_time", "order": "desc"},
                timeout=30,
            )
            if r.status_code != 200:
                # Some regions don't support the list endpoint
                return []
            jobs = r.json().get("items", [])
        except Exception:
            return []

        results = []
        for j in jobs:
            jid = j["metadata"]["id"]
            name = j["metadata"]["name"]
            phase_str = j["status"].get("phase", "Unknown")
            try:
                phase = JobPhase(phase_str)
            except ValueError:
                phase = JobPhase.UNKNOWN
            duration = j["status"].get("duration", 0) // 1000
            results.append(
                JobStatus(
                    job_id=jid,
                    phase=phase,
                    duration_s=duration,
                    flavor=name,  # reuse flavor field for name in list view
                )
            )
        return results
