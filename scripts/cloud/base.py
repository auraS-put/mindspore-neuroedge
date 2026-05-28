"""Abstract base classes for cloud provider abstraction.

All cloud-specific implementations (Huawei ModelArts, AWS SageMaker, etc.)
must implement these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobPhase(str, Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    TERMINATED = "Terminated"
    UNKNOWN = "Unknown"


@dataclass
class ResourceMetrics:
    cpu_avg: float = -1.0
    cpu_max: float = -1.0
    ram_avg: float = -1.0
    ram_max: float = -1.0
    gpu_util_avg: float = -1.0
    gpu_util_max: float = -1.0
    gpu_mem_avg: float = -1.0
    gpu_mem_max: float = -1.0


@dataclass
class JobStatus:
    job_id: str
    phase: JobPhase
    duration_s: int = 0
    flavor: str = ""
    flavor_detail: str = ""
    metrics: Optional[ResourceMetrics] = None
    events: list[dict] = field(default_factory=list)


@dataclass
class JobConfig:
    """Configuration for submitting a training job."""
    name: str
    boot_script: str  # local path to the boot script
    code_paths: list[str]  # local paths to include in code tarball
    data_path: str  # remote storage path to training data
    output_path: str  # remote storage path for outputs
    flavor: str  # instance type identifier
    env_vars: dict[str, str] = field(default_factory=dict)
    framework: str = "mindspore"  # framework identifier


class ObjectStore(ABC):
    """Abstract interface for cloud object storage (OBS, S3, etc.)."""

    @abstractmethod
    def upload_file(self, local_path: str, remote_key: str) -> None:
        """Upload a local file to remote storage."""

    @abstractmethod
    def upload_bytes(self, data: bytes, remote_key: str) -> None:
        """Upload bytes to remote storage."""

    @abstractmethod
    def download_file(self, remote_key: str, local_path: str) -> None:
        """Download a remote object to a local file."""

    @abstractmethod
    def get_bytes(self, remote_key: str) -> bytes:
        """Get the contents of a remote object as bytes."""

    @abstractmethod
    def list_objects(self, prefix: str, delimiter: str = "") -> list[str]:
        """List object keys matching a prefix."""

    @abstractmethod
    def list_prefixes(self, prefix: str, delimiter: str = "/") -> list[str]:
        """List common prefixes (subdirectories) under a prefix."""

    @abstractmethod
    def head(self, remote_key: str) -> bool:
        """Check if an object exists."""

    def close(self) -> None:
        """Release resources (optional)."""


class TrainingService(ABC):
    """Abstract interface for managed training job services."""

    @abstractmethod
    def submit_job(self, config: JobConfig) -> str:
        """Submit a training job. Returns job_id."""

    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        """Get the current status of a training job."""

    @abstractmethod
    def terminate(self, job_id: str) -> None:
        """Terminate a running training job."""

    @abstractmethod
    def list_jobs(self, limit: int = 10) -> list[JobStatus]:
        """List recent training jobs."""


class CloudProvider(ABC):
    """Factory that provides storage and training service for a cloud."""

    @abstractmethod
    def storage(self) -> ObjectStore:
        """Get the object storage client."""

    @abstractmethod
    def training(self) -> TrainingService:
        """Get the training service client."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'huawei', 'aws')."""

    def close(self) -> None:
        """Release all resources."""
        self.storage().close()
