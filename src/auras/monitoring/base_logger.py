"""Abstract logger interface for monitoring backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLogger(ABC):
    """Interface that all monitoring backends must implement."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics."""
        ...

    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log the experiment configuration once at startup."""
        ...

    @abstractmethod
    def finish(self) -> None:
        """Finalize the logging session (flush, close)."""
        ...


class ConsoleLogger(BaseLogger):
    """Simple fallback logger that prints to stdout."""

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        prefix = f"[step {step}] " if step is not None else ""
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        print(prefix + " | ".join(parts))

    def log_config(self, config: Dict[str, Any]) -> None:
        print(f"Config: {config}")

    def finish(self) -> None:
        pass
