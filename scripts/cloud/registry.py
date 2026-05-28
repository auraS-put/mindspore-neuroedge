"""Provider registry — resolves provider name to implementation."""

from __future__ import annotations

import os
from typing import Optional

from scripts.cloud.base import CloudProvider


_PROVIDERS: dict[str, type[CloudProvider]] = {}


def register_provider(name: str):
    """Decorator to register a CloudProvider implementation."""
    def decorator(cls: type[CloudProvider]):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def get_provider(name: Optional[str] = None, **kwargs) -> CloudProvider:
    """Get a cloud provider instance by name.

    If name is None, reads from CLOUD_PROVIDER env var (default: 'huawei').
    """
    if name is None:
        name = os.environ.get("CLOUD_PROVIDER", "huawei")

    # Lazy-import providers to register them
    if not _PROVIDERS:
        import scripts.cloud.huawei  # noqa: F401
        import scripts.cloud.aws  # noqa: F401

    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    return _PROVIDERS[name](**kwargs)
