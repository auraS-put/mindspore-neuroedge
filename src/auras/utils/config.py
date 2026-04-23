"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file with OmegaConf."""
    return OmegaConf.load(str(path))


def merge_configs(*cfgs: DictConfig) -> DictConfig:
    """Deep-merge multiple OmegaConf configs (later overrides earlier)."""
    return OmegaConf.merge(*cfgs)


def to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Convert config to a plain dict (for JSON serialization / WandB)."""
    return OmegaConf.to_container(cfg, resolve=True)
