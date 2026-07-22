"""Hydra config composition helpers."""

from __future__ import annotations

from omegaconf import DictConfig


def unwrap_config(cfg: DictConfig) -> DictConfig:
    """Hydra can wrap grouped configs under a group name; unwrap to top-level."""
    if "model" in cfg:
        return cfg
    return cfg
