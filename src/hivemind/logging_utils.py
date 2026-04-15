"""Logging utilities for training metrics."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLogger(ABC):
    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class NullLogger(BaseLogger):
    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class JSONLogger(BaseLogger):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        entry = {"step": step, **metrics}
        self._entries.append(entry)

    def close(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)


def init_logger(cfg: dict[str, Any]) -> BaseLogger:
    """Initialize a logger from config."""
    if not cfg.get("enabled", False):
        return NullLogger()
    backend = cfg.get("backend", "json")
    if backend == "json":
        return JSONLogger(cfg.get("path", "logs/metrics.json"))
    if backend == "wandb":
        try:
            import wandb

            wandb.init(project=cfg.get("project", "hivemind"), config=cfg)
            return _WandBLogger()
        except ImportError:
            return JSONLogger(cfg.get("path", "logs/metrics.json"))
    return NullLogger()


class _WandBLogger(BaseLogger):
    def log(self, metrics: dict[str, Any], step: int) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def close(self) -> None:
        import wandb

        wandb.finish()
