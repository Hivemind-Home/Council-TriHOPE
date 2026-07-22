"""Logging utilities for training metrics.

Two backends:

* ``JSONLogger`` — newline-delimited JSON written to disk one line per
  ``log()`` call. Crash-safe: flushes after every step so an OOM/SIGKILL
  loses at most one record. The file is appended to on resume so the
  full training history accumulates across restarts.
* ``_WandBLogger`` — thin wrapper around ``wandb.log``. Resume is
  controlled via the ``id`` field in the config: pass the same id you
  used originally and the new run continues the wandb metrics stream.
"""

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
    """Newline-delimited JSON logger.

    Each ``log()`` call writes one line and flushes the OS buffer. A
    legacy ``.json`` extension is accepted; the format on disk is JSONL
    regardless. The path is opened in append mode so resuming a run
    keeps the prior step records instead of overwriting them.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open append-binary; we'll write our own newlines as JSONL.
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        entry = {"step": step, **metrics}
        self._fh.write(json.dumps(entry, default=str) + "\n")
        # Flush to OS buffers every step. Crash-safety > write throughput.
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


def init_logger(cfg: dict[str, Any]) -> BaseLogger:
    """Initialize a logger from config.

    Recognised keys:
      enabled: bool — turn logging on/off
      backend: "json" | "wandb" | "null"
      path:    JSONLogger output (default ``logs/metrics.json``)
      project: wandb project name (default ``hivemind``)
      run_name: wandb run display name (optional)
      id:      wandb run id; pass the same id to resume an existing run
      resume:  wandb resume mode — "allow" | "must" | "never" (default "allow")
    """
    if not cfg.get("enabled", False):
        return NullLogger()
    backend = cfg.get("backend", "json")
    if backend == "json":
        return JSONLogger(cfg.get("path", "logs/metrics.json"))
    if backend == "wandb":
        try:
            import wandb
        except ImportError:
            # Fall back to JSON if wandb isn't installed in this env.
            return JSONLogger(cfg.get("path", "logs/metrics.json"))
        wandb.init(
            project=cfg.get("project", "hivemind"),
            name=cfg.get("run_name"),
            id=cfg.get("id"),
            resume=cfg.get("resume", "allow"),
            config=cfg,
        )
        return _WandBLogger()
    return NullLogger()


class _WandBLogger(BaseLogger):
    def log(self, metrics: dict[str, Any], step: int) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def close(self) -> None:
        import wandb

        wandb.finish()
