"""Checkpoint save/load for the training loop.

Persists everything needed to resume training bit-exactly:

* Student weights (base + LoRA — both live in the ``nn.Module`` state_dict;
  we separate them into two files so consolidation audits can inspect P
  vs F without loading the whole model)
* Optimizer state (Adam m, v, step)
* Controller state (per-module stability EMA, repetition buckets,
  retrieval hit buffer)
* R-store contents (deque of retrieval entries)
* RNG state (python, numpy, torch CPU + CUDA)
* Step counter + training metadata

Directory layout (atomically written via tmp → rename):

    {root}/
      step_000500/
        base.pt           # base parameters (P-store snapshot)
        lora.pt           # LoRA parameters (F-store snapshot)
        optimizer.pt      # Adam m/v/step per param
        controller.pt     # signal computer state
        stores.pt         # R-store entries
        rng.pt            # RNG snapshot
        meta.json         # step, timestamp, config digest
      step_001000/
        ...
      latest -> step_001000   # symlink, best-effort

``CheckpointManager.load()`` with no argument restores ``latest``; pass
``step=500`` to restore a specific earlier point.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .controller.consolidation import ConsolidationScheduler
from .controller.signals import SignalComputer
from .evaluation import ForgettingTracker
from .stores.retrieval import RetrievalStore

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    enabled: bool = False
    dir: str = "checkpoints"
    save_every: int = 1000
    keep_last: int = 3
    resume_from: Optional[str] = None  # "latest" | "{step}" | absolute path


def _split_student_state(model: nn.Module) -> tuple[dict, dict]:
    """Split student state_dict into base vs LoRA tensors.

    We key off the parameter name containing ``.lora_a`` / ``.lora_b``
    (the convention in ``LoRALinear``). Everything else goes into base.
    """
    base: dict[str, torch.Tensor] = {}
    lora: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        lname = name.lower()
        if "lora_a" in lname or "lora_b" in lname or ".lora." in lname:
            lora[name] = tensor.detach().cpu()
        else:
            base[name] = tensor.detach().cpu()
    return base, lora


def _snapshot_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointManager:
    """Owns the checkpoint directory and the save/load protocol."""

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.dir)
        if cfg.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- save ---------------------------------------------------------------

    def save(
        self,
        *,
        step: int,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        signal_computer: SignalComputer,
        r_store: RetrievalStore,
        consolidator: Optional[ConsolidationScheduler] = None,
        forgetting: Optional[ForgettingTracker] = None,
        extra: dict[str, Any] | None = None,
    ) -> Optional[Path]:
        if not self.cfg.enabled:
            return None

        step_dir = self.root / f"step_{step:08d}"
        tmp_dir = self.root / f"_tmp_{step:08d}_{int(time.time())}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            base, lora = _split_student_state(student)
            torch.save(base, tmp_dir / "base.pt")
            torch.save(lora, tmp_dir / "lora.pt")
            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            torch.save(signal_computer.state_dict(), tmp_dir / "controller.pt")
            torch.save(r_store.state_dict(), tmp_dir / "stores.pt")
            torch.save(_snapshot_rng(), tmp_dir / "rng.pt")
            if consolidator is not None:
                torch.save(consolidator.state_dict(), tmp_dir / "consolidation.pt")
            if forgetting is not None:
                torch.save(forgetting.state_dict(), tmp_dir / "forgetting.pt")

            meta = {
                "step": int(step),
                "timestamp": time.time(),
                "extra": extra or {},
            }
            (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            # Atomic swap into place
            if step_dir.exists():
                shutil.rmtree(step_dir)
            tmp_dir.rename(step_dir)

            # Update "latest" symlink (best effort — symlinks are unreliable
            # on some filesystems so we also scan directory names on load)
            latest = self.root / "latest"
            try:
                if latest.is_symlink() or latest.exists():
                    latest.unlink()
                latest.symlink_to(step_dir.name)
            except OSError:
                pass

            self._prune()
            logger.info("checkpoint saved: %s", step_dir)
            return step_dir
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _prune(self) -> None:
        if self.cfg.keep_last <= 0:
            return
        step_dirs = sorted(
            [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("step_")]
        )
        for old in step_dirs[: -self.cfg.keep_last]:
            shutil.rmtree(old, ignore_errors=True)

    # -- load ---------------------------------------------------------------

    def resolve_resume_path(self) -> Optional[Path]:
        spec = self.cfg.resume_from
        if not spec:
            return None
        if spec == "latest":
            return self._latest()
        # Direct step number
        try:
            step = int(spec)
            cand = self.root / f"step_{step:08d}"
            return cand if cand.exists() else None
        except ValueError:
            pass
        # Absolute path to a step directory
        p = Path(spec)
        return p if p.exists() else None

    def _latest(self) -> Optional[Path]:
        latest = self.root / "latest"
        if latest.is_symlink() and latest.exists():
            return latest.resolve()
        step_dirs = sorted(
            [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("step_")]
        )
        return step_dirs[-1] if step_dirs else None

    def load(
        self,
        path: Path,
        *,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        signal_computer: SignalComputer,
        r_store: RetrievalStore,
        consolidator: Optional[ConsolidationScheduler] = None,
        forgetting: Optional[ForgettingTracker] = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        # weights_only=False: our own checkpoints carry numpy arrays in
        # controller/RNG state, which torch>=2.6's default unpickler rejects.
        # Trust is fine here because we only load checkpoints we wrote.
        base = torch.load(path / "base.pt", map_location=map_location, weights_only=False)
        lora = torch.load(path / "lora.pt", map_location=map_location, weights_only=False)
        merged = {**base, **lora}
        missing, unexpected = student.load_state_dict(merged, strict=False)
        if missing:
            logger.warning("checkpoint missing keys: %s", list(missing)[:8])
        if unexpected:
            logger.warning("checkpoint unexpected keys: %s", list(unexpected)[:8])

        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=map_location, weights_only=False))
        signal_computer.load_state_dict(torch.load(path / "controller.pt", map_location=map_location, weights_only=False))
        r_store.load_state_dict(torch.load(path / "stores.pt", map_location=map_location, weights_only=False))
        _restore_rng(torch.load(path / "rng.pt", map_location="cpu", weights_only=False))

        cons_path = path / "consolidation.pt"
        if consolidator is not None and cons_path.exists():
            consolidator.load_state_dict(torch.load(cons_path, map_location=map_location, weights_only=False))
        forget_path = path / "forgetting.pt"
        if forgetting is not None and forget_path.exists():
            forgetting.load_state_dict(torch.load(forget_path, map_location="cpu", weights_only=False))

        meta = json.loads((path / "meta.json").read_text())
        logger.info("checkpoint loaded: %s (step=%d)", path, meta["step"])
        return meta
