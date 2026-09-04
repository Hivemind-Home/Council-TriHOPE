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
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .controller.consolidation import ConsolidationScheduler
from .controller.signals import SignalComputer
from .data.sampler import StatefulSampler
from .evaluation import ForgettingTracker, PhaseEvalTracker
from .stores.retrieval import RetrievalStore
from .tracing import ModuleLedger

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    enabled: bool = False
    dir: str = "checkpoints"
    save_every: int = 1000
    keep_last: int = 3
    keep_tagged: int = 4  # cap for tagged (e.g. pre_merge rollback) dirs
    resume_from: Optional[str] = None  # "latest" | "{step}" | absolute path
    # Tagged ``pre_phase_<name>`` save at the last step before each listed
    # stream phase (E5's full-restore counterfactual resumes from it).
    save_before_phases: list[str] = field(default_factory=list)
    # Escape hatch for the key-mismatch guard in ``load()``. Leave false:
    # a mismatch means the checkpoint does not describe this model, and
    # silently loading nothing is far worse than stopping.
    allow_partial_load: bool = False


_UNTAGGED_RE = re.compile(r"^step_\d+$")


def _reject_wrapped(model: nn.Module, where: str) -> None:
    """Refuse a ``DistributedDataParallel``-wrapped student.

    DDP's ``state_dict`` keys carry a ``module.`` prefix. Saving through the
    wrapper and loading through the raw module (or vice versa) makes *every*
    key mismatch — and because ``load_state_dict`` is called with
    ``strict=False``, the model would silently load nothing while the
    optimizer, controller EMAs, R-store and RNG all restore correctly. The
    run then continues from a fresh backbone with a fully-warmed controller
    and never reports a problem.

    There is no safe way to guess which side the prefix belongs on, so this
    is a hard error rather than a fix-up.
    """
    if isinstance(model, nn.parallel.DistributedDataParallel):
        raise TypeError(
            f"CheckpointManager.{where}() requires the unwrapped student module. "
            "A DistributedDataParallel wrapper produces 'module.'-prefixed "
            "state_dict keys, which would silently match nothing on load. "
            "Pass ddp.module (or the raw student you wrapped)."
        )


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

    def __init__(self, cfg: CheckpointConfig, *, is_main: bool = True) -> None:
        self.cfg = cfg
        self.root = Path(cfg.dir)
        # Under DDP only rank 0 writes checkpoints: the tmp→rename→symlink
        # dance in ``save`` is not safe to run concurrently from N processes
        # against one directory. Every rank still *reads* (resume restores the
        # same state everywhere), so ``load`` and the resolve helpers are
        # unguarded — and so is this mkdir, which is idempotent, so that the
        # read paths never race against rank 0 creating the root.
        self.is_main = is_main
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
        ledger: Optional[ModuleLedger] = None,
        sampler: Optional[StatefulSampler] = None,
        phase_eval: Optional[PhaseEvalTracker] = None,
        tag: Optional[str] = None,
        extra: dict[str, Any] | None = None,
    ) -> Optional[Path]:
        """Save a checkpoint; ``tag`` (e.g. ``"pre_merge"``) creates a
        separately-pruned rollback directory that ``latest`` never resolves
        to — resume from it explicitly via ``checkpoint.resume_from=<path>``.
        """
        if not self.cfg.enabled:
            return None
        _reject_wrapped(student, "save")
        if not self.is_main:
            return None

        suffix = f"_{tag}" if tag else ""
        step_dir = self.root / f"step_{step:08d}{suffix}"
        tmp_dir = self.root / f"_tmp_{step:08d}{suffix}_{int(time.time())}"
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
            if ledger is not None:
                torch.save(ledger.state_dict(), tmp_dir / "ledger.pt")
            if sampler is not None:
                torch.save(sampler.state_dict(), tmp_dir / "data.pt")
            if phase_eval is not None:
                torch.save(phase_eval.state_dict(), tmp_dir / "phase_eval.pt")

            meta = {
                "step": int(step),
                "timestamp": time.time(),
                "tag": tag,
                "extra": extra or {},
            }
            (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            # Atomic swap into place
            if step_dir.exists():
                shutil.rmtree(step_dir)
            tmp_dir.rename(step_dir)

            # Update "latest" symlink (best effort — symlinks are unreliable
            # on some filesystems so we also scan directory names on load).
            # Tagged rollback checkpoints never become "latest".
            if tag is None:
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
        all_dirs = [
            p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("step_")
        ]
        untagged = sorted(p for p in all_dirs if _UNTAGGED_RE.match(p.name))
        tagged = sorted(p for p in all_dirs if not _UNTAGGED_RE.match(p.name))
        if self.cfg.keep_last > 0:
            for old in untagged[: -self.cfg.keep_last]:
                shutil.rmtree(old, ignore_errors=True)
        if self.cfg.keep_tagged > 0:
            for old in tagged[: -self.cfg.keep_tagged]:
                shutil.rmtree(old, ignore_errors=True)

    # -- load ---------------------------------------------------------------

    def resolve_resume_path(self) -> Optional[Path]:
        spec = self.cfg.resume_from
        if not spec:
            return None
        if not self.root.exists():
            # checkpoint.enabled=false never creates the root, so _latest()
            # would raise FileNotFoundError from iterdir() instead of saying
            # what is actually wrong.
            raise FileNotFoundError(
                f"checkpoint.resume_from={spec!r} but {self.root} does not exist. "
                "Set checkpoint.enabled=true, or point resume_from at a real path."
            )
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
            p
            for p in self.root.iterdir()
            if p.is_dir() and _UNTAGGED_RE.match(p.name)
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
        ledger: Optional[ModuleLedger] = None,
        sampler: Optional[StatefulSampler] = None,
        phase_eval: Optional[PhaseEvalTracker] = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        _reject_wrapped(student, "load")
        # weights_only=False: our own checkpoints carry numpy arrays in
        # controller/RNG state, which torch>=2.6's default unpickler rejects.
        # Trust is fine here because we only load checkpoints we wrote.
        base = torch.load(path / "base.pt", map_location=map_location, weights_only=False)
        lora = torch.load(path / "lora.pt", map_location=map_location, weights_only=False)
        merged = {**base, **lora}
        # ``merged`` is the full state_dict split by name and rejoined, so a
        # correctly-matched checkpoint yields two empty lists. Anything else
        # means this checkpoint does not describe this model — stop rather
        # than resume onto partially-restored weights.
        missing, unexpected = student.load_state_dict(merged, strict=False)
        if (missing or unexpected) and not self.cfg.allow_partial_load:
            raise RuntimeError(
                f"checkpoint {path} does not match the student model: "
                f"{len(missing)} missing key(s) {list(missing)[:8]}, "
                f"{len(unexpected)} unexpected key(s) {list(unexpected)[:8]}. "
                "This usually means the model config changed, or the "
                "checkpoint was written from a differently-wrapped module. "
                "Set checkpoint.allow_partial_load=true to load anyway."
            )
        if missing:
            logger.warning("checkpoint missing keys: %s", list(missing)[:8])
        if unexpected:
            logger.warning("checkpoint unexpected keys: %s", list(unexpected)[:8])

        optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location=map_location, weights_only=False)
        )
        signal_computer.load_state_dict(
            torch.load(path / "controller.pt", map_location=map_location, weights_only=False)
        )
        r_store.load_state_dict(
            torch.load(path / "stores.pt", map_location=map_location, weights_only=False)
        )
        _restore_rng(torch.load(path / "rng.pt", map_location="cpu", weights_only=False))

        cons_path = path / "consolidation.pt"
        if consolidator is not None and cons_path.exists():
            consolidator.load_state_dict(
                torch.load(cons_path, map_location=map_location, weights_only=False)
            )
        forget_path = path / "forgetting.pt"
        if forgetting is not None and forget_path.exists():
            forgetting.load_state_dict(
                torch.load(forget_path, map_location="cpu", weights_only=False)
            )
        ledger_path = path / "ledger.pt"
        if ledger is not None and ledger_path.exists():
            ledger.load_state_dict(torch.load(ledger_path, map_location="cpu", weights_only=False))
        data_path = path / "data.pt"
        if sampler is not None and data_path.exists():
            sampler.load_state_dict(torch.load(data_path, map_location="cpu", weights_only=False))
        phase_eval_path = path / "phase_eval.pt"
        if phase_eval is not None and phase_eval_path.exists():
            phase_eval.load_state_dict(
                torch.load(phase_eval_path, map_location="cpu", weights_only=False)
            )

        meta = json.loads((path / "meta.json").read_text())
        logger.info("checkpoint loaded: %s (step=%d)", path, meta["step"])
        return meta
