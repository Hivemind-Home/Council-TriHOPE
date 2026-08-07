"""Checkpoint save/load roundtrip."""

from __future__ import annotations

import pytest
import torch

from hivemind.checkpoint import CheckpointConfig, CheckpointManager
from hivemind.controller.config import ControllerConfig
from hivemind.controller.module_index import build_module_index
from hivemind.controller.signals import SignalComputer
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.stores.retrieval import RetrievalEntry, RetrievalStore
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def _build_stack():
    cfg = StudentConfig(
        vocab_size=64, dim=32, num_layers=2, heads=4, max_seq_len=16,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    student = StudentModel(cfg)
    opt = MaskedAdamW(student.parameters(), lr=1e-3)
    mods = build_module_index(student)
    sig = SignalComputer(ControllerConfig(), mods)
    r = RetrievalStore(max_size=16)
    return student, opt, sig, r


def test_save_and_load_roundtrip(tmp_path):
    student, opt, sig, r = _build_stack()

    # Create some meaningful state to check it round-trips
    tokens = torch.randint(0, 64, (2, 16))
    loss = student(tokens).sum()
    loss.backward()
    opt.step()

    r.add(RetrievalEntry(embedding=torch.randn(32), teacher_id=0, bucket_id=42, step=1))
    # Seed stability EMA by touching one tracker
    for tr in sig._stability.values():
        tr.update_windowed_variance(1.5)

    mgr = CheckpointManager(
        CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1, keep_last=2)
    )
    mgr.save(step=5, student=student, optimizer=opt, signal_computer=sig, r_store=r, extra={"k": 1})

    # New stack, restore, compare
    s2, o2, sig2, r2 = _build_stack()
    # Different initial state by design
    assert not torch.equal(s2.embed.weight, student.embed.weight)

    path = mgr._latest()
    assert path is not None
    meta = mgr.load(path, student=s2, optimizer=o2, signal_computer=sig2, r_store=r2)
    assert meta["step"] == 5

    # Weights match
    assert torch.equal(s2.embed.weight, student.embed.weight)
    # R-store entries preserved
    assert r2.size == 1
    # Stability state preserved (pick any tracker)
    any_tracker = next(iter(sig._stability.values()))
    any_tracker2 = next(iter(sig2._stability.values()))
    assert any_tracker2._initialized == any_tracker._initialized


def test_resume_from_latest(tmp_path):
    student, opt, sig, r = _build_stack()
    cfg = CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1, keep_last=3)
    mgr = CheckpointManager(cfg)
    mgr.save(step=1, student=student, optimizer=opt, signal_computer=sig, r_store=r)
    mgr.save(step=2, student=student, optimizer=opt, signal_computer=sig, r_store=r)

    cfg2 = CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), resume_from="latest")
    mgr2 = CheckpointManager(cfg2)
    path = mgr2.resolve_resume_path()
    assert path is not None
    assert path.name == "step_00000002"


def test_keep_last_prunes(tmp_path):
    student, opt, sig, r = _build_stack()
    cfg = CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1, keep_last=2)
    mgr = CheckpointManager(cfg)
    for s in range(1, 5):
        mgr.save(step=s, student=student, optimizer=opt, signal_computer=sig, r_store=r)

    step_dirs = sorted(
        p.name for p in (tmp_path / "ckpt").iterdir() if p.name.startswith("step_")
    )
    assert step_dirs == ["step_00000003", "step_00000004"]


# -- guards against the silent-corruption paths --------------------------------


class TestWrappedModelRejected:
    """A DDP-wrapped student must never reach save/load.

    Its state_dict keys carry a ``module.`` prefix, so with ``strict=False``
    the model would load *nothing* while the optimizer, controller and RNG
    all restore — a resumed run on a fresh backbone, with no error.
    """

    @staticmethod
    def _ddp(student):
        # Construct the wrapper without a process group: DDP's __init__ needs
        # one, so build the object directly and attach the module. We only
        # need isinstance() to be true.
        ddp = torch.nn.parallel.DistributedDataParallel.__new__(
            torch.nn.parallel.DistributedDataParallel
        )
        torch.nn.Module.__init__(ddp)
        ddp.module = student
        return ddp

    def test_save_rejects_ddp(self, tmp_path):
        student, opt, sig, r = _build_stack()
        mgr = CheckpointManager(
            CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1)
        )
        with pytest.raises(TypeError, match="unwrapped student"):
            mgr.save(
                step=1,
                student=self._ddp(student),
                optimizer=opt,
                signal_computer=sig,
                r_store=r,
            )

    def test_load_rejects_ddp(self, tmp_path):
        student, opt, sig, r = _build_stack()
        mgr = CheckpointManager(
            CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1)
        )
        mgr.save(step=1, student=student, optimizer=opt, signal_computer=sig, r_store=r)
        with pytest.raises(TypeError, match="unwrapped student"):
            mgr.load(
                mgr._latest(),
                student=self._ddp(student),
                optimizer=opt,
                signal_computer=sig,
                r_store=r,
            )

    def test_save_rejects_before_the_is_main_guard(self, tmp_path):
        """A config bug must fail on every rank, not just rank 0."""
        student, opt, sig, r = _build_stack()
        mgr = CheckpointManager(
            CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1),
            is_main=False,
        )
        with pytest.raises(TypeError, match="unwrapped student"):
            mgr.save(
                step=1,
                student=self._ddp(student),
                optimizer=opt,
                signal_computer=sig,
                r_store=r,
            )


class TestKeyMismatchIsLoud:
    def test_load_raises_on_missing_keys(self, tmp_path):
        student, opt, sig, r = _build_stack()
        mgr = CheckpointManager(
            CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1)
        )
        mgr.save(step=1, student=student, optimizer=opt, signal_computer=sig, r_store=r)

        # Simulate the DDP-prefix failure mode without needing a process group.
        path = mgr._latest()
        base = torch.load(path / "base.pt", weights_only=False)
        torch.save({f"module.{k}": v for k, v in base.items()}, path / "base.pt")

        s2, o2, sig2, r2 = _build_stack()
        with pytest.raises(RuntimeError, match="does not match the student model"):
            mgr.load(path, student=s2, optimizer=o2, signal_computer=sig2, r_store=r2)

    def test_allow_partial_load_overrides(self, tmp_path):
        student, opt, sig, r = _build_stack()
        cfg = CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1)
        mgr = CheckpointManager(cfg)
        mgr.save(step=1, student=student, optimizer=opt, signal_computer=sig, r_store=r)

        path = mgr._latest()
        base = torch.load(path / "base.pt", weights_only=False)
        torch.save({f"module.{k}": v for k, v in base.items()}, path / "base.pt")

        s2, o2, sig2, r2 = _build_stack()
        mgr.cfg.allow_partial_load = True
        meta = mgr.load(path, student=s2, optimizer=o2, signal_computer=sig2, r_store=r2)
        assert meta["step"] == 1

    def test_clean_roundtrip_has_no_key_mismatch(self, tmp_path):
        """The guard must not fire on a correct checkpoint."""
        student, opt, sig, r = _build_stack()
        mgr = CheckpointManager(
            CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1)
        )
        mgr.save(step=1, student=student, optimizer=opt, signal_computer=sig, r_store=r)
        s2, o2, sig2, r2 = _build_stack()
        mgr.load(mgr._latest(), student=s2, optimizer=o2, signal_computer=sig2, r_store=r2)
        assert torch.equal(s2.embed.weight, student.embed.weight)


class TestNonMainRankDoesNotWrite:
    def test_save_is_a_noop_off_rank_zero(self, tmp_path):
        student, opt, sig, r = _build_stack()
        mgr = CheckpointManager(
            CheckpointConfig(enabled=True, dir=str(tmp_path / "ckpt"), save_every=1),
            is_main=False,
        )
        assert mgr.save(
            step=1, student=student, optimizer=opt, signal_computer=sig, r_store=r
        ) is None
        assert not any(
            p.name.startswith("step_") for p in (tmp_path / "ckpt").iterdir()
        )
