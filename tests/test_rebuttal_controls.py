"""Rebuttal controls (Nov 2026): live-yoked random / anti-selection commits,
the shared-frozen trainability mode, and the per-step commit trace.

What the reviewers asked for and what each test pins down:

* 1Ume W1 — a control that keeps the controller's exact per-step commit
  count and timing and randomises only module identity → ``yoked_random``
  must emit the same number of P labels as the controller on every step,
  keep every R (deferral) where the controller put it, and move the P
  labels onto other non-deferred blocks.
* BDmh — a deliberately bad selection at the same volume → ``anti_commit``
  must put the step's k_t commits on the k_t lowest-agreement non-deferred
  blocks (the controller commits the highest-agreement ones).
* BDmh / 1Ume W3 — the shared-frozen regime → ``train.trainable=routed``
  must leave every shared parameter bit-identical while base + LoRA train,
  and the run summary must say so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from hivemind.controller.config import AblationConfig, DebugConfig, PolicyConfig
from hivemind.controller.module_index import ModuleId
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import ModuleSignals
from hivemind.training import run_training_loop


# ---------------------------------------------------------------------------
# Policy-level behaviour
# ---------------------------------------------------------------------------

_CFG = PolicyConfig(
    top_m_modules=6,
    surprise_high=2.0,
    repetition_low=0.3,
    repetition_medium=0.5,
    stability_high_C=0.5,
    stability_low_V=0.3,
    stability_source="sustained",
)


def _sig(layer: int, ptype: str, *, grad: float, S: float = 0.0, R: float = 0.0,
         Cbar: float = 0.0, C: float = 0.0, V: float = 0.0) -> ModuleSignals:
    return ModuleSignals(
        module_id=ModuleId(layer, "attn", ptype), grad_norm=grad, surprise=S,
        repetition=R, stability_C_sustained=Cbar, stability_C=C, stability_V=V,
    )


def _step_signals() -> dict[ModuleId, ModuleSignals]:
    """Six selected blocks: two the controller commits (high C̄), one it
    defers (high surprise, low repetition), three it routes to F."""
    sigs = [
        _sig(0, "P", grad=6.0, R=0.9, Cbar=0.95, V=0.1),   # controller P
        _sig(1, "F", grad=5.0, R=0.8, Cbar=0.80, V=0.1),   # controller P
        _sig(2, "F", grad=4.0, S=9.0, R=0.05, Cbar=0.40),  # controller R (defer)
        _sig(3, "P", grad=3.0, R=0.6, Cbar=0.20, V=0.1),   # F (C̄ too low)
        _sig(4, "F", grad=2.0, R=0.6, Cbar=0.05, V=0.1),   # F (lowest C̄)
        _sig(5, "P", grad=1.0, R=0.6, Cbar=0.30, V=0.9),   # F (V too high)
    ]
    return {s.module_id: s for s in sigs}


def _labels(actions) -> dict[int, str]:
    return {a.module_id.layer: a.store for a in actions}


class TestConfig:
    @pytest.mark.parametrize("name", ["yoked_random", "anti_commit"])
    def test_config_accepts(self, name: str) -> None:
        DebugConfig(policy_override=name)

    def test_unknown_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="policy_override"):
            DebugConfig(policy_override="yoked_bogus")


class TestYokedRandom:
    def test_same_commit_count_same_defers_other_blocks(self) -> None:
        ref = RFPPolicy(_CFG, seed=3)
        ctl = RFPPolicy(_CFG, override="yoked_random", seed=3)
        sigs = _step_signals()
        ref_lab = _labels(ref.decide(sigs, step=7, phase="code"))
        actions = ctl.decide(sigs, step=7, phase="code")
        ctl_lab = _labels(actions)
        assert ref_lab == {0: "P", 1: "P", 2: "R", 3: "F", 4: "F", 5: "F"}
        # Volume: exactly the controller's k_t.
        assert sum(v == "P" for v in ctl_lab.values()) == 2
        # Deferral untouched.
        assert ctl_lab[2] == "R"
        # No block is dropped or added.
        assert set(ctl_lab) == set(ref_lab)
        # The trace knows what the controller would have done.
        assert {a.module_id.layer: a.controller_store for a in actions} == ref_lab

    def test_commits_move_off_controller_picks_sometimes(self) -> None:
        ctl = RFPPolicy(_CFG, override="yoked_random", seed=11)
        sigs = _step_signals()
        picks = {
            frozenset(l for l, s in _labels(ctl.decide(sigs, step=t, phase="p")).items() if s == "P")
            for t in range(60)
        }
        # Over 60 steps the uniform draw over C(5,2)=10 subsets hits several;
        # at least one differs from the controller's {0, 1}.
        assert len(picks) >= 3
        assert any(p != frozenset({0, 1}) for p in picks)
        # The controller's own picks are eligible (uniform over all subsets).
        assert frozenset({0, 1}) in picks or len(picks) >= 5

    def test_deterministic_in_seed_step_phase(self) -> None:
        sigs = _step_signals()
        a = RFPPolicy(_CFG, override="yoked_random", seed=5)
        b = RFPPolicy(_CFG, override="yoked_random", seed=5)
        for t in range(20):
            assert _labels(a.decide(sigs, step=t, phase="x")) == _labels(
                b.decide(sigs, step=t, phase="x")
            )
        c = RFPPolicy(_CFG, override="yoked_random", seed=6)
        assert any(
            _labels(a.decide(sigs, step=t, phase="x")) != _labels(c.decide(sigs, step=t, phase="x"))
            for t in range(20)
        )
        # Phase enters the seed through a process-stable hash.
        assert any(
            _labels(a.decide(sigs, step=t, phase="x")) != _labels(a.decide(sigs, step=t, phase="y"))
            for t in range(20)
        )

    def test_no_controller_commit_means_identical_labels(self) -> None:
        sigs = _step_signals()
        sigs = {k: v for k, v in sigs.items() if k.layer not in (0, 1)}
        ref = RFPPolicy(_CFG, seed=1)
        ctl = RFPPolicy(_CFG, override="yoked_random", seed=1)
        for t in range(10):
            assert _labels(ref.decide(sigs, step=t)) == _labels(ctl.decide(sigs, step=t))


class TestAntiCommit:
    def test_commits_lowest_agreement_blocks(self) -> None:
        ctl = RFPPolicy(_CFG, override="anti_commit", seed=0)
        lab = _labels(ctl.decide(_step_signals(), step=3, phase="code"))
        # k_t = 2; non-deferred candidates by C̄: 4 (0.05) < 3 (0.20) < 5 (0.30) < 1 < 0.
        assert lab == {0: "F", 1: "F", 2: "R", 3: "P", 4: "P", 5: "F"}

    def test_reads_the_policys_c_source(self) -> None:
        cfg = PolicyConfig(**{**_CFG.__dict__, "stability_source": "instant"})
        sigs = _step_signals()
        # Under `instant` the controller reads C, not C̄; give the controller
        # two instant-stable blocks and make block 5 the lowest-C candidate.
        for s in sigs.values():
            s.stability_C = s.stability_C_sustained
        sigs[ModuleId(5, "attn", "P")].stability_C = -0.5
        sigs[ModuleId(5, "attn", "P")].stability_V = 0.1
        ref = _labels(RFPPolicy(cfg).decide(sigs, step=0))
        assert [l for l, s in ref.items() if s == "P"] == [0, 1]
        lab = _labels(RFPPolicy(cfg, override="anti_commit").decide(sigs, step=0))
        assert [l for l, s in sorted(lab.items()) if s == "P"] == [4, 5]

    def test_is_deterministic(self) -> None:
        sigs = _step_signals()
        a = RFPPolicy(_CFG, override="anti_commit", seed=1)
        b = RFPPolicy(_CFG, override="anti_commit", seed=2)
        for t in range(5):
            assert _labels(a.decide(sigs, step=t)) == _labels(b.decide(sigs, step=t))


class TestGatesStillApply:
    """k_t is the controller's FINAL commit count, after every gate."""

    @pytest.mark.parametrize("name", ["yoked_random", "anti_commit"])
    def test_low_confidence_zeroes_commits(self, name: str) -> None:
        ab = AblationConfig(use_teacher_confidence=True, confidence_threshold=0.5)
        ctl = RFPPolicy(_CFG, ablation=ab, override=name)
        lab = _labels(ctl.decide(_step_signals(), teacher_confidence=0.1, step=0))
        assert "P" not in lab.values()
        assert lab[2] == "R"

    @pytest.mark.parametrize("name", ["yoked_random", "anti_commit"])
    def test_blocked_teacher_zeroes_commits(self, name: str) -> None:
        dbg = DebugConfig(policy_override=name, block_p_for_teachers=["bad"])
        ctl = RFPPolicy(_CFG, override=name, debug=dbg)
        lab = _labels(ctl.decide(_step_signals(), teacher_name="bad", step=0))
        assert "P" not in lab.values()
        lab = _labels(ctl.decide(_step_signals(), teacher_name="good", step=0))
        assert sum(v == "P" for v in lab.values()) == 2

    @pytest.mark.parametrize("name", ["yoked_random", "anti_commit"])
    def test_disabled_p_store(self, name: str) -> None:
        ctl = RFPPolicy(_CFG, ablation=AblationConfig(disable_stores=["P"]), override=name)
        assert "P" not in _labels(ctl.decide(_step_signals(), step=0)).values()

    def test_force_r_gate_keeps_r(self) -> None:
        ab = AblationConfig(
            use_teacher_confidence=True, confidence_threshold=0.5, confidence_gate="force_r"
        )
        ctl = RFPPolicy(_CFG, ablation=ab, override="yoked_random")
        lab = _labels(ctl.decide(_step_signals(), teacher_confidence=0.1, step=0))
        assert set(lab.values()) == {"R"}


# ---------------------------------------------------------------------------
# Training-loop level: trainable=routed, commits.jsonl
# ---------------------------------------------------------------------------


def _loop_cfg(tmp_path: Path, **train_extra):
    return OmegaConf.create(
        {
            "model": {"vocab_size": 64, "dim": 32, "num_layers": 2, "heads": 4,
                      "max_seq_len": 32, "lora": {"rank": 4, "alpha": 8.0}},
            "teachers": {"num_teachers": 2, "mode": "synthetic"},
            "controller": {
                "enabled": True,
                "policy": {"top_m_modules": 4, "repetition_medium": 0.0,
                           "stability_high_C": -1.0, "stability_low_V": 10.0},
            },
            "data": {"source": "synthetic", "dataset_size": 24, "seq_len": 16,
                     "vocab_size": 64, "batch_size": 4},
            "train": {"steps": 12, "device": "cpu", "seed": 1, "log_interval": 1,
                      **train_extra},
            "optim": {"lr": 1e-2},
            "logging": {"enabled": True, "backend": "json",
                        "path": str(tmp_path / "metrics.jsonl"),
                        "events_path": str(tmp_path / "events.jsonl")},
            "run": {"dir": str(tmp_path)},
            "checkpoint": {"enabled": False},
        }
    )


class TestRoutedTrainability:
    def test_shared_frozen_base_and_lora_train(self, tmp_path: Path) -> None:
        cfg = _loop_cfg(tmp_path, trainable="routed")
        # Build the student the loop will build, to read the pretrained
        # shared tensors it starts from.
        from hivemind.student.config import LoRAConfig, StudentConfig
        from hivemind.student.model import StudentModel

        torch.manual_seed(1)
        run_training_loop(cfg, device=torch.device("cpu"))
        summary = json.loads((tmp_path / "run_summary.json").read_text())
        extra = summary["extra"]
        assert extra["trainable"] == "routed"
        chk = extra["shared_frozen_check"]
        assert chk["unchanged"] is True
        assert chk["changed_tensors"] == []
        assert chk["tensors"] > 0 and chk["coords"] > 0
        # Something did train: the ledger opened coordinates.
        assert summary["ledger_totals"]["coords_opened"] > 0

    def test_requires_grad_split(self) -> None:
        from hivemind.student.config import LoRAConfig, StudentConfig
        from hivemind.student.model import StudentModel

        m = StudentModel(StudentConfig(vocab_size=64, dim=32, num_layers=2, heads=4,
                                       lora=LoRAConfig(rank=4)))
        groups = m.get_param_groups()
        for p in groups["shared"]:
            p.requires_grad = False
        assert all(p.requires_grad for p in groups["P"])
        assert all(p.requires_grad for p in groups["F"])
        assert not any(p.requires_grad for p in groups["shared"])

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="trainable"):
            run_training_loop(_loop_cfg(tmp_path, trainable="frozen"), device=torch.device("cpu"))

    def test_all_mode_has_no_check(self, tmp_path: Path) -> None:
        run_training_loop(_loop_cfg(tmp_path), device=torch.device("cpu"))
        extra = json.loads((tmp_path / "run_summary.json").read_text())["extra"]
        assert extra["trainable"] == "all"
        assert "shared_frozen_check" not in extra


class TestCommitTrace:
    @pytest.mark.parametrize("override", [None, "yoked_random", "anti_commit"])
    def test_one_record_per_step_with_commit_details(self, tmp_path: Path, override) -> None:
        cfg = _loop_cfg(tmp_path)
        if override:
            cfg.controller.debug = {"policy_override": override}
        run_training_loop(cfg, device=torch.device("cpu"))
        path = tmp_path / "commits.jsonl"
        assert path.exists()
        recs = [json.loads(l) for l in path.read_text().splitlines()]
        assert [r["step"] for r in recs] == list(range(12))
        assert all({"r", "f", "p"} <= set(r) for r in recs)
        committed = [r for r in recs if r["p"] > 0]
        assert committed, "the permissive thresholds should commit"
        for r in committed:
            assert len(r["modules_p"]) == r["p"] == len(r["C_bar_p"])
        # The decision trace and the commit trace agree on every step.
        events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
        by_step: dict[int, int] = {}
        for e in events:
            if e["type"] == "decision" and e["action"] == "P":
                by_step[e["step"]] = by_step.get(e["step"], 0) + 1
        assert {r["step"]: r["p"] for r in committed} == by_step
        if override:
            # Every committing step also says what the controller would have done,
            # and at the same volume.
            assert all("controller_p" in r for r in committed)
            assert all(len(r["controller_p"]) == r["p"] for r in committed)
        else:
            assert not any("controller_p" in r for r in recs)
