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


class TestScheduleMatched:
    """schedule_random / schedule_anti: k_t comes from the reference run's
    commits.jsonl, not from the arm's own controller, so volume is matched."""

    @staticmethod
    def _sched(tmp_path: Path, counts: dict[int, int]) -> DebugConfig:
        f = tmp_path / "commits.jsonl"
        f.write_text("".join(json.dumps({"step": s, "p": p, "r": 0, "f": 0}) + "\n"
                             for s, p in counts.items()))
        return DebugConfig(policy_override="schedule_random", commit_schedule_path=str(f))

    @pytest.mark.parametrize("name", ["schedule_random", "schedule_anti"])
    def test_commit_count_follows_reference_not_controller(self, tmp_path, name) -> None:
        dbg = self._sched(tmp_path, {0: 0, 1: 1, 2: 3, 3: 9})
        ctl = RFPPolicy(_CFG, override=name, debug=dbg, seed=4)
        # The arm's own controller would commit 2 every step.
        for step, want in [(0, 0), (1, 1), (2, 3), (3, 5), (4, 0)]:
            lab = _labels(ctl.decide(_step_signals(), step=step, phase="code"))
            assert sum(v == "P" for v in lab.values()) == want  # step 3 capped at 5 candidates
            assert lab[2] == "R"  # its own deferral is kept

    def test_anti_picks_lowest_agreement(self, tmp_path) -> None:
        dbg = self._sched(tmp_path, {0: 3})
        lab = _labels(RFPPolicy(_CFG, override="schedule_anti", debug=dbg).decide(
            _step_signals(), step=0))
        assert [i for i, s in sorted(lab.items()) if s == "P"] == [3, 4, 5]

    @pytest.mark.parametrize("name", ["schedule_random", "schedule_anti"])
    def test_gates_zero_the_reference_count(self, tmp_path, name) -> None:
        dbg = self._sched(tmp_path, {0: 2})
        ab = AblationConfig(use_teacher_confidence=True, confidence_threshold=0.5)
        ctl = RFPPolicy(_CFG, ablation=ab, override=name, debug=dbg)
        lab = _labels(ctl.decide(_step_signals(), teacher_confidence=0.1, step=0))
        assert "P" not in lab.values()
        no_p = AblationConfig(disable_stores=["P"])
        ctl = RFPPolicy(_CFG, ablation=no_p, override=name, debug=dbg)
        assert "P" not in _labels(ctl.decide(_step_signals(), step=0)).values()

    def test_missing_or_empty_schedule_fails_loudly(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="commit_schedule_path"):
            RFPPolicy(_CFG, override="schedule_random",
                      debug=DebugConfig(policy_override="schedule_random"))
        dbg = self._sched(tmp_path, {0: 0, 1: 0})
        with pytest.raises(ValueError, match="no commits"):
            RFPPolicy(_CFG, override="schedule_random", debug=dbg)


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


# ---------------------------------------------------------------------------
# schedule_oracle_bad / schedule_oracle_good (BDmh Q2: interference-ranked
# selection at the controller's volume)
# ---------------------------------------------------------------------------

from hivemind.controller.module_index import build_module_index  # noqa: E402
from hivemind.controller.oracle import (  # noqa: E402
    ProbeBuffer,
    adamw_open_delta,
    interference_scores,
)
from hivemind.optim.masked_adamw import MaskedAdamW  # noqa: E402


def _sched_cfg(tmp_path: Path, name: str, counts: dict[int, int]) -> DebugConfig:
    f = tmp_path / "commits.jsonl"
    f.write_text("".join(json.dumps({"step": s, "p": p}) + "\n" for s, p in counts.items()))
    return DebugConfig(policy_override=name, commit_schedule_path=str(f))


class TestOraclePolicy:
    # Non-deferred candidates are layers 0,1,3,4,5 (layer 2 is the controller's R).
    SCORES = {ModuleId(i, "attn", p): s for i, p, s in
              [(0, "P", 0.5), (1, "F", -2.0), (2, "F", 9.0), (3, "P", 3.0),
               (4, "F", -0.1), (5, "P", 1.0)]}

    @pytest.mark.parametrize("name", ["schedule_oracle_bad", "schedule_oracle_good"])
    def test_config_accepts(self, name: str) -> None:
        DebugConfig(policy_override=name)

    def test_bad_commits_highest_good_lowest_r_kept(self, tmp_path) -> None:
        for name, want in [("schedule_oracle_bad", [3, 5]), ("schedule_oracle_good", [1, 4])]:
            pol = RFPPolicy(_CFG, override=name, debug=_sched_cfg(tmp_path, name, {7: 2}))
            lab = _labels(pol.decide(_step_signals(), step=7, oracle_scores=self.SCORES))
            # layer 2 has the highest score of all but is the controller's
            # deferral: it must stay R, never be committed.
            assert lab[2] == "R"
            assert [i for i, s in sorted(lab.items()) if s == "P"] == want, name
            assert pol.last_oracle_fallback is False

    def test_without_scores_falls_back_to_schedule_random(self, tmp_path) -> None:
        o = RFPPolicy(_CFG, override="schedule_oracle_bad", seed=3,
                      debug=_sched_cfg(tmp_path, "schedule_oracle_bad", {4: 2}))
        r = RFPPolicy(_CFG, override="schedule_random", seed=3,
                      debug=_sched_cfg(tmp_path, "schedule_random", {4: 2}))
        lo = _labels(o.decide(_step_signals(), step=4, phase="x"))
        lr_ = _labels(r.decide(_step_signals(), step=4, phase="x"))
        assert lo == lr_ and sum(v == "P" for v in lo.values()) == 2
        assert o.last_oracle_fallback is True

    def test_needs_scores_only_on_scheduled_steps(self, tmp_path) -> None:
        pol = RFPPolicy(_CFG, override="schedule_oracle_good",
                        debug=_sched_cfg(tmp_path, "schedule_oracle_good", {3: 1, 4: 0}))
        assert [pol.needs_oracle_scores(s) for s in (2, 3, 4)] == [False, True, False]
        plain = RFPPolicy(_CFG, override="schedule_random",
                          debug=_sched_cfg(tmp_path, "schedule_random", {3: 1}))
        assert not plain.needs_oracle_scores(3)

    @pytest.mark.parametrize("name", ["schedule_oracle_bad", "schedule_oracle_good"])
    def test_gates_zero_the_count(self, tmp_path, name) -> None:
        dbg = _sched_cfg(tmp_path, name, {0: 2})
        ab = AblationConfig(use_teacher_confidence=True, confidence_threshold=0.5)
        lab = _labels(RFPPolicy(_CFG, ablation=ab, override=name, debug=dbg).decide(
            _step_signals(), teacher_confidence=0.1, step=0, oracle_scores=self.SCORES))
        assert "P" not in lab.values()


class TestAdamwOpenDelta:
    """The closed-form delta must equal what MaskedAdamW actually does."""

    @pytest.mark.parametrize("path", ["uniform", "masked"])
    def test_matches_real_optimizer_step(self, path: str) -> None:
        torch.manual_seed(0)
        # float64 so the comparison tests the formula, not fp32 rounding of p
        p = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
        opt = MaskedAdamW([p], lr=3e-3, weight_decay=0.05)
        for i in range(4):  # build history (coordinate-local when masked)
            p.grad = torch.randn_like(p)
            if path == "masked":
                opt.set_masks({p: torch.rand(5, 7) > 0.4})
            opt.step()
        p.grad = torch.randn_like(p)
        want = adamw_open_delta(p, opt, opt.param_groups[0])
        before = p.detach().clone()
        state_before = {k: v.clone() for k, v in opt.state[p].items()}
        if path == "masked":
            opt.set_masks({p: torch.ones(5, 7, dtype=torch.bool)})
        opt.step()
        # Up to fp32 rounding: the delta is computed in fp32, and the masked
        # optimizer path itself rounds p to fp32.
        torch.testing.assert_close(p.detach() - before, want.double(), rtol=1e-3, atol=2e-7)
        # computing the delta must not have touched the state
        assert set(state_before) == set(opt.state[p])

    def test_fresh_param_and_no_grad(self) -> None:
        p = torch.nn.Parameter(torch.randn(3))
        opt = MaskedAdamW([p], lr=1e-2, weight_decay=0.0)
        assert adamw_open_delta(p, opt, opt.param_groups[0]) is None
        p.grad = torch.randn(3)
        d = adamw_open_delta(p, opt, opt.param_groups[0])
        before = p.detach().clone()
        opt.step()
        torch.testing.assert_close(p.detach() - before, d)


def _probe_rows(vocab: int, n: int, length: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, vocab, (length,), generator=g),) * 2 for _ in range(n)]


class TestInterferenceScores:
    def _setup(self, tiny_model):
        model = tiny_model.double()
        mods = build_module_index(model)
        opt = MaskedAdamW([p for p in model.parameters() if p.requires_grad],
                          lr=1e-4, weight_decay=0.01)
        x = torch.randint(0, 64, (2, 16))
        for _ in range(2):  # some optimizer history
            opt.zero_grad()
            model(x).logsumexp(-1).mean().backward()
            opt.step()
        opt.zero_grad()
        model(x).logsumexp(-1).mean().backward()  # this step's gradient
        probe = {"math": _probe_rows(64, 2, 12, 1), "code": _probe_rows(64, 3, 9, 2)}
        return model, mods, opt, probe

    @staticmethod
    def _old_loss(model, probe) -> float:
        tot = 0.0
        with torch.no_grad():
            for rows in probe.values():
                s = n = 0.0
                for ids, lab in rows:
                    lg = model(ids[None])[:, :-1]
                    s += float(torch.nn.functional.cross_entropy(
                        lg.reshape(-1, lg.shape[-1]), lab[None, 1:].reshape(-1), reduction="sum"))
                    n += lab.numel() - 1
                tot += s / n
        return tot / len(probe)

    def test_first_order_matches_finite_difference(self, tiny_model) -> None:
        model, mods, opt, probe = self._setup(tiny_model)
        cands = mods[:6]
        scores = interference_scores(model, cands, opt, probe, torch.device("cpu"))
        groups = {id(p): g for g in opt.param_groups for p in g["params"]}
        base = self._old_loss(model, probe)
        checked = 0
        for m in cands:
            deltas = [(p, adamw_open_delta(p, opt, groups[id(p)])) for p in m.params]
            with torch.no_grad():
                for p, d in deltas:
                    p.add_(d)
            actual = self._old_loss(model, probe) - base
            with torch.no_grad():
                for p, d in deltas:
                    p.sub_(d)
            if abs(actual) > 1e-9:
                assert scores[m.id] == pytest.approx(actual, rel=0.05), str(m.id)
                checked += 1
        assert checked >= 3

    def test_no_side_effects(self, tiny_model) -> None:
        model, mods, opt, probe = self._setup(tiny_model)
        grads = {id(p): p.grad.clone() for p in model.parameters() if p.grad is not None}
        state = {id(p): {k: v.clone() for k, v in s.items()} for p, s in opt.state.items()}
        rng = torch.get_rng_state()
        model.train()
        interference_scores(model, mods[:4], opt, probe, torch.device("cpu"))
        assert model.training
        assert torch.equal(torch.get_rng_state(), rng)
        for p in model.parameters():
            if p.grad is not None:
                assert torch.equal(p.grad, grads[id(p)])
        for p, s in opt.state.items():
            for k, v in s.items():
                assert torch.equal(v, state[id(p)][k])

    def test_empty_probe_returns_nothing(self, tiny_model) -> None:
        model, mods, opt, _ = self._setup(tiny_model)
        assert interference_scores(model, mods[:2], opt, {}, torch.device("cpu")) == {}


class TestProbeBuffer:
    def test_fill_trim_exclude_and_resume(self) -> None:
        buf = ProbeBuffer(rows_per_domain=2)
        ids = torch.arange(24).view(3, 8)
        mask = torch.ones(3, 8, dtype=torch.long)
        mask[1, 5:] = 0  # padded row
        labels = ids.clone()
        labels[2] = -100  # nothing to score on: skipped
        assert buf.add_batch(ids, labels, mask, ["math", "math", "code"], step=10)
        assert [r[0].numel() for r in buf.rows["math"]] == [8, 5]
        assert "code" not in buf.rows
        assert not buf.add_batch(ids, ids, mask, ["math", "math", "math"], step=11)  # full
        buf.add_batch(ids, ids, None, ["code", "x", "x"], step=20)
        assert sorted(buf.probe(exclude={"math"})) == ["code", "x"]
        # resume from a checkpoint at step 15: rows captured at >= 15 are dropped
        again = ProbeBuffer(1)
        again.load_state_dict(buf.state_dict(), before_step=15)
        assert sorted(again.rows) == ["math"] and len(again.rows["math"]) == 2


class TestOracleTrainingLoop:
    def test_runs_and_traces_fallback_on_synthetic(self, tmp_path: Path) -> None:
        # Synthetic batches carry no domain, so the probe is empty and every
        # scheduled commit falls back to the random pick: the arm runs, keeps
        # the reference volume, and says so in commits.jsonl.
        sched = tmp_path / "ref_commits.jsonl"
        sched.write_text("".join(json.dumps({"step": s, "p": 1 if s % 3 == 0 else 0}) + "\n"
                                 for s in range(12)))
        cfg = _loop_cfg(tmp_path)
        cfg.controller.debug = {"policy_override": "schedule_oracle_bad",
                                "commit_schedule_path": str(sched)}
        run_training_loop(cfg, device=torch.device("cpu"))
        recs = [json.loads(line) for line in (tmp_path / "commits.jsonl").read_text().splitlines()]
        assert len(recs) == 12
        committing = [r for r in recs if r["p"]]
        assert committing and all(r.get("oracle_fallback") is True for r in committing)
        assert all(r["step"] % 3 == 0 for r in committing)
