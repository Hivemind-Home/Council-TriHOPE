"""E1 baseline controllers (task T3): surprise-only, Adam-score, budget-matched
random routing, and the loss-plateau merge trigger."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from hivemind.controller.config import (
    AblationConfig,
    ConsolidationConfig,
    DebugConfig,
    PolicyConfig,
    WriterConfig,
)
from hivemind.controller.consolidation import PlateauDetector
from hivemind.controller.module_index import ModuleId
from hivemind.controller.policy import RFPPolicy, load_random_shares
from hivemind.controller.signals import ModuleSignals
from hivemind.training import run_training_loop


def _sig(i: int, **kw) -> ModuleSignals:
    base = dict(grad_norm=1.0 + i, surprise=1.0, repetition=0.5, stability_C=0.5,
                stability_C_sustained=0.5, stability_V=0.3, stability_adam=0.1)
    base.update(kw)
    return ModuleSignals(module_id=ModuleId(i, "attn", "F"), **base)


_GRID = [
    dict(surprise=s, repetition=r, stability_C=c, stability_C_sustained=c,
         stability_V=v, stability_adam=a)
    for s in (0.5, 5.0)
    for r in (0.1, 0.9)
    for c in (0.1, 0.9)
    for v in (0.1, 0.9)
    for a in (0.1, 0.9)
]


def _labels(policy: RFPPolicy) -> set[str]:
    out = set()
    for kw in _GRID:
        sig = _sig(0, **kw)
        out |= {a.store for a in policy.decide({sig.module_id: sig})}
    return out


class TestModes:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="policy.mode"):
            PolicyConfig(mode="bogus")

    def test_surprise_only_labels(self) -> None:
        pol = RFPPolicy(PolicyConfig(mode="surprise_only", surprise_high=2.0))
        assert _labels(pol) == {"R", "F"}
        sig = _sig(0, surprise=5.0, repetition=0.9, stability_C=0.9, stability_V=0.1)
        assert pol.decide({sig.module_id: sig})[0].store == "R"  # would be P under rfp
        pol_no_r = RFPPolicy(
            PolicyConfig(mode="surprise_only"), ablation=AblationConfig(disable_stores=["R"])
        )
        assert _labels(pol_no_r) == {"F"}

    def test_adam_score_labels(self) -> None:
        pol = RFPPolicy(PolicyConfig(mode="adam_score", adam_score_high=0.5,
                                     adam_score_rule="snr_threshold"))
        assert _labels(pol) == {"P", "F"}
        hi = _sig(0, stability_adam=0.9, surprise=5.0, repetition=0.0)
        lo = _sig(0, stability_adam=0.1, surprise=5.0, repetition=0.0)
        assert pol.decide({hi.module_id: hi})[0].store == "P"
        assert pol.decide({lo.module_id: lo})[0].store == "F"

    def test_epd_argmax_routes_each_block_to_one_expert(self) -> None:
        pol = RFPPolicy(PolicyConfig(mode="adam_score", top_m_modules=1))
        f0 = ModuleSignals(module_id=ModuleId(0, "attn", "F"), grad_norm=1.0, epd_score=0.2)
        p0 = ModuleSignals(module_id=ModuleId(0, "attn", "P"), grad_norm=1.0, epd_score=0.5)
        f1 = ModuleSignals(module_id=ModuleId(1, "ffn", "F"), grad_norm=1.0, epd_score=0.9)
        p1 = ModuleSignals(module_id=ModuleId(1, "ffn", "P"), grad_norm=1.0, epd_score=0.1)
        sigs = {s.module_id: s for s in (f0, p0, f1, p1)}
        acts = {str(a.module_id): a.store for a in pol.decide(sigs)}
        # every block routes (top_m ignored); exactly one expert per block
        assert acts == {"L0.attn.P": "P", "L1.ffn.F": "F"}
        # the learning-rate ratio is the knob: a cheap LoRA lr flips block 0
        pol2 = RFPPolicy(PolicyConfig(mode="adam_score", epd_lr_lora=10.0))
        assert {str(a.module_id): a.store for a in pol2.decide(sigs)} == {
            "L0.attn.F": "F", "L1.ffn.F": "F"}
        # blocked teacher → the LoRA expert takes the step
        pol3 = RFPPolicy(PolicyConfig(mode="adam_score"),
                         debug=DebugConfig(block_p_for_teachers=["bad"]))
        assert {str(a.module_id): a.store for a in pol3.decide(sigs, teacher_name="bad")} == {
            "L0.attn.F": "F", "L1.ffn.F": "F"}
        with pytest.raises(ValueError, match="adam_score_rule"):
            PolicyConfig(adam_score_rule="x")

    def test_rfp_default_unchanged(self) -> None:
        pol = RFPPolicy(PolicyConfig())
        assert _labels(pol) == {"R", "F", "P"}


class TestAdamScoreNeverFlagsConsolidation:
    def test_writer_flag_off(self, tiny_model) -> None:
        from hivemind.controller.consolidation import ConsolidationScheduler
        from hivemind.controller.module_index import build_module_index
        from hivemind.controller.policy import StoreAction
        from hivemind.controller.writer import WriteExecutor
        from hivemind.optim.masked_adamw import MaskedAdamW
        from hivemind.stores.fast import FastStore
        from hivemind.stores.permanent import PermanentStore
        from hivemind.stores.retrieval import RetrievalStore

        opt = MaskedAdamW(tiny_model.parameters(), lr=1e-3)
        index = build_module_index(tiny_model)
        module_map = {m.id: m for m in index}
        p_store = PermanentStore()
        cons = ConsolidationScheduler(ConsolidationConfig(period=0), tiny_model, p_store, opt)
        f_mid = next(m.id for m in index if m.id.param_type == "F")

        flagging = WriteExecutor(tiny_model, opt, RetrievalStore(), FastStore(), p_store,
                                 WriterConfig(), consolidator=cons)
        flagging.execute([StoreAction(f_mid, "P")], module_map, step=0)
        assert cons.pending_p == {f_mid}

        cons2 = ConsolidationScheduler(ConsolidationConfig(period=0), tiny_model, p_store, opt)
        molf = WriteExecutor(tiny_model, opt, RetrievalStore(), FastStore(), p_store,
                             WriterConfig(flag_p_for_consolidation=False), consolidator=cons2)
        metrics, masks = molf.execute([StoreAction(f_mid, "P")], module_map, step=0)
        assert cons2.pending_p == set()
        assert metrics["coords_opened"] > 0 and masks  # the adapter still opened


def _shares_csv(path: Path) -> None:
    rows = [
        # a second spec must be ignored
        {"spec_id": "full_ft", "phase": "warm", "r_share_mean": 0, "f_share_mean": 0,
         "p_share_mean": 100},
        {"spec_id": "trihope", "phase": "warm", "r_share_mean": 70.0, "r_share_std": 1,
         "f_share_mean": 30.0, "f_share_std": 1, "p_share_mean": 0.0, "p_share_std": 0},
        {"spec_id": "trihope", "phase": "recurrent", "r_share_mean": 10.0, "r_share_std": 1,
         "f_share_mean": 60.0, "f_share_std": 1, "p_share_mean": 30.0, "p_share_std": 0},
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestRandomMatched:
    def test_debug_config_accepts_random_matched(self) -> None:
        DebugConfig(policy_override="random_matched")
        with pytest.raises(ValueError, match="random_unit"):
            DebugConfig(random_unit="x")

    def test_frequencies_match_shares_per_phase(self, tmp_path: Path) -> None:
        _shares_csv(tmp_path / "shares.csv")
        shares = load_random_shares(tmp_path / "shares.csv", "trihope")
        assert shares["warm"] == pytest.approx((0.7, 0.3, 0.0))
        dbg = DebugConfig(policy_override="random_matched",
                          random_shares_path=str(tmp_path / "shares.csv"))
        pol = RFPPolicy(PolicyConfig(), override="random_matched", debug=dbg, seed=11)
        pol2 = RFPPolicy(PolicyConfig(), override="random_matched", debug=dbg, seed=11)
        rng_before = torch.get_rng_state()
        for phase, expected in (("warm", (0.7, 0.3, 0.0)), ("recurrent", (0.1, 0.6, 0.3))):
            counts = {"R": 0, "F": 0, "P": 0}
            for step in range(2000):
                sig = _sig(0)
                (a,) = pol.decide({sig.module_id: sig}, step=step, phase=phase)
                (b,) = pol2.decide({sig.module_id: sig}, step=step, phase=phase)
                assert a.store == b.store
                counts[a.store] += 1
            for label, p in zip("RFP", expected):
                assert abs(counts[label] / 2000 - p) < 0.03, (phase, label, counts)
        assert torch.equal(rng_before, torch.get_rng_state())

    def test_missing_file_is_uniform_with_warning(self) -> None:
        dbg = DebugConfig(policy_override="random_matched", random_shares_path="/nope.csv")
        with pytest.warns(UserWarning, match="not found"):
            pol = RFPPolicy(PolicyConfig(), override="random_matched", debug=dbg, seed=1)
        counts = {"R": 0, "F": 0, "P": 0}
        for step in range(3000):
            sig = _sig(0)
            counts[pol.decide({sig.module_id: sig}, step=step, phase=None)[0].store] += 1
        for label in "RFP":
            assert abs(counts[label] / 3000 - 1 / 3) < 0.03

    def test_disabled_stores_are_respected_and_step_unit_shares_one_draw(self) -> None:
        dbg = DebugConfig(policy_override="random_matched", random_unit="step")
        with pytest.warns(UserWarning):
            pol = RFPPolicy(PolicyConfig(top_m_modules=3), override="random_matched",
                            debug=dbg, seed=3, ablation=AblationConfig(disable_stores=["P"]))
        seen = set()
        for step in range(200):
            sigs = {s.module_id: s for s in (_sig(0), _sig(1), _sig(2))}
            acts = pol.decide(sigs, step=step, phase=None)
            labels = {a.store for a in acts}
            assert len(labels) == 1  # one draw per step
            seen |= labels
        assert seen == {"R", "F"}


class TestPlateau:
    def test_flat_series_fires_once_per_gap_and_decreasing_never(self) -> None:
        det = PlateauDetector(window=10, tolerance=0.01, min_gap=20)
        fires = [step for step in range(60) if det.observe(1.0, step)]
        assert fires == [9, 29, 49]
        det2 = PlateauDetector(window=10, tolerance=0.01, min_gap=0)
        assert not any(det2.observe(2.0 - 0.05 * step, step) for step in range(100))

    def test_require_peak_blocks_the_initial_flat_stretch(self) -> None:
        """Online-LoRA's precondition: a plateau counts only after a loss peak."""
        det = PlateauDetector(window=6, tolerance=0.01, min_gap=0, require_peak=True)
        assert not any(det.observe(1.0, s) for s in range(30))  # flat from the start
        # a jump (peak) ...
        fires = [s for s in range(30, 45) if det.observe(3.0 if s >= 32 else 1.0, s)]
        assert fires and fires[0] > 32
        # ... after the fire the peak flag is cleared: flat again → no refire
        assert not any(det.observe(3.0, s) for s in range(45, 60))
        rt = PlateauDetector(window=6, tolerance=0.01, min_gap=0, require_peak=True)
        rt.load_state_dict(det.state_dict())
        assert rt.state_dict() == det.state_dict()

    def test_state_roundtrip_continues_the_window(self) -> None:
        det = PlateauDetector(window=6, tolerance=0.01, min_gap=0)
        for step in range(4):
            det.observe(1.0, step)
        fresh = PlateauDetector(window=6, tolerance=0.01, min_gap=0)
        fresh.load_state_dict(det.state_dict())
        assert not fresh.observe(1.0, 4)
        assert fresh.observe(1.0, 5)

    def test_invalid_trigger_raises(self) -> None:
        with pytest.raises(ValueError, match="trigger"):
            ConsolidationConfig(trigger="bogus")

    def test_signal_sweep_disabled_under_plateau(self) -> None:
        from hivemind.controller.consolidation import ConsolidationScheduler

        cfg = ConsolidationConfig(trigger="plateau", period=10)
        sched = ConsolidationScheduler(cfg, torch.nn.Linear(2, 2), None)
        assert not sched.should_check(10)

    def test_plateau_trigger_with_controller_off(self, tmp_path: Path) -> None:
        cfg = OmegaConf.create(
            {
                "model": {"vocab_size": 64, "dim": 32, "num_layers": 2, "heads": 4,
                          "max_seq_len": 32, "lora": {"rank": 4, "alpha": 8.0}},
                "teachers": {"num_teachers": 2, "mode": "synthetic"},
                "controller": {
                    "enabled": False,
                    "consolidation": {"trigger": "plateau", "plateau_window": 4,
                                      "plateau_tolerance": 10.0, "plateau_min_gap": 3},
                },
                "data": {"source": "synthetic", "dataset_size": 20, "seq_len": 16,
                         "vocab_size": 64, "batch_size": 4},
                "train": {"steps": 10, "device": "cpu", "seed": 1, "log_interval": 1,
                          "trainable": "lora"},
                "optim": {"lr": 1e-2},
                "logging": {"enabled": True, "backend": "json",
                            "path": str(tmp_path / "metrics.jsonl"),
                            "events_path": str(tmp_path / "events.jsonl")},
                "checkpoint": {"enabled": True, "dir": str(tmp_path / "ckpt"),
                               "save_every": 100},
            }
        )
        run_training_loop(cfg, device=torch.device("cpu"))
        events = [json.loads(ln) for ln in (tmp_path / "events.jsonl").read_text().splitlines()]
        merges = [e for e in events if e["type"] == "consolidation"]
        assert merges and all(e["trigger"] == "plateau" and e["forced"] is False for e in merges)
        # tolerance=10 → fires as soon as the window fills (step 3); the window is
        # cleared on a fire, so the next fire needs four more losses (step 7).
        assert sorted({e["step"] for e in merges}) == [3, 7]
        # Every adapter block (2 layers × attn/ffn) is merged on each fire, and
        # the ledger agrees with the trace.
        assert len(merges) == 2 * 4
        ledger = torch.load(tmp_path / "ckpt" / "step_00000009" / "ledger.pt",
                            weights_only=False)
        assert sum(m["consolidations"] for m in ledger["modules"].values()) == len(merges)
        cons = torch.load(tmp_path / "ckpt" / "step_00000009" / "consolidation.pt",
                          weights_only=False)
        assert cons["plateau"]["last_fire"] == 7


class TestRandomCommit:
    """Commit-selection control: the policy's own defers, random commits."""

    def test_config_accepts_random_commit(self) -> None:
        DebugConfig(policy_override="random_commit")

    def test_defers_match_policy_and_commit_share_matches_reference(self, tmp_path: Path) -> None:
        _shares_csv(tmp_path / "shares.csv")
        dbg = DebugConfig(policy_override="random_commit",
                          random_shares_path=str(tmp_path / "shares.csv"))
        ref = RFPPolicy(PolicyConfig())
        pol = RFPPolicy(PolicyConfig(), override="random_commit", debug=dbg, seed=5)
        pol2 = RFPPolicy(PolicyConfig(), override="random_commit", debug=dbg, seed=5)
        # defers are exactly the policy's own, across the whole signal grid
        for step in range(50):
            for kw in _GRID:
                sig = _sig(0, **kw)
                (want,) = ref.decide({sig.module_id: sig})
                (got,) = pol.decide({sig.module_id: sig}, step=step, phase="recurrent")
                (again,) = pol2.decide({sig.module_id: sig}, step=step, phase="recurrent")
                assert got.store == again.store  # deterministic given (seed, step)
                assert (got.store == "R") == (want.store == "R")
        # "recurrent": p_R .1, p_F .6, p_P .3 -> commit prob among non-R = .3/.9
        sig = _sig(0)  # a non-deferred block (the policy itself would commit it)
        commits = sum(
            pol.decide({sig.module_id: sig}, step=step, phase="recurrent")[0].store == "P"
            for step in range(4000)
        )
        assert commits / 4000 == pytest.approx(0.3 / 0.9, abs=0.025)

    def test_no_commits_without_shares_or_with_P_disabled(self, tmp_path: Path) -> None:
        _shares_csv(tmp_path / "shares.csv")
        dbg = DebugConfig(policy_override="random_commit",
                          random_shares_path=str(tmp_path / "shares.csv"))
        warm = RFPPolicy(PolicyConfig(), override="random_commit", debug=dbg, seed=1)
        nop = RFPPolicy(PolicyConfig(), AblationConfig(disable_stores=["P"]),
                        override="random_commit", debug=dbg, seed=1)
        for step in range(200):
            for kw in _GRID:
                sig = _sig(0, **kw)
                # "warm" has p_P = 0: never commit, even where the policy would
                (a,) = warm.decide({sig.module_id: sig}, step=step, phase="warm")
                assert a.store != "P"
                (b,) = nop.decide({sig.module_id: sig}, step=step, phase="recurrent")
                assert b.store != "P"
