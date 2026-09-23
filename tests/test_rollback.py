"""Selective rollback (task T5.4): the policy demotes a blocked teacher's P
actions, a resume from the pre-merge checkpoint resets its dominated
adapters, the sampler honours skip ranges, and the script plans correctly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from test_stream import _FakeDataset

from hivemind.controller.config import DebugConfig, PolicyConfig
from hivemind.controller.module_index import ModuleId
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import ModuleSignals
from hivemind.data.stream import PhaseSpec, StreamBatchSampler, StreamConfig, StreamSchedule
from hivemind.training import run_training_loop

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rollback_teacher as rb  # noqa: E402


def test_policy_demotes_p_for_blocked_teacher() -> None:
    dbg = DebugConfig(block_p_for_teachers=["bad"])
    pol = RFPPolicy(PolicyConfig(), debug=dbg)
    mid = ModuleId(0, "attn", "F")
    sig = ModuleSignals(module_id=mid, grad_norm=1.0, surprise=1.0, repetition=0.9,
                        stability_C=0.9, stability_C_sustained=0.9, stability_V=0.1)
    assert pol.decide({mid: sig}, teacher_name="good")[0].store == "P"
    assert pol.decide({mid: sig}, teacher_name="bad")[0].store == "F"
    forced = RFPPolicy(PolicyConfig(), override="always_p", debug=dbg)
    assert forced.decide({mid: sig}, teacher_name="bad")[0].store == "F"


def _cfg(tmp_path: Path, steps: int, **kw) -> OmegaConf:
    debug = {"policy_override": "always_p", "force_consolidate_steps": [6]}
    debug.update(kw.pop("debug", {}))
    ckpt = {"enabled": True, "dir": str(tmp_path / "ckpt"), "save_every": 100, "keep_tagged": 0}
    ckpt.update(kw.pop("checkpoint", {}))
    return OmegaConf.create(
        {
            "model": {"vocab_size": 64, "dim": 32, "num_layers": 2, "heads": 4,
                      "max_seq_len": 32, "lora": {"rank": 4, "alpha": 8.0}},
            "teachers": {"num_teachers": 1, "mode": "synthetic"},
            "controller": {"consolidation": {"period": 0, "checkpoint_before_merge": True},
                           "debug": debug},
            "data": {"source": "synthetic", "dataset_size": 20, "seq_len": 16,
                     "vocab_size": 64, "batch_size": 4},
            "train": {"steps": steps, "device": "cpu", "seed": 3, "log_interval": 1},
            "optim": {"lr": 1e-3},
            "run": {"dir": str(tmp_path / "run")},
            "checkpoint": ckpt,
            "logging": {"enabled": True, "backend": "json",
                        "path": str(tmp_path / "metrics.jsonl"),
                        "events_path": str(tmp_path / "events.jsonl")},
        }
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_resume_from_pre_merge_resets_dominated_adapters(tmp_path: Path) -> None:
    a = tmp_path / "a"
    a.mkdir()
    run_training_loop(_cfg(a, steps=8), device=torch.device("cpu"))
    events = _events(a / "events.jsonl")
    teacher = next(e["teacher"] for e in events if e["type"] == "decision")
    point = rb.find_rollback_point(events, teacher, 0.5)
    assert point is not None and point["step"] == 6
    pre = a / "ckpt" / "step_00000006_pre_merge"
    assert pre.exists()
    pre_base = torch.load(pre / "base.pt", weights_only=False)

    b = tmp_path / "b"
    b.mkdir()
    cfg_b = _cfg(
        b, steps=8,
        debug={"policy_override": "always_f", "force_consolidate_steps": [],
               "block_p_for_teachers": [teacher]},
        checkpoint={"resume_from": str(pre)},
    )
    run_training_loop(cfg_b, device=torch.device("cpu"))
    ev_b = _events(b / "events.jsonl")
    resets = [e for e in ev_b if e["type"] == "rollback_reset"]
    assert resets
    assert all(e["step"] == 7 and set(e["attribution"]) == {teacher} for e in resets)
    # Exactly the modules the pre-merge checkpoint listed as pending (the
    # Top-M-selected adapters holding the blocked teacher's evidence).
    meta = json.loads((pre / "meta.json").read_text())
    assert {e["module"] for e in resets} == set(meta["extra"]["pending_modules"])
    assert not [e for e in ev_b if e["type"] == "consolidation"]
    assert not [e for e in ev_b if e["type"] == "decision" and e["action"] == "P"]
    # The pending consolidation flags were dropped and the adapters reset
    # (B factors are exactly zero right after reset; step 7 then trains F).
    cons = torch.load(b / "ckpt" / "step_00000007" / "consolidation.pt", weights_only=False)
    assert cons["pending_p"] == []
    # Block base weights never moved after the restore: they equal the
    # pre-merge snapshot (only always_f ran, and nothing was merged).
    final_base = torch.load(b / "ckpt" / "step_00000007" / "base.pt", weights_only=False)
    for key, tensor in pre_base.items():
        # Controller-indexed base weights only: block norms are shared
        # parameters and stay ordinary-AdamW (always open).
        if key.startswith("blocks.") and "norm" not in key:
            assert torch.equal(tensor, final_base[key]), key


def test_sampler_skip_ranges() -> None:
    phases = [PhaseSpec(name="p", steps=10, mode="random", domain="a")]
    sched = StreamSchedule(StreamConfig(enabled=True, phases=phases), _FakeDataset(),
                           batch_size=2, seed=1)
    sampler = StreamBatchSampler(sched, start_step=2)
    sampler.skip_ranges = [(4, 6)]
    yielded = list(sampler)
    assert len(yielded) == len(sampler) == 10 - 2 - 3
    assert [sampler.is_skipped(s) for s in range(10)] == [False] * 4 + [True] * 3 + [False] * 3
    assert yielded[2] == sched.indices_for_step(7)


class TestScriptPlanning:
    def test_find_rollback_point_skips_unattributed_and_respects_min_share(self) -> None:
        events = [
            {"type": "consolidation", "step": 5, "module": "m", "attribution_share": {}},
            {"type": "consolidation", "step": 9, "module": "m",
             "attribution_share": {"bad": 0.3, "good": 0.7}},
            {"type": "consolidation", "step": 12, "module": "n",
             "attribution_share": {"bad": 0.8, "good": 0.2}},
        ]
        assert rb.find_rollback_point(events, "bad", 0.5)["step"] == 12
        assert rb.find_rollback_point(events, "bad", 0.25)["step"] == 9
        assert rb.find_rollback_point(events, "nobody", 0.5) is None

    def test_phase_range_and_corrupted_phase(self) -> None:
        events = [
            {"type": "run_config", "corrupt_teacher": {"enabled": True, "phase": "math"}},
            {"type": "phase_start", "step": 0, "phase": "warm"},
            {"type": "phase_start", "step": 100, "phase": "math"},
            {"type": "phase_start", "step": 250, "phase": "tail"},
        ]
        assert rb.corrupted_phase(events) == "math"
        assert rb.find_phase_range(events, "math") == (100, 249)
        assert rb.find_phase_range(events, "nope") is None

    def test_rewrite_overrides_replaces_run_dir_keys(self, tmp_path: Path) -> None:
        orig = ["controller.retrieval.replay_on_hit=true", "++train.seed=7",
                "++checkpoint.dir=/old/ckpt", "++run.dir=/old", "++logging.path=/old/m.jsonl",
                "++logging.events_path=/old/e.jsonl", "++hydra.run.dir=/old/hydra/rank0"]
        out = rb.rewrite_overrides(orig, tmp_path, ["++checkpoint.resume_from=/x"])
        assert out[:2] == orig[:2]
        assert f"++run.dir={tmp_path}" in out and out[-1] == "++checkpoint.resume_from=/x"
        assert not any(o.startswith("++checkpoint.dir=/old") for o in out)

    def test_plan_selective_and_full_restore(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = tmp_path / "trihope-seed1"
        (run_dir / "checkpoints" / "step_00000012_pre_merge").mkdir(parents=True)
        (run_dir / "checkpoints" / "step_00000099_pre_phase_math").mkdir(parents=True)
        (run_dir / "hydra" / "rank0" / ".hydra").mkdir(parents=True)
        (run_dir / "hydra" / "rank0" / ".hydra" / "overrides.yaml").write_text(
            "- data.corrupt_teacher.enabled=true\n- ++run.dir=/old\n"
        )
        (run_dir / "status.json").write_text(json.dumps(
            {"cmd": "python train.py --config-name stream_small data.corrupt_teacher.enabled=true"}
        ))
        events = [
            {"type": "run_config", "corrupt_teacher": {"enabled": True, "phase": "math"}},
            {"type": "phase_start", "step": 0, "phase": "warm"},
            {"type": "phase_start", "step": 100, "phase": "math"},
            {"type": "phase_start", "step": 250, "phase": "tail"},
            {"type": "consolidation", "step": 12, "module": "L0.ffn.F",
             "attribution_share": {"bad": 0.9}},
        ]
        (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")

        info = rb.plan(run_dir, "bad", tmp_path / "out")
        assert info["mode"] == "selective" and info["restore_step"] == 12
        cmd = info["cmd"]
        assert cmd[cmd.index("--config-name") + 1] == "stream_small"
        assert "data.corrupt_teacher.enabled=true" in cmd
        assert "++controller.debug.block_p_for_teachers=[bad]" in cmd
        assert "++controller.debug.block_min_share=0.5" in cmd
        assert any(o.startswith("++checkpoint.resume_from=") and o.endswith("_pre_merge")
                   for o in cmd)

        info = rb.plan(run_dir, "bad", tmp_path / "out2", baseline="full_restore")
        assert info["mode"] == "full_restore" and info["skip_range"] == [100, 249]
        assert "++train.skip_step_ranges=[[100,249]]" in info["cmd"]
        # Containment holding is a result, not an error (0b6bdf3): plan()
        # reports it and returns None so run_campaign.sh keeps going.
        assert rb.plan(run_dir, "nobody", tmp_path / "out3") is None
        assert "nothing to roll back" in capsys.readouterr().out

    def test_write_summary(self, tmp_path: Path) -> None:
        run_dir, out = tmp_path / "r", tmp_path / "o"
        run_dir.mkdir()
        out.mkdir()
        hist_a = [{"phase": "x", "loss": {"code": 1.0, "math": 2.0},
                   "retention_delta": {"code": 0.5}}]
        hist_b = [{"phase": "x", "loss": {"code": 0.8, "math": 2.5},
                   "retention_delta": {"code": 0.1}}]
        (run_dir / "run_summary.json").write_text(json.dumps({"retention": {"history": hist_a}}))
        (out / "run_summary.json").write_text(json.dumps({"retention": {"history": hist_b}}))
        path = rb.write_summary({"mode": "selective", "cmd": ["x"]}, run_dir, out)
        payload = json.loads(path.read_text())
        assert payload["final_loss_delta"] == {
            "code": pytest.approx(-0.2), "math": pytest.approx(0.5)}
        assert payload["rollback"]["worst_retention_delta"] == 0.1 and "cmd" not in payload
