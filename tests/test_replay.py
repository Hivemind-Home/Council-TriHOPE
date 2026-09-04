"""R-store replay (task T2): a deferred observation is replayed into the
fast store, inside the same step, once its bucket recurs.

The synthetic tiny config below drives the R→F transition purely through
the bucket-frequency term: with one teacher every batch shares bucket 0,
``bucket_smoothing_k=0.5`` makes the fused repetition 1/3 on the first
visit (→ R) and 0.4 on the second (→ not < repetition_low=0.4 → F), and the
momentum/retrieval components are disabled so nothing else moves it.
Surprise stays at ``s_max`` because R-routed modules never open (v stays 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from hivemind.stores.retrieval import RetrievalEntry, RetrievalStore
from hivemind.training import _capture_replay_sample, run_training_loop

# --- store -----------------------------------------------------------------


def _entry(step: int, bucket: int = 7, with_sample: bool = True) -> RetrievalEntry:
    return RetrievalEntry(
        embedding=torch.ones(4),
        teacher_id=0,
        bucket_id=bucket,
        step=step,
        input_ids=torch.arange(6) + step if with_sample else None,
        labels=torch.arange(6) if with_sample else None,
        kd_row_mask=0.0 if with_sample else None,
        teacher_confidence=0.9 if with_sample else None,
    )


class TestRetrievalStoreReplayApi:
    def test_step_keyed_hits_and_pop(self) -> None:
        store = RetrievalStore(max_size=100)
        # The writer appends one entry per R action: six identical entries
        # per step must count as ONE recurrence.
        for _ in range(6):
            store.add(_entry(3))
        for _ in range(6):
            store.add(_entry(5))
        store.add(_entry(4, bucket=9))
        assert store.distinct_steps_for_bucket(7) == [3, 5]
        assert store.distinct_steps_for_bucket(9) == [4]

        popped = store.pop_for_bucket(7, 1)
        assert [e.step for e in popped] == [3]
        assert store.size == 7  # six step-3 entries removed, step-5 + bucket-9 stay
        assert store.distinct_steps_for_bucket(7) == [5]
        assert store.replayed_total == 1

        assert store.pop_for_bucket(7, 5) and store.distinct_steps_for_bucket(7) == []
        assert store.pop_for_bucket(7, 1) == []

    def test_eviction_never_leaves_stale_hits(self) -> None:
        store = RetrievalStore(max_size=3)
        for step in range(5):
            store.add(_entry(step))
        # deque(maxlen=3) silently dropped steps 0 and 1.
        assert store.distinct_steps_for_bucket(7) == [2, 3, 4]
        popped = store.pop_for_bucket(7, 1)
        assert popped[0].step == 2 and store.size == 2

    def test_state_dict_roundtrip_with_sample_tensors(self) -> None:
        store = RetrievalStore(max_size=10)
        store.add(_entry(1))
        store.add(_entry(2, with_sample=False))
        store.pop_for_bucket(7, 1)
        state = store.state_dict()

        fresh = RetrievalStore(max_size=10)
        fresh.load_state_dict(state)
        assert fresh.replayed_total == 1
        (e,) = fresh.entries_for_bucket(7)
        assert e.step == 2 and e.input_ids is None and e.kd_row_mask is None

        store2 = RetrievalStore(max_size=10)
        store2.add(_entry(1))
        fresh2 = RetrievalStore(max_size=10)
        fresh2.load_state_dict(store2.state_dict())
        (e2,) = fresh2.entries_for_bucket(7)
        assert torch.equal(e2.input_ids, torch.arange(6) + 1)
        assert torch.equal(e2.labels, torch.arange(6))
        assert e2.kd_row_mask == 0.0 and e2.teacher_confidence == 0.9

    def test_old_format_entries_load(self) -> None:
        legacy = {
            "max_size": 10,
            "entries": [
                {
                    "embedding": torch.ones(4),
                    "teacher_id": 0,
                    "bucket_id": 1,
                    "step": 0,
                }
            ],
        }
        store = RetrievalStore(max_size=10)
        store.load_state_dict(legacy)
        assert store.size == 1 and store.replayed_total == 0
        assert store.entries_for_bucket(1)[0].input_ids is None

    def test_entries_compare_by_identity(self) -> None:
        a, b = _entry(1), _entry(1)
        assert a != b and a == a  # no tensor __eq__ blow-up


# --- sample capture ----------------------------------------------------------


class _NoDist:
    enabled = False
    is_main = True


def test_capture_sample_kd_mask_semantics() -> None:
    dev = torch.device("cpu")
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    synthetic = {"input_ids": ids, "labels": ids, "teacher_logits": None}
    s = _capture_replay_sample(synthetic, _NoDist(), dev)
    assert s["kd_row_mask"] is None and s["teacher_confidence"] is None
    assert torch.equal(s["input_ids"], ids[0])

    hf = {
        "input_ids": ids,
        "labels": ids,
        "teacher_logits": torch.zeros(2, 3, 1),
        "teacher_confidence": torch.tensor([0.7, 0.2]),
    }
    s = _capture_replay_sample(hf, _NoDist(), dev)
    assert s["kd_row_mask"] == 0.0 and s["teacher_confidence"] == pytest.approx(0.7)


# --- end-to-end --------------------------------------------------------------


def _cfg(tmp_path: Path, *, replay: bool | None, steps: int = 6) -> OmegaConf:
    controller: dict = {
        "policy": {
            "top_m_modules": 4,
            "surprise_high": 2.0,
            "repetition_low": 0.4,
            "repetition_medium": 0.9,
            "stability_high_C": 0.99,
        },
        "repetition": {"bucket_smoothing_k": 0.5},
        "ablation": {"disable_signals": ["repetition_mom", "repetition_ret"]},
        "consolidation": {"period": 0},
    }
    if replay is not None:
        controller["retrieval"] = {"replay_on_hit": replay, "hit_threshold": 1}
    return OmegaConf.create(
        {
            "model": {
                "vocab_size": 64, "dim": 32, "num_layers": 2, "heads": 4,
                "max_seq_len": 32, "lora": {"rank": 4, "alpha": 8.0},
            },
            "teachers": {"num_teachers": 1, "mode": "synthetic"},
            "controller": controller,
            "data": {
                "source": "synthetic", "dataset_size": 40, "seq_len": 16,
                "vocab_size": 64, "batch_size": 4,
            },
            "train": {"steps": steps, "device": "cpu", "seed": 5, "log_interval": 1},
            "optim": {"lr": 1e-3},
            "run": {"dir": str(tmp_path / "run")},
            "logging": {
                "enabled": True, "backend": "json",
                "path": str(tmp_path / "metrics.jsonl"),
                "events_path": str(tmp_path / "events.jsonl"),
            },
        }
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _losses(path: Path) -> list[float]:
    out = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if "loss/total" in rec:
            out.append(rec["loss/total"])
    return out


def test_replay_fires_within_the_step_and_stream_is_not_lost(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, replay=True)
    run_training_loop(cfg, device=torch.device("cpu"))
    events = _events(tmp_path / "events.jsonl")

    decisions = [e for e in events if e["type"] == "decision"]
    replays = [e for e in events if e["type"] == "replay"]
    assert {e["action"] for e in decisions if e["step"] == 0} == {"R"}
    assert replays, "no replay fired"
    # Step 1 routes the bucket to F, so step 0's parked row is replayed at step 1.
    assert {e["step"] for e in replays} == {1}
    assert all(e["origin_step"] == 0 and e["action"] == "F" for e in replays)
    assert all(e["coords_opened"] > 0 for e in replays)
    # decision records stay one-per-Top-M-module-per-step; no stream step is skipped.
    per_step = {}
    for e in decisions:
        per_step[e["step"]] = per_step.get(e["step"], 0) + 1
    assert sorted(per_step) == list(range(cfg.train.steps))
    assert max(per_step.values()) <= 4
    # The parked entry was removed, so it is replayed exactly once.
    assert len({(e["origin_step"], e["module"]) for e in replays}) == len(replays)

    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text())
    assert summary["extra"]["retrieval"]["replayed_total"] == 1
    metrics = [json.loads(ln) for ln in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert sum(m.get("write/replay_count", 0) for m in metrics) == 1


def test_replay_off_is_bit_identical_to_a_config_without_the_key(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    run_training_loop(_cfg(a, replay=False), device=torch.device("cpu"))
    run_training_loop(_cfg(b, replay=None), device=torch.device("cpu"))
    la, lb = _losses(a / "metrics.jsonl"), _losses(b / "metrics.jsonl")
    assert la and la == lb
    assert not [e for e in _events(a / "events.jsonl") if e["type"] == "replay"]


def test_replay_changes_the_trajectory(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    run_training_loop(_cfg(a, replay=True), device=torch.device("cpu"))
    run_training_loop(_cfg(b, replay=False), device=torch.device("cpu"))
    la, lb = _losses(a / "metrics.jsonl"), _losses(b / "metrics.jsonl")
    assert la[:2] == lb[:2]  # nothing differs until the replay at step 1 has landed
    assert la[2:] != lb[2:]
