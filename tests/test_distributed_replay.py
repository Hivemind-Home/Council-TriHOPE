"""The DDP-only paths added for replay and attribution, on gloo/CPU:
``DistContext.broadcast_int`` and the replay-row broadcast in
``_capture_replay_sample`` (rank 0's row, even when the ranks' shards are
padded to different lengths)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with gloo is required",
)


def _run(rank: int, world_size: int, out_dir: str) -> None:
    from hivemind.distributed import DistConfig, DistContext
    from hivemind.training import _capture_replay_sample

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    ctx = DistContext(
        cfg=DistConfig(enabled=True, backend="gloo"), rank=rank, world_size=world_size
    )
    # Each rank pads its shard to its own length: rank 0 → T=5, rank 1 → T=7.
    T = 5 + 2 * rank
    ids = torch.arange(T).unsqueeze(0) + 100 * (rank + 1)
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "teacher_logits": torch.zeros(1, T, 1),
        "teacher_confidence": torch.tensor([0.25 + rank]),
    }
    sample = _capture_replay_sample(batch, ctx, torch.device("cpu"))
    synced = ctx.broadcast_int(10 + rank)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(
            {
                "input_ids": sample["input_ids"].tolist(),
                "labels": sample["labels"].tolist(),
                "kd": sample["kd_row_mask"],
                "conf": sample["teacher_confidence"],
                "teacher": synced,
            },
            fh,
        )
    dist.barrier()
    dist.destroy_process_group()


def test_every_rank_gets_rank_zeros_row_and_teacher() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as out:
        os.environ["MASTER_PORT"] = str(29700)
        mp.spawn(_run, args=(world_size, out), nprocs=world_size, join=True)
        results = [json.load(open(os.path.join(out, f"rank{r}.json"))) for r in range(world_size)]
    expected_ids = (torch.arange(5) + 100).tolist()
    for r in results:
        assert r["input_ids"] == expected_ids
        assert r["labels"] == expected_ids
        assert r["kd"] == 0.0
        assert r["conf"] == pytest.approx(0.25)
        assert r["teacher"] == 10
