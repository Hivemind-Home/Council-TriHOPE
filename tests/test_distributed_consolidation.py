"""Consolidation under DDP: no hang, and identical weights afterwards.

Merging is weight surgery performed *outside* the optimizer, so it needs
its own synchronization:

* ``reset_lora`` draws A from ``nn.init.kaiming_uniform_`` off the global
  torch RNG. Ranks agree today only because they share a seed and every
  shipped config runs dropout=0 — a latent invariant, not a guarantee.
* the ``distill`` strategy runs an inner forward/backward/step against
  ``self._replay[-1]``, which is the rank's own shard.
* that inner backward runs on the unwrapped module, so DDP's reducer
  early-returns. Correct today, version-dependent tomorrow — hence the
  timeout on the distill test rather than a plain assertion.
"""

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

VOCAB, SEQ, GLOBAL_BATCH = 32, 8, 4


def _run(rank: int, world_size: int, out_path: str, strategy: str) -> None:
    from hivemind.controller.config import ConsolidationConfig
    from hivemind.controller.consolidation import ConsolidationScheduler
    from hivemind.controller.module_index import ModuleId
    from hivemind.distributed import DistConfig, DistContext
    from hivemind.optim.masked_adamw import MaskedAdamW
    from hivemind.stores.permanent import PermanentStore
    from hivemind.student.config import LoRAConfig, StudentConfig
    from hivemind.student.model import StudentModel

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    ctx = DistContext(
        cfg=DistConfig(enabled=world_size > 1, backend="gloo"),
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(99)
    student = StudentModel(
        StudentConfig(
            vocab_size=VOCAB, dim=16, num_layers=2, heads=2, max_seq_len=SEQ,
            dropout=0.0,
            lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o"]),
        )
    )
    opt = MaskedAdamW(student.parameters(), lr=1e-2)
    model_fwd = ctx.wrap(student)

    cons = ConsolidationScheduler(
        ConsolidationConfig(
            merge_strategy=strategy, distill_iters=2, distill_replay_size=4
        ),
        student,
        PermanentStore(),
        optimizer=opt,
        dist=ctx,
    )
    cons._ddp_handle = model_fwd if model_fwd is not student else None

    # Train briefly so the adapters carry something, and so each rank's
    # replay buffer holds a DIFFERENT shard (which is the thing that must
    # not leak into the merge).
    g = torch.Generator().manual_seed(rank + 1)
    for _ in range(3):
        tokens = torch.randint(0, VOCAB, (GLOBAL_BATCH // world_size, SEQ), generator=g)
        cons.record_batch(tokens)
        opt.zero_grad()
        logits = model_fwd(tokens)
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1)
        ).backward()
        opt.step()

    merged = cons.force_consolidate([ModuleId(0, "attn", "F")])

    if rank == 0 or world_size == 1:
        flat = torch.cat(
            [v.detach().reshape(-1).double() for v in student.state_dict().values()]
        )
        payload = {"merged": [str(m) for m in merged], "weights": flat.tolist()}
    else:
        payload = None

    # Every rank reports, so a per-rank difference is visible.
    gathered = [None] * world_size
    if world_size > 1:
        flat = torch.cat(
            [v.detach().reshape(-1).double() for v in student.state_dict().values()]
        )
        dist.all_gather_object(gathered, flat.tolist())
    if rank == 0:
        with open(out_path, "w") as fh:
            json.dump({**(payload or {}), "per_rank": gathered}, fh)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def _collect(world_size: int, strategy: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "o.json")
        if world_size == 1:
            _run(0, 1, out, strategy)
        else:
            os.environ["MASTER_PORT"] = str(29800 + world_size + len(strategy))
            mp.spawn(_run, args=(world_size, out, strategy), nprocs=world_size, join=True)
        with open(out) as fh:
            return json.load(fh)


class TestDirectMerge:
    def test_merge_happens_on_every_rank(self):
        res = _collect(2, "direct")
        assert res["merged"] == ["L0.attn.F"]

    def test_weights_identical_across_ranks_after_merge(self):
        """Catches a per-rank reset_lora draw producing different adapters."""
        res = _collect(2, "direct")
        a, b = torch.tensor(res["per_rank"][0]), torch.tensor(res["per_rank"][1])
        assert torch.equal(a, b), (
            f"max |Δ| = {(a - b).abs().max().item():.3e} between ranks after a merge"
        )


class TestDistillMerge:
    def test_distill_strategy_does_not_hang(self):
        """The inner forward/backward must not engage DDP's reducer.

        If a future torch release stops early-returning from
        ``Reducer::autograd_hook`` for a module that never went through
        DDP's forward, this deadlocks — which is why it is pinned here
        rather than left to surface at hour 4 of a headline run.
        """
        res = _collect(2, "distill")
        assert res["merged"] == ["L0.attn.F"]

    def test_distill_weights_identical_across_ranks(self):
        """The replay batch must be rank 0's, not each rank's own shard.

        Each rank was fed a different generator seed above, so without the
        broadcast in ``_pick_replay`` the inner AdamW would train each
        replica's base weights on different data.
        """
        res = _collect(2, "distill")
        a, b = torch.tensor(res["per_rank"][0]), torch.tensor(res["per_rank"][1])
        assert torch.equal(a, b), (
            f"max |Δ| = {(a - b).abs().max().item():.3e} between ranks after a "
            "distill merge — the replay batch is not shared"
        )
