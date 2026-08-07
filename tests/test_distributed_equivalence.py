"""1 rank must equal N ranks: identical routing decisions, identical weights.

This is the load-bearing test for the whole multi-GPU design. DDP
synchronizes gradients but never parameters after construction, so if the
controller's three rank-local inputs (bucket_id, embedding, conf_mean) are
not made global, each replica opens *different* coordinates, their
coordinate-local update counters diverge, and the replicas drift apart
silently — no exception, no NaN, a run that finishes looking healthy.

Runs on gloo/CPU with mp.spawn so it is exercised without any GPU.

A caveat worth stating: KD and CE normalize by the *local* contributing
token count (distillation.py), so mean-of-per-rank-gradients equals the
global gradient only when every rank contributes the same number of
tokens. The synthetic rows here are fixed-length so that holds exactly;
with ragged real batches the N-GPU objective is a slightly different
weighting, which is ordinary DDP semantics rather than a bug.
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

STEPS = 40
GLOBAL_BATCH = 4
SEQ = 8
VOCAB = 32
NUM_BUCKETS = 2


# -- a tiny deterministic stand-in for the real pipeline ----------------------


def _make_batches(seed: int = 0):
    """Fixed global batches: (indices, bucket_id, per-row confidence)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in range(STEPS):
        tokens = torch.randint(0, VOCAB, (GLOBAL_BATCH, SEQ), generator=g)
        conf = torch.rand(GLOBAL_BATCH, generator=g) * 0.5 + 0.5
        out.append((tokens, t % NUM_BUCKETS, conf))
    return out


def _build(seed: int):
    from hivemind.controller.config import ControllerConfig
    from hivemind.controller.module_index import build_module_index
    from hivemind.controller.policy import RFPPolicy
    from hivemind.controller.signals import SignalComputer
    from hivemind.controller.writer import WriteExecutor
    from hivemind.optim.masked_adamw import MaskedAdamW
    from hivemind.stores.fast import FastStore
    from hivemind.stores.permanent import PermanentStore
    from hivemind.stores.retrieval import RetrievalStore
    from hivemind.student.config import LoRAConfig, StudentConfig
    from hivemind.student.model import StudentModel

    torch.manual_seed(seed)
    student = StudentModel(
        StudentConfig(
            vocab_size=VOCAB, dim=16, num_layers=2, heads=2, max_seq_len=SEQ,
            dropout=0.0,
            lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o"]),
        )
    )
    opt = MaskedAdamW(student.parameters(), lr=1e-2)
    mods = build_module_index(student)
    # Tuned so repetition actually rises inside a short test: with the
    # shipped k=10 / alpha_mom=0.05 nothing leaves the R branch in 40 steps
    # and the comparison would be vacuously true (every action R, zero
    # coordinates opened). test_the_run_actually_opens_coordinates guards it.
    ctrl = ControllerConfig()
    ctrl.repetition.bucket_smoothing_k = 1.0
    ctrl.repetition.alpha_mom = 0.5
    ctrl.repetition.alpha_hash = 0.5
    ctrl.policy.repetition_low = 0.15
    sig = SignalComputer(ctrl, mods)
    policy = RFPPolicy(ctrl.policy, ablation=ctrl.ablation)
    r = RetrievalStore(max_size=64)
    f = FastStore(ctrl.writer.top_k_granularity, ctrl.writer.top_k_fraction)
    p = PermanentStore()
    writer = WriteExecutor(student, opt, r, f, p, config=ctrl.writer)
    opt.register_controller_params(pp for m in mods for pp in m.params)
    return student, opt, mods, sig, policy, writer


def _run(rank: int, world_size: int, out_path: str, steps: int = STEPS) -> None:
    """One training rank. Writes its decision log + weight checksum to disk.

    ``steps`` is an explicit argument, not the module global, because
    mp.spawn re-imports this module in the child — a monkeypatched global
    would silently not reach the ranks.
    """
    from hivemind.distributed import DistConfig, DistContext
    from hivemind.embedding import SharedEmbedding

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    ctx = DistContext(
        cfg=DistConfig(enabled=world_size > 1, backend="gloo",
                       assert_rank_consistency=1),
        rank=rank, world_size=world_size,
    )

    student, opt, mods, sig, policy, writer = _build(seed=1234)
    module_map = {m.id: m for m in mods}
    model_fwd = ctx.wrap(student)
    embed = SharedEmbedding(student.embed)

    decisions = []
    for step, (tokens_full, bucket, conf_full) in enumerate(_make_batches()[:steps]):
        # Contiguous shard — exactly what StreamBatchSampler does.
        per = GLOBAL_BATCH // world_size
        tokens = tokens_full[rank * per : (rank + 1) * per]
        conf = conf_full[rank * per : (rank + 1) * per]

        h_t = embed(tokens)
        opt.zero_grad()
        logits = model_fwd(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1)
        )
        loss.backward()

        bucket_id, embedding, conf_mean = ctx.sync_controller_inputs(
            bucket_id=bucket,
            embedding=h_t[0],
            conf_sum=float(conf.sum().item()),
            conf_count=int(conf.numel()),
        )
        signals = sig.compute_all(
            optimizer=opt, modules=mods, bucket_id=bucket_id, embedding=embedding
        )
        actions = policy.decide(signals, teacher_confidence=conf_mean)
        metrics, masks = writer.execute(
            actions=actions, module_map=module_map, step=step,
            embedding=embedding, bucket_id=bucket_id, teacher_id=0,
        )
        payload = (
            step,
            bucket_id,
            round(conf_mean, 12) if conf_mean is not None else None,
            tuple((d["module"], d["store"], d["coords_opened"])
                  for d in metrics["actions"]),
        )
        ctx.assert_rank_consistent(payload, what="routing decisions")
        decisions.append([step, bucket_id, [list(x) for x in payload[3]]])

        opt.set_masks(masks)
        try:
            opt.step()
        finally:
            opt.clear_masks()

    if rank == 0:
        # Store the FLAT parameter vector, not per-tensor sums: exp_avg and
        # centred weights are near-cancelling sums, so their totals have
        # terrible relative conditioning and a 1e-7 per-element difference
        # shows up as a 0.5% difference in the total. Compare element-wise.
        flat = torch.cat(
            [v.detach().reshape(-1).double() for v in student.state_dict().values()]
        )
        moments = torch.cat(
            [
                st["exp_avg"].detach().reshape(-1).double()
                for st in opt.state.values()
                if "exp_avg" in st
            ]
        )
        _, _, cs_sum = ctx.state_checksum(student, opt)
        with open(out_path, "w") as fh:
            json.dump(
                {
                    "decisions": decisions,
                    "weights": flat.tolist(),
                    "moments": moments.tolist(),
                    "coord_step_total": cs_sum,
                },
                fh,
            )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def _collect(world_size: int, steps: int = STEPS) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.json")
        if world_size == 1:
            _run(0, 1, out, steps)
        else:
            os.environ["MASTER_PORT"] = str(29500 + world_size + steps)
            mp.spawn(_run, args=(world_size, out, steps), nprocs=world_size, join=True)
        with open(out) as fh:
            return json.load(fh)


@pytest.fixture(scope="module")
def results():
    return {1: _collect(1), 2: _collect(2)}


@pytest.fixture(scope="module")
def one_step():
    """A single step, before any drift can compound."""
    return {1: _collect(1, steps=1), 2: _collect(2, steps=1)}


class TestOneRankEqualsTwo:
    def test_routing_decisions_are_identical(self, results):
        """Same store, same module, same coordinate count, every step."""
        assert results[1]["decisions"] == results[2]["decisions"]

    def test_coordinate_counters_are_exactly_identical(self, results):
        """coord_step is an integer count of accepted updates per coordinate.

        The sharpest signal available: it can only match if every rank opened
        exactly the same coordinates on every step, and it drives the
        coordinate-local bias correction of Corollary 1. Exact equality.
        """
        assert results[1]["coord_step_total"] == results[2]["coord_step_total"]

    def test_final_weights_agree_to_float32_precision(self, results):
        """Not bit-exact, and cannot be.

        DDP's all-reduce sums per-rank gradients in a different order than a
        single process sums over the whole batch. The results are equal in
        exact arithmetic (the loss normalizes by token count, and the shards
        are equal-sized here) but float addition is not associative, so they
        differ in the last ulp and that difference compounds over steps.
        What must match exactly is the discrete decisions, above.
        """
        a = torch.tensor(results[1]["weights"])
        b = torch.tensor(results[2]["weights"])
        assert a.shape == b.shape
        max_diff = (a - b).abs().max().item()
        assert max_diff < 1e-3, (
            f"max |Δweight| = {max_diff:.3e} over {a.numel()} params — far "
            "beyond compounded float32 reduction noise (measured ~4e-5 at 40 steps)"
        )

    def test_optimizer_moments_agree_to_float32_precision(self, results):
        a = torch.tensor(results[1]["moments"])
        b = torch.tensor(results[2]["moments"])
        max_diff = (a - b).abs().max().item()
        assert max_diff < 1e-2, f"max |Δexp_avg| = {max_diff:.3e}"


class TestDivergenceIsReductionNoise:
    """Pin down WHY 1 rank and 2 ranks are not bit-identical.

    After a single step the only possible source of difference is the order
    in which float32 gradients are summed: DDP averages two per-rank means,
    a single process takes one mean over the whole batch. Those are equal in
    exact arithmetic (equal shards, token-count-normalized loss) so any
    difference must be at the last ulp. Adam's 1/sqrt(v + eps) then amplifies
    it, which is why the gap grows with steps:

        steps    max|dW|    max|dM|   decisions  coord_step
            1  3.725e-09  2.794e-09      same        same
            2  5.960e-08  3.725e-09      same        same
            5  1.490e-07  7.101e-09      same        same
           10  2.384e-07  1.211e-08      same        same
           20  4.768e-07  9.518e-07      same        same
           40  4.218e-05  3.165e-04      same        same

    The discrete behaviour — which coordinates open — is identical at every
    horizon. That is the property the design guarantees; bit-identical
    floats across different reduction trees are not achievable and are not
    claimed.
    """

    def test_one_step_difference_is_at_float32_epsilon(self, one_step):
        a = torch.tensor(one_step[1]["weights"])
        b = torch.tensor(one_step[2]["weights"])
        max_diff = (a - b).abs().max().item()
        assert max_diff < 1e-7, (
            f"after ONE step max |Δweight| = {max_diff:.3e}; anything above "
            "float32 epsilon means a systematic difference, not reduction order"
        )

    def test_one_step_decisions_match(self, one_step):
        assert one_step[1]["decisions"] == one_step[2]["decisions"]
        assert one_step[1]["coord_step_total"] == one_step[2]["coord_step_total"]

    def test_the_run_actually_opens_coordinates(self, results):
        """Guard against the whole comparison being vacuously true.

        If every action were R with zero coordinates opened, all the equality
        assertions above would pass while testing nothing.
        """
        opened = sum(d[2] for step in results[1]["decisions"] for d in step[2])
        stores = {d[1] for step in results[1]["decisions"] for d in step[2]}
        assert opened > 0, f"no coordinates were ever opened (stores seen: {stores})"
        assert stores & {"F", "P"}, f"only R actions fired: {stores}"


class TestSyncControllerInputs:
    def test_noop_at_world_size_one(self):
        from hivemind.distributed import DistContext

        ctx = DistContext()
        emb = torch.randn(4)
        b, e, c = ctx.sync_controller_inputs(
            bucket_id=7, embedding=emb, conf_sum=1.5, conf_count=2
        )
        assert b == 7
        assert e is emb
        assert c == 0.75

    def test_no_confidence_rows_yields_none(self):
        from hivemind.distributed import DistContext

        _, _, c = DistContext().sync_controller_inputs(
            bucket_id=0, embedding=None, conf_sum=0.0, conf_count=0
        )
        assert c is None


def _run_without_sync(rank: int, world_size: int, out_path: str, steps: int) -> None:
    """Same loop, but with the controller-input sync defeated.

    Each rank keeps its own batch's bucket id — which is precisely the
    situation before ``sync_controller_inputs`` existed.
    """
    from hivemind.distributed import DistContext

    DistContext.sync_controller_inputs = (
        lambda self, *, bucket_id, embedding, conf_sum, conf_count: (
            bucket_id, embedding, (conf_sum / conf_count if conf_count else None)
        )
    )
    orig = _make_batches
    globals()["_make_batches"] = lambda seed=0: [
        (t, (b + rank) % NUM_BUCKETS, c) for t, b, c in orig(seed)
    ]
    _run(rank, world_size, out_path, steps)


class TestTheGuardrailIsNotVacuous:
    """Defeat the sync on purpose; the consistency check must fire.

    Without this, ``assert_rank_consistent`` passing would only prove that
    it never raises — not that it can.
    """

    def test_divergent_controller_inputs_are_detected(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "o.json")
            os.environ["MASTER_PORT"] = "29997"
            with pytest.raises(Exception) as excinfo:
                mp.spawn(
                    _run_without_sync, args=(2, out, 20), nprocs=2, join=True
                )
        assert "diverged across ranks" in str(excinfo.value)
