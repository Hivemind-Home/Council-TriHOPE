"""Multi-GPU support: DistributedDataParallel, and only DDP.

Why not DeepSpeed or FSDP
-------------------------

TriHOPE's formal guarantees live inside :class:`~hivemind.optim.masked_adamw.MaskedAdamW`:
Theorem 1 (a closed coordinate receives no parameter update, no moment
update, no AMSGrad max, no decoupled weight decay, and its coordinate-local
update counter does not age) and Proposition 1 (opening K rank components
opens exactly ``K(d_in + d_out)`` coordinates). Both depend on three things
the sharded engines take away:

* **The optimizer step must be ours.** DeepSpeed's ``engine.step()``
  replaces it, so masks staged by ``optimizer.set_masks()`` are never
  consumed and the entire R/F/P write authorization becomes a *silent*
  no-op — training proceeds, every coordinate updates, and no error is
  raised. ``MaskedAdamW`` also rejects ``fused``/``foreach``, which
  DeepSpeed's ``FusedAdam``/``CPUAdam`` paths assume.
* **The controller must be able to read ``m`` and ``v`` unsharded.**
  ZeRO-2/3 partitions ``exp_avg``/``exp_avg_sq``, so
  ``get_state_for_param`` hands back a shard: surprise's ``g² / v``
  broadcast-fails, or (ZeRO-3, outside a gather scope) ``p.grad`` is None
  and every signal silently reads zero.
* **Masks are full-shape tensors keyed by parameter object.** FSDP with
  ``use_orig_params=False`` removes the original ``nn.Parameter`` objects
  from ``param_groups``, so ``register_controller_params`` skips all of
  them and the default flips from closed to *open* — Theorem 1 evaporates
  with no traceback. With ``use_orig_params=True`` each rank holds a 1-D
  slice of a flat shard and the masks shape-mismatch.

DDP has none of these problems. It replicates the model, all-reduces
gradients during backward, leaves ``optimizer.step()` entirely to us with
unsharded parameters and moments, and — crucially — **shares the identical
``nn.Parameter`` objects with the module it wraps**, so the module index,
the optimizer state dict, ``_controller_params`` and the per-step write
masks (all keyed by object identity) keep working untouched.

The replication invariant
-------------------------

Under DDP, gradients are synchronized; parameters are *not*, after
construction. The controller is therefore only sound because of this:

    If at step t (i) every replica's parameters and MaskedAdamW state are
    identical, (ii) DDP all-reduces gradients so ``p.grad`` is identical,
    and (iii) ``bucket_id``, ``embedding`` and ``conf_mean`` are identical,
    then the signals are a pure function of identical inputs, the policy is
    stateless with a stable sort, the write masks derive purely from
    ``p.grad`` — so the actions and masks are identical and (i) holds at
    t+1.

The base case is DDP's construction-time parameter broadcast plus the
shared seed. Only (iii) is not free: those three values are otherwise
derived from the *rank-local* batch. :meth:`DistContext.sync_controller_inputs`
is the entire job, and :meth:`DistContext.assert_rank_consistent` is the
guardrail that proves it held.

Single-process behaviour
------------------------

Every method here is a hard no-op when ``world_size == 1``: no collective
is constructed, no tensor allocated, no branch taken. That is the contract
that keeps the single-GPU path bit-identical to before this module existed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

_REJECTED_STRATEGIES = {
    "deepspeed": (
        "DeepSpeed's engine.step() replaces MaskedAdamW.step(), so the R/F/P "
        "write masks are staged and never consumed — Theorem 1 (exact state "
        "isolation) silently stops holding and every coordinate updates. "
        "ZeRO-2/3 additionally shards exp_avg/exp_avg_sq, which the controller "
        "reads directly."
    ),
    "fsdp": (
        "FSDP with use_orig_params=False removes the original nn.Parameter "
        "objects from param_groups, so register_controller_params skips them "
        "and closed coordinates default to OPEN — Theorem 1 evaporates with no "
        "error. With use_orig_params=True each rank holds a 1-D shard slice and "
        "the full-shape write masks shape-mismatch."
    ),
}


@dataclass
class DistConfig:
    """Parsed ``distributed:`` config block."""

    enabled: bool = False
    backend: str = "nccl"          # gloo for CPU tests
    strategy: str = "ddp"
    world_size: Any = "auto"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = False
    #: Verify every N steps that all ranks still agree on model + optimizer
    #: state (tier 2). Tier 1 — the per-step decision digest — runs on every
    #: step whenever this is non-zero. 0 disables both.
    assert_rank_consistency: int = 100
    eval_mode: str = "rank0"       # rank0 | (sharded: not implemented)
    consolidation_mode: str = "replicate"   # replicate | rank0_broadcast
    on_empty_batch: str = "fail"   # fail | skip
    allow_world_size_change: bool = False
    timeout_minutes: int = 30

    @staticmethod
    def parse(raw: Optional[dict]) -> "DistConfig":
        raw = dict(raw or {})
        cfg = DistConfig(**{k: v for k, v in raw.items() if k in DistConfig.__dataclass_fields__})
        unknown = set(raw) - set(DistConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown distributed config keys: {sorted(unknown)}")
        strategy = str(cfg.strategy).lower()
        # Checked even when disabled: otherwise `strategy: fsdp, enabled:
        # false` validates clean and silently becomes DDP the day someone
        # flips the switch.
        if strategy in _REJECTED_STRATEGIES:
            raise ValueError(
                f"distributed.strategy={strategy!r} is not supported. "
                f"{_REJECTED_STRATEGIES[strategy]} Use strategy=ddp."
            )
        if strategy != "ddp":
            raise ValueError(
                f"distributed.strategy must be 'ddp', got {strategy!r}."
            )
        if cfg.eval_mode not in ("rank0",):
            raise ValueError(
                f"distributed.eval_mode={cfg.eval_mode!r} is not implemented; "
                "use 'rank0'."
            )
        if cfg.consolidation_mode not in ("replicate", "rank0_broadcast"):
            raise ValueError(
                f"distributed.consolidation_mode={cfg.consolidation_mode!r} must be "
                "'replicate' or 'rank0_broadcast'."
            )
        if cfg.on_empty_batch not in ("fail", "skip"):
            raise ValueError(
                f"distributed.on_empty_batch={cfg.on_empty_batch!r} must be "
                "'fail' or 'skip'."
            )
        return cfg


@dataclass
class DistContext:
    """Handle for the (possibly absent) process group.

    Construct via :func:`init_distributed`. When ``world_size == 1`` every
    method returns immediately without touching ``torch.distributed``.
    """

    cfg: DistConfig = field(default_factory=DistConfig)
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    # -- properties --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    # -- collectives (all no-ops at world_size == 1) -----------------------

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def wrap(self, module: nn.Module) -> nn.Module:
        """Return the DDP-wrapped module, or the module itself.

        Call this AFTER any ``requires_grad`` mutation — DDP freezes the
        trainable set at construction. The caller must keep using the
        unwrapped module for everything except the forward pass: DDP has no
        ``__getattr__`` forwarding, so ``model.blocks`` / ``.embed`` /
        ``.get_param_groups()`` all raise through it.
        """
        if not self.enabled:
            return module
        try:
            on_cuda = next(module.parameters()).is_cuda
        except StopIteration:
            on_cuda = False
        device_ids = [self.local_rank] if on_cuda else None
        return nn.parallel.DistributedDataParallel(
            module,
            device_ids=device_ids,
            output_device=self.local_rank if device_ids else None,
            find_unused_parameters=self.cfg.find_unused_parameters,
            gradient_as_bucket_view=self.cfg.gradient_as_bucket_view,
            broadcast_buffers=self.cfg.broadcast_buffers,
        )

    def broadcast_tensor_(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.enabled:
            dist.broadcast(tensor, src=src)
        return tensor

    def broadcast_obj(self, obj: Any, src: int = 0) -> Any:
        if not self.enabled:
            return obj
        box = [obj if self.rank == src else None]
        dist.broadcast_object_list(box, src=src)
        return box[0]

    def all_reduce_max_int(self, value: int) -> int:
        if not self.enabled:
            return int(value)
        t = torch.tensor([int(value)], dtype=torch.long, device=self._coll_device())
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def _coll_device(self) -> torch.device:
        if self.cfg.backend == "nccl" and torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    # -- the controller-consistency mechanism ------------------------------

    def sync_controller_inputs(
        self,
        *,
        bucket_id: int,
        embedding: Optional[torch.Tensor],
        conf_sum: float,
        conf_count: int,
        conf_local_mean: Optional[float] = None,
    ) -> tuple[int, Optional[torch.Tensor], Optional[float]]:
        """Make the three rank-local controller inputs globally identical.

        ``bucket_id`` and ``embedding`` are taken from **rank 0**, which —
        because :class:`~hivemind.data.stream.StreamBatchSampler` shards each
        step's index list *contiguously* — holds the global batch's first
        row. That reproduces the single-process semantics of
        ``metadata["bucket_id"][0]`` and ``h_t[0]`` exactly.

        ``conf_mean`` is reduced as a sum and a count, so it equals the mean
        over the *whole* global batch. Averaging per-rank means would only
        agree when every shard has the same size, and broadcasting rank 0's
        mean would not match single-GPU at all.

        One fused all-reduce carries all of it: ``d + 3`` float64 values,
        ~8 KB for Qwen3-0.6B and ~16 KB for 1.7B — tens of microseconds
        against a step measured in tens of milliseconds.
        """
        if not self.enabled:
            # Return the caller's own float32 mean verbatim. Recomputing it
            # as a float64 sum/count would differ in the last bits, and
            # RFPPolicy compares confidence against its threshold strictly —
            # so that would be a decision-visible change to the single-GPU
            # path, which this module promises not to touch.
            mean = (
                conf_local_mean
                if conf_local_mean is not None
                else (conf_sum / conf_count if conf_count else None)
            )
            return bucket_id, embedding, mean

        d = embedding.numel() if embedding is not None else 0
        device = self._coll_device()
        buf = torch.zeros(d + 3, dtype=torch.float64, device=device)
        if self.is_main:
            buf[0] = float(bucket_id)
            if embedding is not None:
                buf[1 : d + 1] = embedding.detach().reshape(-1).to(torch.float64).to(device)
        buf[d + 1] = float(conf_sum)
        buf[d + 2] = float(conf_count)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

        out_bucket = int(round(float(buf[0].item())))
        out_emb: Optional[torch.Tensor] = None
        if embedding is not None:
            out_emb = buf[1 : d + 1].to(embedding.dtype).to(embedding.device).view_as(embedding)
        total = float(buf[d + 2].item())
        out_conf = float(buf[d + 1].item()) / total if total > 0 else None
        return out_bucket, out_emb, out_conf

    def assert_rank_consistent(self, payload: Any, *, what: str) -> None:
        """Tier 1: verify every rank produced the same routing decisions.

        Hashes ``payload`` and compares the digest via MIN and MAX
        all-reduces (16 bytes viewed as two int64s), so no pickling or
        gathering is involved. The failure this catches — replicas quietly
        opening different coordinates and drifting apart — produces no error
        of its own, which is exactly why it needs an explicit check.
        """
        if not self.enabled or not self.cfg.assert_rank_consistency:
            return
        import hashlib

        digest = hashlib.blake2s(repr(payload).encode("utf-8"), digest_size=16).digest()
        local = torch.tensor(
            [
                int.from_bytes(digest[:8], "little", signed=False) - (1 << 63),
                int.from_bytes(digest[8:], "little", signed=False) - (1 << 63),
            ],
            dtype=torch.int64,
            device=self._coll_device(),
        )
        lo, hi = local.clone(), local.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not (torch.equal(lo, local) and torch.equal(hi, local)):
            raise RuntimeError(
                f"rank {self.rank}: {what} diverged across ranks. The replicas "
                "are no longer computing the same routing decisions, so their "
                "weights will drift apart silently. Local payload: "
                f"{payload!r}"
            )

    @staticmethod
    def _positional_sum(t: torch.Tensor) -> float:
        """Index-weighted sum: unlike a plain sum, it detects a PERMUTATION.

        That matters because the failure this guards against is a merge
        broadcasting adapters in a different order per rank, which swaps
        equally-shaped tensors' contents. A plain sum is invariant under
        exactly that.
        """
        flat = t.detach().reshape(-1).to(torch.float64)
        ramp = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
        return float((flat * ramp).sum().item())

    def state_checksum(
        self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> tuple[float, float, float]:
        """Tier 2 ingredients: index-weighted float64 sums.

        Compared with EXACT equality by :meth:`assert_state_consistent`, and
        that is deliberate: within one job DDP hands every rank the same
        all-reduced gradient, so the replicas stay bit-identical (measured
        over 120 steps: zero difference). The float drift documented in the
        equivalence test is between a 1-rank JOB and a 2-rank JOB — different
        reduction trees — which is a different comparison entirely.
        """
        p_sum = 0.0
        for p in model.parameters():
            p_sum += self._positional_sum(p)
        m_sum = cs_sum = 0.0
        if optimizer is not None:
            for st in optimizer.state.values():
                ea = st.get("exp_avg")
                if ea is not None:
                    m_sum += self._positional_sum(ea)
                cs = st.get("coord_step")
                if cs is not None:
                    cs_sum += self._positional_sum(cs)
        return p_sum, m_sum, cs_sum

    def assert_state_consistent(
        self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer], *, step: int
    ) -> None:
        """Tier 2: catch drift that has not yet flipped a discrete decision."""
        if not self.enabled or not self.cfg.assert_rank_consistency:
            return
        local = torch.tensor(
            self.state_checksum(model, optimizer),
            dtype=torch.float64,
            device=self._coll_device(),
        )
        lo, hi = local.clone(), local.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not (torch.equal(lo, local) and torch.equal(hi, local)):
            raise RuntimeError(
                f"rank {self.rank}: model/optimizer state diverged at step {step}. "
                f"local checksums (params, exp_avg, coord_step) = {local.tolist()}, "
                f"min = {lo.tolist()}, max = {hi.tolist()}."
            )


def _resolve_world_size(cfg: DistConfig) -> int:
    env = os.environ.get("WORLD_SIZE")
    if env is not None:
        ws = int(env)
    elif cfg.world_size in (None, "auto"):
        ws = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        ws = int(cfg.world_size)
    declared = cfg.world_size
    if declared not in (None, "auto") and int(declared) != ws:
        raise RuntimeError(
            f"distributed.world_size={declared} but the launcher provided "
            f"WORLD_SIZE={ws}. Launch with "
            f"`python -m torch.distributed.run --nproc_per_node={declared}` or "
            "drop the config value."
        )
    return max(1, ws)


def init_distributed(cfg: Any) -> DistContext:
    """Parse ``cfg.distributed`` and, if enabled, join the process group."""
    raw = cfg.get("distributed", {}) if hasattr(cfg, "get") else {}
    try:
        from omegaconf import OmegaConf

        if raw is not None and not isinstance(raw, dict):
            raw = OmegaConf.to_container(raw, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    dcfg = DistConfig.parse(raw)

    if not dcfg.enabled:
        return DistContext(cfg=dcfg)

    # "Was this launched by torchrun?" is a question about the ENVIRONMENT,
    # not about how many GPUs exist. Keying on device_count meant a plain
    # `python train.py ++distributed.enabled=true` on a 4-GPU box tried to
    # join an env:// rendezvous that no launcher had set up.
    launched_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    world_size = _resolve_world_size(dcfg)
    if not launched_distributed or world_size <= 1:
        if not launched_distributed and world_size > 1:
            print(
                "[warn] distributed.enabled=true but this process was not started "
                "by a launcher (no RANK/WORLD_SIZE in the environment) — running "
                "single-process. Use: python -m torch.distributed.run "
                f"--standalone --nproc_per_node={world_size} train.py ..."
            )
        return DistContext(cfg=dcfg)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count() or 1)))

    if not dist.is_initialized():
        dist.init_process_group(
            backend=dcfg.backend,
            timeout=timedelta(minutes=dcfg.timeout_minutes),
        )
    return DistContext(
        cfg=dcfg, rank=rank, local_rank=local_rank, world_size=dist.get_world_size()
    )


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reject_unsupported_features(cfg: Any, dist_ctx: DistContext) -> None:
    """Refuse config combinations DDP cannot serve correctly.

    Both are unused by every shipped config, so this trades a fragile
    untested path for a startup error rather than costing anything.
    """
    if not dist_ctx.enabled:
        return
    reg = cfg.get("regularization", {}) or {}
    if float(reg.get("consistency_weight", 0.0) or 0.0) > 0:
        raise ValueError(
            "regularization.consistency_weight > 0 is not supported under "
            "distributed.enabled. It runs three forward passes for one "
            "backward, and DDP's reducer sets its expectation from the last "
            "forward — the behaviour is version-fragile and untested. Set it "
            "to 0 or run single-GPU."
        )
    mp = (cfg.get("train", {}) or {}).get("mixed_precision", {}) or {}
    if bool(mp.get("enabled", False)) and str(mp.get("dtype", "bf16")) == "fp16":
        raise ValueError(
            "mixed_precision.dtype=fp16 is not supported under "
            "distributed.enabled: GradScaler's found_inf is per-rank, so ranks "
            "would disagree about skipping a step and their coordinate update "
            "counters would drift. Use dtype=bf16 (which disables the scaler "
            "entirely) or run single-GPU."
        )
