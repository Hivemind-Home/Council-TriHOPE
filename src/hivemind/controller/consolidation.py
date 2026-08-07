"""Periodic F→P consolidation scheduler.

Merges LoRA adapters into base weights when a module has been
consistently recurring and stable for a sustained period.

Two strategies:

* ``direct``: ``base.weight += B @ A * (α/r)`` then reset LoRA — a
  lossless merge of the LoRA delta into base. Same outputs before/after
  for the consolidated block.
* ``distill``: run the block with LoRA to get target logits on a replay
  batch, zero out LoRA, and optimise base weights to minimise
  ``KL(p_with_LoRA ‖ p_base_only)`` for a few inner steps. Then reset
  LoRA. This is softer — it preserves generalisation structure of the
  base while folding in learned behaviour. Use when the LoRA has drifted
  enough that a hard merge would shift the base too abruptly.
"""

from __future__ import annotations

import contextlib
from collections import deque
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..optim.masked_adamw import MaskedAdamW
from ..stores.permanent import PermanentStore
from ..student.lora_adapter import LoRAAdapter, get_adapter
from .config import ConsolidationConfig
from .module_index import ModuleId
from .signals import ModuleSignals


@contextlib.contextmanager
def _lora_zeroed(adapters: Iterable[LoRAAdapter]):
    """Temporarily zero LoRA B matrices so forward() = base-only.

    Saves and restores the original B tensors so distillation can be
    computed against the with-LoRA target without mutating state.
    """
    saved: list[tuple[LoRAAdapter, torch.Tensor]] = []
    for ad in adapters:
        saved.append((ad, ad.lora_b.detach().clone()))
        with torch.no_grad():
            ad.lora_b.zero_()
    try:
        yield
    finally:
        for ad, backup in saved:
            with torch.no_grad():
                ad.lora_b.copy_(backup)


def _module_sort_key(mid: ModuleId) -> tuple:
    """Rank-stable ordering for ModuleId (whose hash is seed-randomized)."""
    return (mid.layer, mid.block_type, mid.param_type)


class ConsolidationScheduler:
    """Periodic F→P consolidation based on signal thresholds."""

    def __init__(
        self,
        config: ConsolidationConfig,
        model: nn.Module,
        p_store: PermanentStore,
        optimizer: Optional[MaskedAdamW] = None,
        dist: Any = None,
    ) -> None:
        self.config = config
        self.model = model
        self.p_store = p_store
        # Process-group handle (hivemind.distributed.DistContext) or None.
        # Merges mutate weights outside the optimizer, so they need their own
        # synchronization — see _sync_after_merge.
        self.dist = dist
        # When provided, the merged adapter's optimizer state (moments,
        # AMSGrad max, coordinate counters) is zeroed after every merge —
        # paper Corollary 2: the reinitialized adapter must start from a
        # fresh optimizer state with first-update bias correction.
        self.optimizer = optimizer
        # Ring buffer of recent input_ids for distill-based consolidation.
        # Populated by ``record_batch`` each training step; bounded to
        # ``config.distill_replay_size``.
        self._replay: deque[torch.Tensor] = deque(maxlen=max(1, config.distill_replay_size))
        # Modules the policy has flagged as ready for P (Theory 101 §8:
        # "modules flagged ready for P"). Drained at each consolidation
        # period and re-validated against current signals before merge.
        self._pending_p: set[ModuleId] = set()

    def flag_for_consolidation(self, mid: ModuleId) -> None:
        """Register a module the policy wants consolidated.

        Only F-type modules are merged into P; other ids are ignored so
        callers can pass through whatever the policy emitted.
        """
        if mid.param_type == "F":
            self._pending_p.add(mid)

    @property
    def pending_p(self) -> set[ModuleId]:
        return set(self._pending_p)

    def record_batch(self, input_ids: torch.Tensor) -> None:
        """Capture a batch for potential distill-based consolidation."""
        if self.config.merge_strategy != "distill":
            return
        # Store on CPU to avoid GPU memory pressure; move back on use.
        self._replay.append(input_ids.detach().cpu())

    def should_check(self, step: int) -> bool:
        if self.config.period <= 0:
            return False
        return step > 0 and step % self.config.period == 0

    def consolidate(
        self,
        module_signals: dict[ModuleId, ModuleSignals],
    ) -> list[ModuleId]:
        """Consolidate qualifying F-modules into P-store.

        A module qualifies when:
          1. It's F-type AND
          2. Either the policy has flagged it via ``flag_for_consolidation``
             OR no flagged set is in use (legacy unconditional sweep), AND
          3. Current signals confirm it's stable (C ≥ min_stability_C) and
             repeating (R ≥ min_repetition) — re-validated at merge time
             so a stale flag from earlier doesn't trigger a bad merge.

        Successfully merged modules are removed from the pending set; any
        flagged module that no longer satisfies thresholds remains pending
        for the next period (its signals may recover).
        """
        consolidated: list[ModuleId] = []

        # Decide which modules to consider this round. If the policy has
        # been flagging modules, only re-validate those. Otherwise fall
        # back to the unconditional sweep across all F-type modules
        # (preserves prior behaviour for callers that don't use flags).
        # Sorted, not set/dict order: ModuleId is a frozen dataclass of strs,
        # so its hash is PYTHONHASHSEED-randomized and torchrun does not pin
        # the seed. Iterating _pending_p directly makes each rank merge — and
        # therefore broadcast in _sync_after_merge — in a DIFFERENT order,
        # pairing rank i's j-th adapter with rank 0's j-th tensor from another
        # module. Between equally-shaped modules that succeeds silently and
        # permutes lora_a; a checksum cannot see it, because a sum is
        # invariant under permutation.
        if self._pending_p:
            candidates = sorted(self._pending_p, key=_module_sort_key)
        else:
            candidates = sorted(
                (mid for mid in module_signals if mid.param_type == "F"),
                key=_module_sort_key,
            )

        for mid in candidates:
            if mid.param_type != "F":
                continue
            sig = module_signals.get(mid)
            if sig is None:
                # No fresh signal for this module — keep it pending; we'll
                # re-check next period.
                continue
            if (
                sig.repetition >= self.config.min_repetition
                and sig.stability_C >= self.config.min_stability_C
            ):
                if self.config.merge_strategy == "distill":
                    merged = self._consolidate_distill(mid)
                else:
                    merged = self.p_store.merge_lora_for_block(
                        self.model, mid.layer, mid.block_type
                    )
                if merged > 0:
                    self._reset_adapter_optimizer_state(mid)
                    self._sync_after_merge(mid)
                    consolidated.append(mid)
                    self._pending_p.discard(mid)

        return consolidated

    def _sync_after_merge(self, mid: ModuleId) -> None:
        """Broadcast the freshly reinitialized adapter from rank 0.

        ``reset_lora`` draws A from ``nn.init.kaiming_uniform_`` off the
        GLOBAL torch RNG. Today every rank happens to agree, because they
        share a seed and all shipped configs run dropout=0 — but that is a
        latent invariant, not a guarantee: enable dropout and the ranks
        consume different numbers of RNG values, their streams desynchronize,
        and the next merge initializes a DIFFERENT A on each rank. That is
        silent, permanent divergence of the model.

        Broadcasting is preferred over threading a shared Generator into
        reset_lora because changing the RNG source would change single-GPU
        numerics. B is exactly zero after a merge, so only A needs sending —
        a few hundred KB, at a period measured in thousands of steps.
        """
        if self.dist is None or not getattr(self.dist, "enabled", False):
            return
        for ad in self._get_lora_adapters(mid):
            self.dist.broadcast_tensor_(ad.lora_a.data)

    def _reset_adapter_optimizer_state(self, mid: ModuleId) -> None:
        """Zero optimizer state for the merged block's LoRA factors.

        Single choke point for both merge strategies (Corollary 2): after
        ``merge_lora_into_base`` reinitializes A and zeroes B, stale Adam
        moments must not carry pre-merge momentum into the fresh adapter.
        """
        if self.optimizer is None:
            return
        lora_params = [
            p for ad in self._get_lora_adapters(mid) for p in ad.lora_params
        ]
        self.optimizer.reset_state_for_params(lora_params)

    def all_f_modules(self) -> list[ModuleId]:
        """Every F-type module that actually has LoRA adapters."""
        out: list[ModuleId] = []
        num_layers = len(self.model.blocks)  # type: ignore[attr-defined]
        for layer in range(num_layers):
            for block_type in ("attn", "ffn"):
                mid = ModuleId(layer, block_type, "F")
                if self._get_lora_adapters(mid):
                    out.append(mid)
        return out

    def force_consolidate(
        self, module_ids: Optional[Iterable[ModuleId]] = None
    ) -> list[ModuleId]:
        """Merge modules NOW, bypassing threshold re-validation.

        Debug/experiment path for the incorrect-consolidation study: merges
        the given F-modules (default: all of them) regardless of signals.
        Optimizer state is still reset (Corollary 2), so the merge itself
        remains mechanically correct — only the *timing* is wrong.
        """
        targets = sorted(
            module_ids if module_ids is not None else self.all_f_modules(),
            key=_module_sort_key,
        )
        merged_ids: list[ModuleId] = []
        for mid in targets:
            if mid.param_type != "F":
                continue
            if self.config.merge_strategy == "distill":
                merged = self._consolidate_distill(mid)
            else:
                merged = self.p_store.merge_lora_for_block(
                    self.model, mid.layer, mid.block_type
                )
            if merged > 0:
                self._reset_adapter_optimizer_state(mid)
                self._sync_after_merge(mid)
                self._pending_p.discard(mid)
                merged_ids.append(mid)
        return merged_ids

    # -- strategies --------------------------------------------------------

    def _get_lora_adapters(self, mid: ModuleId) -> list[LoRAAdapter]:
        block = self.model.blocks[mid.layer]  # type: ignore[attr-defined]
        if mid.block_type == "attn":
            container = block.attn
        elif mid.block_type == "ffn":
            container = block.ffn
        else:
            return []
        out: list[LoRAAdapter] = []
        for mod in container.get_lora_modules().values():
            ad = get_adapter(mod)
            if ad is not None:
                out.append(ad)
        return out

    def _consolidate_distill(self, mid: ModuleId) -> int:
        adapters = self._get_lora_adapters(mid)
        if not adapters:
            return 0

        replay = self._pick_replay()
        if replay is None:
            # Fallback to direct merge when we have nothing to distill against.
            return self.p_store.merge_lora_for_block(
                self.model, mid.layer, mid.block_type
            )

        replay = replay.to(next(self.model.parameters()).device)
        was_training = self.model.training
        self.model.eval()

        # Step 1: target logits from student with LoRA active.
        with torch.no_grad():
            target_logits = self.model(replay).detach()

        # Step 2: inner distillation — train base of this block only.
        base_params: list[nn.Parameter] = []
        for ad in adapters:
            for p in ad.base_params:
                base_params.append(p)

        # Freeze everything except the target block's base weights.
        previously_trainable: list[tuple[nn.Parameter, bool]] = []
        for p in self.model.parameters():
            previously_trainable.append((p, p.requires_grad))
            p.requires_grad = False
        for p in base_params:
            p.requires_grad = True

        opt = torch.optim.AdamW(base_params, lr=self.config.distill_lr)
        tau = max(self.config.distill_tau, 1e-3)

        # ``self.model`` is the unwrapped student, so these inner backwards
        # bypass DDP's forward and its reducer early-returns today. That is
        # correct but version-dependent, so pin it: clearing the sync flag
        # makes the reducer's non-participation explicit rather than
        # incidental. tests/test_distributed_consolidation.py fails on a
        # timeout if a future torch release changes the behaviour.
        ddp = getattr(self, "_ddp_handle", None)
        prev_sync = getattr(ddp, "require_backward_grad_sync", None)
        if ddp is not None:
            ddp.require_backward_grad_sync = False

        try:
            with _lora_zeroed(adapters):
                for _ in range(max(1, self.config.distill_iters)):
                    opt.zero_grad()
                    base_logits = self.model(replay)
                    loss = (
                        F.kl_div(
                            F.log_softmax(base_logits / tau, dim=-1),
                            F.softmax(target_logits / tau, dim=-1),
                            reduction="batchmean",
                        )
                        * tau * tau
                    )
                    loss.backward()
                    opt.step()
        finally:
            # Restore requires_grad, training mode; reset LoRA.
            for p, was in previously_trainable:
                p.requires_grad = was
            if ddp is not None and prev_sync is not None:
                ddp.require_backward_grad_sync = prev_sync
            if was_training:
                self.model.train()
            for ad in adapters:
                ad.reset_lora()

        return len(adapters)

    def _pick_replay(self) -> Optional[torch.Tensor]:
        if not self._replay:
            return None
        # Use the most recent batch; simple and avoids mixing distributions.
        replay = self._replay[-1]
        if self.dist is not None and getattr(self.dist, "enabled", False):
            # record_batch stored this rank's SHARD, so distilling against it
            # would train each replica's base weights on different data.
            # Broadcast rank 0's — once per merge, not once per step, which
            # is why the fix lives here and not in record_batch.
            #
            # It must happen ON THE MODEL'S DEVICE: record_batch keeps the
            # tensor on CPU to save GPU memory, and a nccl-only process group
            # cannot broadcast a CPU tensor. A gloo test would never catch it,
            # because gloo accepts CPU tensors.
            device = next(self.model.parameters()).device
            shape = self.dist.broadcast_obj(tuple(replay.shape), src=0)
            replay = (
                replay.to(device)
                if self.dist.is_main
                else torch.empty(shape, dtype=replay.dtype, device=device)
            )
            self.dist.broadcast_tensor_(replay)
        return replay

    # -- serialisation ----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "replay": [t.clone() for t in self._replay],
            "pending_p": [
                {"layer": m.layer, "block_type": m.block_type, "param_type": m.param_type}
                for m in self._pending_p
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self._replay.clear()
        for t in state.get("replay", []):
            self._replay.append(t)
        self._pending_p.clear()
        for raw in state.get("pending_p", []):
            self._pending_p.add(
                ModuleId(int(raw["layer"]), str(raw["block_type"]), str(raw["param_type"]))
            )
