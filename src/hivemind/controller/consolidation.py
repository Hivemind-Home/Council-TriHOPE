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
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ConsolidationScheduler:
    """Periodic F→P consolidation based on signal thresholds."""

    def __init__(
        self,
        config: ConsolidationConfig,
        model: nn.Module,
        p_store: PermanentStore,
    ) -> None:
        self.config = config
        self.model = model
        self.p_store = p_store
        # Ring buffer of recent input_ids for distill-based consolidation.
        # Populated by ``record_batch`` each training step; bounded to
        # ``config.distill_replay_size``.
        self._replay: deque[torch.Tensor] = deque(maxlen=max(1, config.distill_replay_size))

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

        A module qualifies if it's F-type, stable (C ≥ threshold), and
        repeating (R ≥ threshold).
        """
        consolidated: list[ModuleId] = []

        for mid, sig in module_signals.items():
            if mid.param_type != "F":
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
                    consolidated.append(mid)

        return consolidated

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
            if was_training:
                self.model.train()
            for ad in adapters:
                ad.reset_lora()

        return len(adapters)

    def _pick_replay(self) -> Optional[torch.Tensor]:
        if not self._replay:
            return None
        # Use the most recent batch; simple and avoids mixing distributions.
        return self._replay[-1]

    # -- serialisation ----------------------------------------------------

    def state_dict(self) -> dict:
        return {"replay": [t.clone() for t in self._replay]}

    def load_state_dict(self, state: dict) -> None:
        self._replay.clear()
        for t in state.get("replay", []):
            self._replay.append(t)
