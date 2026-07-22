"""Write executor: applies R/F/P actions to the model and stores.

For each StoreAction:
- R: zero the module's gradient, store in retrieval memory
- F: apply masked Top-K AdamW on LoRA params only
- P: allow base weight update or flag for consolidation
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from typing import Optional

from ..optim.masked_adamw import MaskedAdamW
from ..stores.fast import FastStore
from ..stores.permanent import PermanentStore
from ..stores.retrieval import RetrievalEntry, RetrievalStore
from .config import WriterConfig
from .consolidation import ConsolidationScheduler
from .module_index import ModuleId, ModuleInfo
from .policy import StoreAction


class WriteExecutor:
    """Executes R/F/P write actions on the model and stores."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: MaskedAdamW,
        r_store: RetrievalStore,
        f_store: FastStore,
        p_store: PermanentStore,
        config: WriterConfig | None = None,
        consolidator: Optional[ConsolidationScheduler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.r_store = r_store
        self.f_store = f_store
        self.p_store = p_store
        self.config = config or WriterConfig()
        # Optional: when a P action targets an F-type module, register it
        # with the scheduler so the next consolidation period merges it
        # (with current-signal re-validation). When ``None`` the writer
        # behaves as before — consolidation does an unconditional sweep.
        self.consolidator = consolidator

    def execute(
        self,
        actions: list[StoreAction],
        module_map: dict[ModuleId, ModuleInfo],
        step: int,
        embedding: torch.Tensor | None = None,
        bucket_id: int = 0,
        teacher_id: int = 0,
        teacher_name: str = "",
        teacher_output_text: str = "",
    ) -> dict[str, Any]:
        """Execute all routing actions.

        Args:
            actions: list of StoreAction from the policy.
            module_map: {ModuleId: ModuleInfo} for parameter access.
            step: current training step.
            embedding: [d] routing embedding (for R-store).
            bucket_id: sample bucket hash.
            teacher_id: integer index of the selected teacher (Theory 101
                §7 retrieval-entry field).
            teacher_name: registered teacher name (e.g. "qwen3_coder_30b_a3b").
            teacher_output_text: teacher's text answer for this sample —
                the cheap stand-in for "teacher soft targets".

        Returns:
            Metrics dict with write statistics.
        """
        metrics: dict[str, Any] = {"r_count": 0, "f_count": 0, "p_count": 0}

        for action in actions:
            mod = module_map.get(action.module_id)
            if mod is None:
                continue

            if action.store == "R":
                self._execute_r(
                    mod, step, embedding, bucket_id,
                    teacher_id=teacher_id,
                    teacher_name=teacher_name,
                    teacher_output_text=teacher_output_text,
                )
                metrics["r_count"] += 1

            elif action.store == "F":
                self._execute_f(mod)
                metrics["f_count"] += 1

            elif action.store == "P":
                self._execute_p(mod)
                metrics["p_count"] += 1

        return metrics

    def _execute_r(
        self,
        mod: ModuleInfo,
        step: int,
        embedding: torch.Tensor | None,
        bucket_id: int,
        teacher_id: int = 0,
        teacher_name: str = "",
        teacher_output_text: str = "",
    ) -> None:
        """R-store action: zero gradients, store in retrieval memory."""
        # Zero gradients — no weight update for this module
        for p in mod.params:
            if p.grad is not None:
                p.grad.zero_()

        # Store in retrieval buffer with full Theory 101 §7 payload.
        if embedding is not None:
            self.r_store.add(RetrievalEntry(
                embedding=embedding.detach().cpu(),
                teacher_id=int(teacher_id),
                bucket_id=int(bucket_id),
                step=int(step),
                teacher_name=str(teacher_name),
                teacher_output_text=str(teacher_output_text),
            ))

    def _execute_f(self, mod: ModuleInfo) -> None:
        """F-store action: apply Top-K masked update to LoRA params only.

        For F-type modules (LoRA params), applies sparse Top-K masking.
        For P-type modules routed to F, zeros gradients (base params don't
        get fast updates).
        """
        if mod.id.param_type == "F":
            adapters = self.f_store.get_lora_modules_for_block(
                self.model, mod.id.layer, mod.id.block_type
            )
            for adapter in adapters:
                self.f_store.apply_top_k_update(adapter, self.optimizer)
        else:
            # P-type module routed to F: zero base gradients (don't update base)
            for p in mod.params:
                if p.grad is not None:
                    p.grad.zero_()

    def _execute_p(self, mod: ModuleInfo) -> None:
        """P-store action: allow base weight update (gradients flow normally).

        For P-type modules, gradients are left intact for the optimizer step.
        For F-type modules routed to P, register the module with the
        consolidation scheduler so the next merge cycle will pick it up
        (re-validating current signals at merge time).
        """
        if mod.id.param_type == "F":
            if self.consolidator is not None:
                self.consolidator.flag_for_consolidation(mod.id)
            # Allow normal LoRA update this step; consolidation happens
            # later on schedule.
        else:
            # P-type module: gradients flow normally to optimizer
            pass
