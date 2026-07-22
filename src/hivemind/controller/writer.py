"""Write executor: turns R/F/P actions into optimizer write-authorization masks.

Controller-indexed parameters are registered default-closed on MaskedAdamW,
so the writer only ever *opens* coordinates:

- R: no mask entry (module stays closed), store the observation in retrieval
  memory instead.
- F: open the Top-K LoRA rank components of the block's adapters.
- P: open base weights fully; a P action on an F-type module opens the whole
  adapter and flags the block for post-step consolidation (paper Algorithm 1).

The masks are staged on the optimizer by the training loop; MaskedAdamW
guarantees exact state isolation for everything left closed (Theorem 1).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

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
        # (with current-signal re-validation).
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
    ) -> tuple[dict[str, Any], dict[nn.Parameter, "torch.Tensor | bool"]]:
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
            ``(metrics, masks)``: write statistics (including a per-action
            ``actions`` detail list for the event trace) and the
            open-authorization masks to stage on the optimizer. Parameters
            absent from ``masks`` stay at their default (closed for
            controller-indexed params).
        """
        metrics: dict[str, Any] = {
            "r_count": 0,
            "f_count": 0,
            "p_count": 0,
            "coords_opened": 0,
            "actions": [],
        }
        masks: dict[nn.Parameter, torch.Tensor | bool] = {}

        for action in actions:
            mod = module_map.get(action.module_id)
            if mod is None:
                continue
            coords_opened = 0

            if action.store == "R":
                self._execute_r(
                    mod, step, embedding, bucket_id,
                    teacher_id=teacher_id,
                    teacher_name=teacher_name,
                    teacher_output_text=teacher_output_text,
                )
                metrics["r_count"] += 1

            elif action.store == "F":
                coords_opened = self._execute_f(mod, masks)
                metrics["f_count"] += 1

            elif action.store == "P":
                coords_opened = self._execute_p(mod, masks)
                metrics["p_count"] += 1

            metrics["coords_opened"] += coords_opened
            metrics["actions"].append(
                {
                    "module": str(action.module_id),
                    "store": action.store,
                    "coords_opened": coords_opened,
                }
            )

        return metrics, masks

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
        """R-store action: no weight update, store in retrieval memory.

        The module's parameters stay closed (default) — MaskedAdamW leaves
        their weights, moments, and counters bit-identical.
        """
        if embedding is not None:
            self.r_store.add(RetrievalEntry(
                embedding=embedding.detach().cpu(),
                teacher_id=int(teacher_id),
                bucket_id=int(bucket_id),
                step=int(step),
                teacher_name=str(teacher_name),
                teacher_output_text=str(teacher_output_text),
            ))

    def _execute_f(
        self,
        mod: ModuleInfo,
        masks: dict[nn.Parameter, "torch.Tensor | bool"],
    ) -> int:
        """F-store action: open Top-K LoRA rank components.

        For F-type modules (LoRA params), opens the selected components.
        For P-type modules routed to F, opens nothing (base params don't
        get fast updates). Returns the number of coordinates opened.
        """
        if mod.id.param_type != "F":
            return 0

        opened = 0
        adapters = self.f_store.get_lora_modules_for_block(
            self.model, mod.id.layer, mod.id.block_type
        )
        for adapter in adapters:
            mask_a, mask_b = self.f_store.compute_top_k_masks(adapter)
            masks[adapter.lora_a] = mask_a
            masks[adapter.lora_b] = mask_b
            opened += int(mask_a.sum().item()) + int(mask_b.sum().item())
        return opened

    def _execute_p(
        self,
        mod: ModuleInfo,
        masks: dict[nn.Parameter, "torch.Tensor | bool"],
    ) -> int:
        """P-store action: open the write surface fully.

        For P-type modules, base weights are opened for the ordinary AdamW
        update. For F-type modules routed to P, the whole adapter is opened
        for one accepted step and the block is flagged for post-step
        consolidation (paper: "opens the corresponding adapter, performs the
        accepted step, and consolidates afterward").
        """
        opened = 0
        for p in mod.params:
            masks[p] = MaskedAdamW.FULLY_OPEN
            opened += p.numel()

        if mod.id.param_type == "F" and self.consolidator is not None:
            self.consolidator.flag_for_consolidation(mod.id)
        return opened
