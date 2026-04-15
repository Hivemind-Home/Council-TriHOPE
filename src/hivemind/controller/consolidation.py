"""Periodic F→P consolidation scheduler.

Merges LoRA adapters into base weights when a module has been
consistently recurring and stable for a sustained period.
"""

from __future__ import annotations

import torch.nn as nn

from ..stores.permanent import PermanentStore
from .config import ConsolidationConfig
from .module_index import ModuleId
from .signals import ModuleSignals


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

    def should_check(self, step: int) -> bool:
        """Whether to check for consolidation at this step."""
        if self.config.period <= 0:
            return False
        return step > 0 and step % self.config.period == 0

    def consolidate(
        self,
        module_signals: dict[ModuleId, ModuleSignals],
    ) -> list[ModuleId]:
        """Consolidate qualifying F-modules into P-store.

        A module qualifies if:
        - It is an F-type (LoRA) module
        - R_t >= min_repetition (clearly repeating)
        - C_t >= min_stability_C (directionally stable)

        Args:
            module_signals: current signals for all modules.

        Returns:
            List of ModuleIds that were consolidated.
        """
        consolidated: list[ModuleId] = []

        for mid, sig in module_signals.items():
            # Only consolidate F-modules
            if mid.param_type != "F":
                continue

            # Check qualification
            if (
                sig.repetition >= self.config.min_repetition
                and sig.stability_C >= self.config.min_stability_C
            ):
                # Merge LoRA into base
                merged = self.p_store.merge_lora_for_block(
                    self.model, mid.layer, mid.block_type
                )
                if merged > 0:
                    consolidated.append(mid)

        return consolidated
