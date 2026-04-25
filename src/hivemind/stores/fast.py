"""F-store: fast LoRA parameter management with Top-K masking.

Speaks to LoRA modules through the ``LoRAAdapter`` protocol so the same
code paths handle both our in-repo ``LoRALinear`` and PEFT's
``lora.Linear`` (used by the Unsloth backbone).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ..student.lora_adapter import LoRAAdapter, get_adapter


class FastStore:
    """Manages LoRA (F-store) parameter updates."""

    def __init__(
        self,
        top_k_granularity: str = "rank_components",
        top_k_fraction: float = 0.5,
    ) -> None:
        self.top_k_granularity = top_k_granularity
        self.top_k_fraction = top_k_fraction

    def apply_top_k_update(
        self,
        adapter: LoRAAdapter,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Apply Top-K masked gradient update to a LoRA module.

        Computes which rank components (or rows) have the highest gradient
        energy, masks the rest, then leaves the optimizer step to the
        outer loop.
        """
        k = max(1, int(adapter.rank * self.top_k_fraction))

        mask_A, mask_B = adapter.get_top_k_mask(
            k=k,
            granularity=self.top_k_granularity,  # type: ignore[arg-type]
        )

        if adapter.lora_a.grad is not None:
            adapter.lora_a.grad.mul_(mask_A.float())
        if adapter.lora_b.grad is not None:
            adapter.lora_b.grad.mul_(mask_B.float())

    def get_lora_modules_for_block(
        self, model: nn.Module, layer_idx: int, block_type: str
    ) -> list[LoRAAdapter]:
        """Return adapter views of every LoRA module under one block."""
        block = model.blocks[layer_idx]  # type: ignore[attr-defined]
        if block_type == "attn":
            container = block.attn
        elif block_type == "ffn":
            container = block.ffn
        else:
            return []

        # Each container exposes ``get_lora_modules() -> dict[name, module]``.
        # The returned modules may be either our LoRALinear or a PEFT
        # lora.Linear depending on backbone — both wrap fine.
        adapters: list[LoRAAdapter] = []
        for mod in container.get_lora_modules().values():
            ad = get_adapter(mod)
            if ad is not None:
                adapters.append(ad)
        return adapters
