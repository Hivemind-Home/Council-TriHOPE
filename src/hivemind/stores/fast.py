"""F-store: Fast LoRA parameter management with Top-K masking.

Manages LoRA adapter updates for modules routed to F-store.
Supports sparse Top-K updates at rank-component or row granularity.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ..student.lora import LoRALinear


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
        lora_module: LoRALinear,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Apply Top-K masked gradient update to a LoRA module.

        Computes which rank components (or rows) have the highest gradient
        energy, masks the rest, then applies the optimizer step.

        Args:
            lora_module: the LoRALinear module to update.
            optimizer: the optimizer (must contain lora_module's params).
        """
        k = max(1, int(lora_module.rank * self.top_k_fraction))

        mask_A, mask_B = lora_module.get_top_k_mask(
            k=k,
            granularity=self.top_k_granularity,  # type: ignore
        )

        # Apply masks to gradients before optimizer step
        if lora_module.lora_A.grad is not None:
            lora_module.lora_A.grad.mul_(mask_A.float())
        if lora_module.lora_B.grad is not None:
            lora_module.lora_B.grad.mul_(mask_B.float())

    def get_lora_modules_for_block(
        self, model: nn.Module, layer_idx: int, block_type: str
    ) -> list[LoRALinear]:
        """Get all LoRA modules for a specific block."""
        block = model.blocks[layer_idx]  # type: ignore
        if block_type == "attn":
            return list(block.attn.get_lora_modules().values())
        elif block_type == "ffn":
            return list(block.ffn.get_lora_modules().values())
        return []
