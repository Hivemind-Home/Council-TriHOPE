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

    def compute_top_k_masks(
        self,
        adapter: LoRAAdapter,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Top-K write-authorization masks for a LoRA module.

        Selects the rank components (or rows) with the highest gradient
        energy. Returns ``(mask_A, mask_B)`` 0/1 tensors shaped like the
        LoRA factors — the caller stages them on ``MaskedAdamW`` so masked
        coordinates receive no update, moment change, or weight decay.
        """
        k = max(1, int(adapter.rank * self.top_k_fraction))

        return adapter.get_top_k_mask(
            k=k,
            granularity=self.top_k_granularity,  # type: ignore[arg-type]
        )

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
