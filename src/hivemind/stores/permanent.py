"""P-store: Permanent base weight consolidation.

Handles F→P consolidation by merging LoRA adapters into base weights,
and direct base weight updates with regularization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..student.lora import LoRALinear


class PermanentStore:
    """Manages permanent (base) weight updates and consolidation."""

    def merge_lora_for_block(
        self, model: nn.Module, layer_idx: int, block_type: str
    ) -> int:
        """Merge all LoRA adapters into base weights for a specific block.

        Args:
            model: the student model.
            layer_idx: which layer.
            block_type: "attn" or "ffn".

        Returns:
            Number of LoRA modules merged.
        """
        block = model.blocks[layer_idx]  # type: ignore
        if block_type == "attn":
            lora_modules = block.attn.get_lora_modules()
        elif block_type == "ffn":
            lora_modules = block.ffn.get_lora_modules()
        else:
            return 0

        count = 0
        for lora in lora_modules.values():
            lora.merge_lora_into_base()
            count += 1
        return count
