"""P-store: permanent base weight consolidation.

Handles F→P consolidation by merging LoRA adapters into base weights.
Speaks to LoRA via the ``LoRAAdapter`` protocol so this works whether
LoRA was attached by us (``LoRALinear``) or by PEFT/Unsloth.
"""

from __future__ import annotations

import torch.nn as nn

from ..student.lora_adapter import get_adapter


class PermanentStore:
    """Manages permanent (base) weight updates and consolidation."""

    def merge_lora_for_block(
        self, model: nn.Module, layer_idx: int, block_type: str
    ) -> int:
        """Merge LoRA adapters into base weights for one block."""
        block = model.blocks[layer_idx]  # type: ignore[attr-defined]
        if block_type == "attn":
            container = block.attn
        elif block_type == "ffn":
            container = block.ffn
        else:
            return 0

        count = 0
        for mod in container.get_lora_modules().values():
            adapter = get_adapter(mod)
            if adapter is None:
                continue
            adapter.merge_lora_into_base()
            count += 1
        return count
