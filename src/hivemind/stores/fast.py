"""F-store: fast LoRA parameter management with Top-K masking.

Speaks to LoRA modules through the ``LoRAAdapter`` protocol so the same
code paths handle both our in-repo ``LoRALinear`` and PEFT's
``lora.Linear`` (used by the Unsloth backbone).
"""

from __future__ import annotations

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

    @staticmethod
    def slice_bounds(rank: int, teacher_index: int, num_teachers: int) -> tuple[int, int]:
        """Contiguous rank slice ``[lo, hi)`` reserved for one teacher.

        Width ``rank // num_teachers``; the remainder (``rank % num_teachers``
        components) stays unused so every teacher gets the same budget.
        """
        k = int(num_teachers)
        if k <= 0 or int(rank) < k:
            raise ValueError(
                f"teacher_partition needs lora.rank >= number of teachers "
                f"({int(rank)} < {k}); raise model.lora.rank."
            )
        width = int(rank) // k
        lo = int(teacher_index) * width
        return lo, lo + width

    def compute_slice_masks(
        self, adapter: LoRAAdapter, teacher_index: int, num_teachers: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masks opening exactly one teacher's rank slice (rows of A, columns
        of B), disjoint across teachers and independent of the gradient."""
        lo, hi = self.slice_bounds(adapter.rank, teacher_index, num_teachers)
        sel = torch.zeros(adapter.rank, dtype=torch.bool, device=adapter.lora_a.device)
        sel[lo:hi] = True
        mask_a = sel.unsqueeze(1).expand_as(adapter.lora_a)
        mask_b = sel.to(adapter.lora_b.device).unsqueeze(0).expand_as(adapter.lora_b)
        return mask_a, mask_b

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
