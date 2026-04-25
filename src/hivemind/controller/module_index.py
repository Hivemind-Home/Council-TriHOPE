"""Module indexing for the R/F/P controller.

Enumerates all (layer, block_type, param_type) modules from a StudentModel
for per-module signal computation and routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..student.lora_adapter import get_adapter


@dataclass(frozen=True)
class ModuleId:
    """Unique identifier for a model module.

    j = (layer, block_type, param_type) as described in Theory 101.
    """

    layer: int
    block_type: str  # "attn" or "ffn"
    param_type: str  # "P" (base) or "F" (LoRA)

    def __str__(self) -> str:
        return f"L{self.layer}.{self.block_type}.{self.param_type}"


@dataclass
class ModuleInfo:
    """Module information including its parameters."""

    id: ModuleId
    params: list[nn.Parameter]
    param_names: list[str]

    @property
    def num_params(self) -> int:
        """d_j: total number of parameters in this module."""
        return sum(p.numel() for p in self.params)


def build_module_index(model: nn.Module) -> list[ModuleInfo]:
    """Enumerate all (layer, block_type, param_type) modules from a StudentModel.

    For a model with L layers, this produces up to 4L modules:
    - (l, attn, P): base attention weights
    - (l, attn, F): LoRA attention parameters
    - (l, ffn, P): base FFN weights
    - (l, ffn, F): LoRA FFN parameters

    Only includes modules that actually have parameters (e.g., if no LoRA
    is configured for a projection, the F-module is omitted).
    """
    modules: list[ModuleInfo] = []

    for layer_idx, block in enumerate(model.blocks):  # type: ignore[attr-defined]
        # --- Attention block ---
        attn = block.attn
        attn_base_params = []
        attn_base_names = []
        attn_lora_params = []
        attn_lora_names = []

        for proj_name in ["w_q", "w_k", "w_v", "w_o"]:
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue
            adapter = get_adapter(proj)
            if adapter is not None:
                for p in adapter.base_params:
                    attn_base_params.append(p)
                    attn_base_names.append(f"blocks.{layer_idx}.attn.{proj_name}.base")
                for p in adapter.lora_params:
                    attn_lora_params.append(p)
                    attn_lora_names.append(f"blocks.{layer_idx}.attn.{proj_name}.lora")
            elif isinstance(proj, nn.Linear):
                attn_base_params.append(proj.weight)
                attn_base_names.append(f"blocks.{layer_idx}.attn.{proj_name}.weight")

        if attn_base_params:
            modules.append(ModuleInfo(
                id=ModuleId(layer_idx, "attn", "P"),
                params=attn_base_params,
                param_names=attn_base_names,
            ))
        if attn_lora_params:
            modules.append(ModuleInfo(
                id=ModuleId(layer_idx, "attn", "F"),
                params=attn_lora_params,
                param_names=attn_lora_names,
            ))

        # --- FFN block ---
        ffn = block.ffn
        ffn_base_params = []
        ffn_base_names = []
        ffn_lora_params = []
        ffn_lora_names = []

        for proj_name in ["w_gate", "w_up", "w_down"]:
            proj = getattr(ffn, proj_name, None)
            if proj is None:
                continue
            adapter = get_adapter(proj)
            if adapter is not None:
                for p in adapter.base_params:
                    ffn_base_params.append(p)
                    ffn_base_names.append(f"blocks.{layer_idx}.ffn.{proj_name}.base")
                for p in adapter.lora_params:
                    ffn_lora_params.append(p)
                    ffn_lora_names.append(f"blocks.{layer_idx}.ffn.{proj_name}.lora")
            elif isinstance(proj, nn.Linear):
                ffn_base_params.append(proj.weight)
                ffn_base_names.append(f"blocks.{layer_idx}.ffn.{proj_name}.weight")

        if ffn_base_params:
            modules.append(ModuleInfo(
                id=ModuleId(layer_idx, "ffn", "P"),
                params=ffn_base_params,
                param_names=ffn_base_names,
            ))
        if ffn_lora_params:
            modules.append(ModuleInfo(
                id=ModuleId(layer_idx, "ffn", "F"),
                params=ffn_lora_params,
                param_names=ffn_lora_names,
            ))

    return modules
