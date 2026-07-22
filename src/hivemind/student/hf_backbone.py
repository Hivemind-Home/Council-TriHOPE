"""HuggingFace-backed student model with in-place LoRA injection.

Wraps ``AutoModelForCausalLM`` so the rest of the Hivemind pipeline —
controller, writer, consolidation, module index — can operate on a real
pretrained backbone (Qwen2/Llama-family today) instead of the in-repo
toy transformer.

Key idea: keep the HF model intact and functional (HF-native state_dict
for save/load), but swap target ``nn.Linear`` projections in place with
``LoRALinear`` instances that own both the base weight and the LoRA A/B
parameters. We then publish a small ``blocks`` view that the existing
``build_module_index`` iterates — same projection names (``w_q``,
``w_gate``, etc.), just pointing at the now-wrapped real modules.

Supported architectures (via attribute-path probing):

* Qwen2 / Qwen2.5 — ``model.layers[i].self_attn.{q,k,v,o}_proj`` +
  ``model.layers[i].mlp.{gate,up,down}_proj``
* Llama / Mistral — same layout

Other architectures raise ``NotImplementedError`` with a hint; extend
``_hf_common.ARCH_SPECS`` to add support.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from ._hf_common import ProjView, detect_arch, getattr_path
from .config import LoRAConfig, StudentConfig
from .lora import LoRALinear

# -- LoRA injection --------------------------------------------------------


def _wrap_linear_with_lora(
    linear: nn.Linear,
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRALinear:
    """Return a LoRALinear whose base weight is copied from ``linear``."""
    out_f, in_f = linear.out_features, linear.in_features
    wrapped = LoRALinear(
        in_features=in_f,
        out_features=out_f,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=linear.bias is not None,
    )
    with torch.no_grad():
        wrapped.base.weight.copy_(linear.weight)
        if linear.bias is not None and wrapped.base.bias is not None:
            wrapped.base.bias.copy_(linear.bias)
    return wrapped


def _inject_into(
    parent: nn.Module,
    attr: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRALinear:
    existing = getattr(parent, attr)
    wrapped = _wrap_linear_with_lora(existing, rank, alpha, dropout)
    # Preserve device/dtype
    wrapped = wrapped.to(existing.weight.device, existing.weight.dtype)
    setattr(parent, attr, wrapped)
    return wrapped


# -- main class ------------------------------------------------------------


class HFStudent(nn.Module):
    """Pretrained HF backbone + in-place LoRA on target projections."""

    def __init__(
        self,
        pretrained_name: str,
        lora_config: LoRAConfig,
        *,
        dtype: str = "float32",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        torch_dtype = getattr(torch, dtype, torch.float32)
        self.hf_config = AutoConfig.from_pretrained(
            pretrained_name, trust_remote_code=trust_remote_code, cache_dir=cache_dir
        )
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            pretrained_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self._arch = detect_arch(self.hf_model)
        self._lora_config = lora_config

        self._inject_lora()
        self._build_block_views()

        # Compatibility shim so modules written against StudentModel.config.*
        # still work (max_seq_len, vocab_size, dim, lora).
        self.config = self._build_config_shim()

    # -- injection & views -------------------------------------------------

    def _inject_lora(self) -> None:
        targets = set(self._lora_config.target_modules)
        layers = getattr_path(self.hf_model, self._arch.layers_path)
        for layer in layers:
            attn = getattr(layer, self._arch.attn_attr)
            ffn = getattr(layer, self._arch.ffn_attr)
            for short, hf_name in self._arch.attn_name_map.items():
                if short in targets:
                    _inject_into(
                        attn, hf_name,
                        rank=self._lora_config.rank,
                        alpha=self._lora_config.alpha,
                        dropout=self._lora_config.dropout,
                    )
            for short, hf_name in self._arch.ffn_name_map.items():
                if short in targets:
                    _inject_into(
                        ffn, hf_name,
                        rank=self._lora_config.rank,
                        alpha=self._lora_config.alpha,
                        dropout=self._lora_config.dropout,
                    )

    def _build_block_views(self) -> None:
        layers = getattr_path(self.hf_model, self._arch.layers_path)
        views: list[SimpleNamespace] = []
        for layer in layers:
            attn_real = getattr(layer, self._arch.attn_attr)
            ffn_real = getattr(layer, self._arch.ffn_attr)

            attn_view = ProjView(self._arch.attn_name_map)
            for short, hf_name in self._arch.attn_name_map.items():
                mod = getattr(attn_real, hf_name, None)
                if mod is not None:
                    attn_view.attach(short, mod)

            ffn_view = ProjView(self._arch.ffn_name_map)
            for short, hf_name in self._arch.ffn_name_map.items():
                mod = getattr(ffn_real, hf_name, None)
                if mod is not None:
                    ffn_view.attach(short, mod)

            views.append(SimpleNamespace(attn=attn_view, ffn=ffn_view))
        self.blocks = views

    def _build_config_shim(self) -> StudentConfig:
        hfc = self.hf_config
        max_seq = getattr(hfc, "max_position_embeddings", 2048)
        # Use num_hidden_layers for layers count; hidden_size for dim.
        return StudentConfig(
            vocab_size=int(hfc.vocab_size),
            dim=int(getattr(hfc, "hidden_size", 0)),
            num_layers=int(getattr(hfc, "num_hidden_layers", 0)),
            heads=int(getattr(hfc, "num_attention_heads", 0)),
            ffn_hidden_multiplier=4,
            max_seq_len=int(max_seq),
            lora=self._lora_config,
        )

    # -- StudentModel-compatible API --------------------------------------

    @property
    def embed(self) -> nn.Embedding:
        return getattr_path(self.hf_model, self._arch.embed_path)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(input_ids=tokens, use_cache=False, return_dict=True)
        return out.logits

    def embed_for_routing(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.embed(tokens)
            return emb.mean(dim=1)

    def get_all_lora_modules(self) -> dict[str, LoRALinear]:
        result: dict[str, LoRALinear] = {}
        for i, block in enumerate(self.blocks):
            for name, mod in block.attn.get_lora_modules().items():
                result[f"blocks.{i}.attn.{name}"] = mod
            for name, mod in block.ffn.get_lora_modules().items():
                result[f"blocks.{i}.ffn.{name}"] = mod
        return result

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        from .lora_adapter import iter_lora_adapters

        lora_params: list[nn.Parameter] = []
        base_params: list[nn.Parameter] = []
        shared_params: list[nn.Parameter] = []
        lora_ids: set[int] = set()
        base_ids: set[int] = set()

        for _name, adapter in iter_lora_adapters(self):
            for p in adapter.lora_params:
                lora_params.append(p)
                lora_ids.add(id(p))
            for p in adapter.base_params:
                base_params.append(p)
                base_ids.add(id(p))

        for p in self.parameters():
            pid = id(p)
            if pid not in lora_ids and pid not in base_ids:
                shared_params.append(p)

        return {"P": base_params, "F": lora_params, "shared": shared_params}

    def merge_all_lora(self) -> None:
        from .lora_adapter import iter_lora_adapters

        for _name, adapter in iter_lora_adapters(self):
            adapter.merge_lora_into_base()

    def reset_all_lora(self) -> None:
        from .lora_adapter import iter_lora_adapters

        for _name, adapter in iter_lora_adapters(self):
            adapter.reset_lora()
