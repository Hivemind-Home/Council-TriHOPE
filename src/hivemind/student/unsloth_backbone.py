"""Unsloth-backed student.

Wraps ``unsloth.FastLanguageModel`` so the rest of the pipeline operates
on a real Qwen3.5 (or other Unsloth-supported) backbone with PEFT LoRA
adapters. The controller talks to LoRA via the ``LoRAAdapter`` protocol
in ``student.lora_adapter``, which transparently handles PEFT's
``lora.Linear`` layout.

Why this exists alongside ``HFStudent``:

* ``HFStudent`` uses raw ``transformers.AutoModelForCausalLM`` and our
  in-repo ``LoRALinear`` injection. Simple, no extra deps.
* ``UnslothStudent`` uses Unsloth's ``FastLanguageModel.from_pretrained``
  + ``get_peft_model``. Wins are big (sub-2x VRAM, ~1.5x training
  throughput) but only available via PEFT — hence the adapter abstraction.

Per Unsloth's Qwen3.5 docs, ``load_in_4bit=False`` is recommended (the
quantization differences hurt fine-tune quality); we default to bf16
LoRA. Override via ``model.dtype`` in the config.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from ._hf_common import ProjView, detect_arch, getattr_path
from .config import LoRAConfig, StudentConfig


class UnslothStudent(nn.Module):
    """``FastLanguageModel`` wrapped with the same ``blocks`` view HFStudent uses."""

    def __init__(
        self,
        pretrained_name: str,
        lora_config: LoRAConfig,
        *,
        max_seq_len: int = 2048,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        load_in_16bit: bool = True,
        full_finetuning: bool = False,
        use_gradient_checkpointing: str | bool = "unsloth",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        from unsloth import FastLanguageModel  # type: ignore

        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        self.tokenizer = None
        self.hf_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=pretrained_name,
            max_seq_length=max_seq_len,
            dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_16bit=load_in_16bit,
            full_finetuning=full_finetuning,
            cache_dir=cache_dir,
        )

        # Translate our short LoRA target names into PEFT-compatible HF names.
        target_modules = _expand_targets(lora_config.target_modules)

        self.hf_model = FastLanguageModel.get_peft_model(
            self.hf_model,
            r=int(lora_config.rank),
            target_modules=target_modules,
            lora_alpha=float(lora_config.alpha),
            lora_dropout=float(lora_config.dropout),
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # PEFT often wraps the model — drill down to the underlying causal LM
        # so detect_arch/getattr_path can find ``model.layers``. Try a few
        # common attribute paths.
        self._underlying = _find_underlying(self.hf_model)
        self._arch = detect_arch(self._underlying)
        self._lora_config = lora_config
        self._build_block_views()
        self.config = self._build_config_shim()

    # -- internals ---------------------------------------------------------

    def _build_block_views(self) -> None:
        layers = getattr_path(self._underlying, self._arch.layers_path)
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
        hfc = self._underlying.config
        max_seq = getattr(hfc, "max_position_embeddings", 2048)
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
        return getattr_path(self._underlying, self._arch.embed_path)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(input_ids=tokens, use_cache=False, return_dict=True)
        return out.logits

    def embed_for_routing(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.embed(tokens)
            return emb.mean(dim=1)

    def get_all_lora_modules(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
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


# -- helpers --------------------------------------------------------------


_TARGET_MAP = {
    "q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj",
    "gate": "gate_proj", "up": "up_proj", "down": "down_proj",
}


def _expand_targets(short_names: list[str]) -> list[str]:
    """Translate ``["q","k","v","o","gate","up","down"]`` → HF projection names."""
    return [_TARGET_MAP.get(n, n) for n in short_names]


def _find_underlying(peft_model: nn.Module) -> nn.Module:
    """Drill through PEFT's wrapper layers to the causal LM with ``.model.layers``.

    ``get_peft_model`` returns a ``PeftModel`` that exposes the underlying
    causal LM at ``.base_model.model``; older PEFT versions may differ.
    We probe a small list of paths.
    """
    candidates = [
        "base_model.model",       # PEFT >= 0.4
        "model",                   # already-unwrapped
        "base_model",              # very old PEFT
    ]
    for path in candidates:
        try:
            cur = peft_model
            for part in path.split("."):
                cur = getattr(cur, part)
            # detect_arch will throw if this doesn't have ``.model.layers``
            detect_arch(cur)
            return cur
        except (AttributeError, NotImplementedError):
            continue
    # Last resort: assume peft_model itself is fine
    return peft_model
