"""Architecture probing + projection-view helpers shared by HF-shaped backbones.

Both ``HFStudent`` (vanilla ``transformers`` + our ``LoRALinear`` injection)
and ``UnslothStudent`` (Unsloth's ``FastLanguageModel`` + PEFT LoRA)
publish the same ``blocks`` view so ``build_module_index`` and the
controller see one shape regardless of provenance. The shared bits live
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class ArchSpec:
    """Names + attribute path for one supported HF architecture."""

    name: str
    layers_path: str           # e.g. "model.layers"
    attn_attr: str              # e.g. "self_attn"
    ffn_attr: str               # e.g. "mlp"
    embed_path: str            # e.g. "model.embed_tokens"
    attn_name_map: dict[str, str]  # hivemind-key → hf-name
    ffn_name_map: dict[str, str]


QWEN_LIKE = ArchSpec(
    name="qwen-like",
    layers_path="model.layers",
    attn_attr="self_attn",
    ffn_attr="mlp",
    embed_path="model.embed_tokens",
    attn_name_map={"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"},
    ffn_name_map={"gate": "gate_proj", "up": "up_proj", "down": "down_proj"},
)

ARCH_SPECS: tuple[ArchSpec, ...] = (QWEN_LIKE,)


def getattr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def detect_arch(hf_model: nn.Module) -> ArchSpec:
    """Probe for a spec whose paths all exist on the model."""
    for spec in ARCH_SPECS:
        try:
            layers = getattr_path(hf_model, spec.layers_path)
            first = layers[0]
            attn = getattr(first, spec.attn_attr)
            ffn = getattr(first, spec.ffn_attr)
            for hf_name in spec.attn_name_map.values():
                if not hasattr(attn, hf_name):
                    raise AttributeError
            for hf_name in spec.ffn_name_map.values():
                if not hasattr(ffn, hf_name):
                    raise AttributeError
            getattr_path(hf_model, spec.embed_path)
            return spec
        except (AttributeError, IndexError, TypeError):
            continue
    raise NotImplementedError(
        "Unknown architecture. Supported layouts: "
        + ", ".join(s.name for s in ARCH_SPECS)
        + ". Extend _hf_common.ARCH_SPECS to add your model."
    )


class ProjView:
    """Attribute container with the ``w_{q,k,v,o,gate,up,down}`` naming
    that ``build_module_index`` expects.

    Holds references to the underlying projection modules (which may be
    plain ``nn.Linear``, our ``LoRALinear``, or PEFT's ``lora.Linear``).
    """

    def __init__(self, name_map: dict[str, str]) -> None:
        self._name_map = name_map

    def attach(self, short: str, module: nn.Module) -> None:
        object.__setattr__(self, f"w_{short}", module)

    def get_lora_modules(self) -> dict[str, nn.Module]:
        """Return projections that have a LoRA adapter attached."""
        from .lora_adapter import get_adapter

        out: dict[str, nn.Module] = {}
        for short in self._name_map:
            m = getattr(self, f"w_{short}", None)
            if m is None:
                continue
            if get_adapter(m) is not None:
                out[f"w_{short}"] = m
        return out
