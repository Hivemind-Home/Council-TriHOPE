"""Student model: Transformer with LoRA adapters and tri-store parameter separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import StudentConfig
from .ffn import SwiGLUFFN
from .lora import LoRALinear


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    """Single transformer block: RMSNorm → Attention → RMSNorm → FFN (pre-norm)."""

    def __init__(self, config: StudentConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attn = CausalSelfAttention(
            dim=config.dim,
            num_heads=config.heads,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            dropout=config.dropout,
            lora_config=config.lora,
        )
        self.ffn = SwiGLUFFN(
            dim=config.dim,
            multiplier=config.ffn_hidden_multiplier,
            dropout=config.dropout,
            lora_config=config.lora,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StudentModel(nn.Module):
    """Transformer student model with LoRA adapters.

    Architecture: Embedding → L × TransformerBlock → RMSNorm → LM Head (tied to embedding).
    LoRA adapters on attention (Q, K, V, O) and FFN (gate, up, down) projections.
    """

    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.dim)

        # LM head tied to embedding
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize non-LoRA weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and not isinstance(module, type(None)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: [B, T] token indices.

        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.embed(tokens)  # [B, T, dim]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)  # [B, T, vocab_size]

    def embed_for_routing(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute routing embedding by mean-pooling token embeddings.

        Args:
            tokens: [B, T] token indices.

        Returns:
            [B, dim] routing vectors.
        """
        with torch.no_grad():
            emb = self.embed(tokens)  # [B, T, dim]
            return emb.mean(dim=1)  # [B, dim]

    def get_all_lora_modules(self) -> dict[str, LoRALinear]:
        """Get all LoRA modules with their full path names."""
        result = {}
        for i, block in enumerate(self.blocks):
            for name, module in block.attn.get_lora_modules().items():
                result[f"blocks.{i}.attn.{name}"] = module
            for name, module in block.ffn.get_lora_modules().items():
                result[f"blocks.{i}.ffn.{name}"] = module
        return result

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Separate parameters into P-store (base), F-store (LoRA), and shared groups.

        Uses the ``LoRAAdapter`` protocol so this is identical for our
        ``LoRALinear`` and any PEFT/Unsloth-attached LoRA.
        """
        from .lora_adapter import iter_lora_adapters

        lora_params: list[nn.Parameter] = []
        base_params: list[nn.Parameter] = []
        shared_params: list[nn.Parameter] = []

        lora_param_ids: set[int] = set()
        base_param_ids: set[int] = set()

        for _name, adapter in iter_lora_adapters(self):
            for p in adapter.lora_params:
                lora_params.append(p)
                lora_param_ids.add(id(p))
            for p in adapter.base_params:
                base_params.append(p)
                base_param_ids.add(id(p))

        for p in self.parameters():
            pid = id(p)
            if pid not in lora_param_ids and pid not in base_param_ids:
                shared_params.append(p)

        return {"P": base_params, "F": lora_params, "shared": shared_params}

    def merge_all_lora(self) -> None:
        """Merge all LoRA adapters into base weights (F→P consolidation)."""
        from .lora_adapter import iter_lora_adapters

        for _name, adapter in iter_lora_adapters(self):
            adapter.merge_lora_into_base()

    def reset_all_lora(self) -> None:
        """Reset all LoRA adapters to initial state."""
        from .lora_adapter import iter_lora_adapters

        for _name, adapter in iter_lora_adapters(self):
            adapter.reset_lora()
