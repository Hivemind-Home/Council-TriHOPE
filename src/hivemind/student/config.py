"""Student model configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q", "k", "v", "o", "up", "gate", "down"]
    )


@dataclass
class StudentConfig:
    """Full student model configuration."""

    vocab_size: int = 32000
    dim: int = 512
    num_layers: int = 12
    heads: int = 8
    ffn_hidden_multiplier: int = 4
    activation: str = "silu"
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)
