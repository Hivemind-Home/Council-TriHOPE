"""Shared test fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def seed():
    """Set deterministic seed for tests."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def tiny_student_config():
    """Tiny student config for unit tests."""
    from hivemind.student.config import LoRAConfig, StudentConfig

    return StudentConfig(
        vocab_size=64,
        dim=32,
        num_layers=2,
        heads=4,
        ffn_hidden_multiplier=4,
        max_seq_len=32,
        lora=LoRAConfig(rank=4, alpha=8.0, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )


@pytest.fixture
def tiny_model(tiny_student_config, seed):
    """Tiny student model for unit tests."""
    from hivemind.student.model import StudentModel

    return StudentModel(tiny_student_config)


@pytest.fixture
def tiny_tokens(tiny_student_config):
    """Tiny token batch [2, 16]."""
    return torch.randint(0, tiny_student_config.vocab_size, (2, 16))
