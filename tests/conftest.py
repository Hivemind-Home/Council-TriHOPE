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
        lora=LoRAConfig(
            rank=4, alpha=8.0, target_modules=["q", "k", "v", "o", "up", "gate", "down"]
        ),
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


# --- HF-path fixtures -----------------------------------------------------


class _WhitespaceTokenizer:
    """Deterministic whitespace tokenizer for tests — no transformers required.

    Token ids are ``hash(word) % vocab_size`` so encoding is stable across
    runs. Reserves id 0 for padding.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        for word in text.split():
            tid = (abs(hash(word)) % (self.vocab_size - 2)) + 2
            out.append(tid)
        return out

    def __call__(self, text, **kwargs):  # pragma: no cover - unused by collator
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": [self.encode(t) for t in text]}


@pytest.fixture
def whitespace_tokenizer():
    return _WhitespaceTokenizer(vocab_size=256)


@pytest.fixture
def hf_sample_rows():
    """Ten synthetic joined rows spanning two domains."""
    rows = []
    for i in range(5):
        rows.append({
            "sample_id": f"code-{i:03d}",
            "input_text": f"write a function that computes the factorial of {i}",
            "target_text": f"def fact(n):\n    return 1 if n <= 1 else n * fact(n-1)  # {i}",
            "teacher_id": "code_teacher_deepseek_r1",
            "teacher_output_text": (
                f"Here is a factorial solution iteration {i}: "
                "for k in range n multiply accumulator"
            ),
            "teacher_logits_path": f"code/train/code-{i:03d}_code.npz",
            "domain": "code",
            "bucket_id": "code_competitive_programming_medium",
            "has_gold_label": True,
        })
    for i in range(5):
        rows.append({
            "sample_id": f"math-{i:03d}",
            "input_text": f"solve for x: 2x + {i} = {2 * i + 4}",
            "target_text": f"x = {(2 * i + 4 - i) // 2}",
            "teacher_id": "math_teacher_deepseek_r1_1p5b",
            "teacher_output_text": (
                f"Subtract {i} from both sides then divide by two "
                f"to find x equals {(2 * i + 4 - i) // 2}"
            ),
            "teacher_logits_path": f"math/train/math-{i:03d}_math.npz",
            "domain": "math",
            "bucket_id": "math_algebra_easy",
            "has_gold_label": True,
        })
    return rows


@pytest.fixture
def npz_cache_root(tmp_path, hf_sample_rows):
    """Write a small NPZ per sample so TeacherLogitsCache hits."""
    import numpy as np

    root = tmp_path / "teacher_logits"
    for row in hf_sample_rows[:6]:
        path = root / row["teacher_logits_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # shape [T, V] — 8 response tokens, vocab 256
        logits = np.random.randn(8, 256).astype(np.float32)
        np.savez(path, logits=logits)
    return root
