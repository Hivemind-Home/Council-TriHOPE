"""Evaluation harness: per-domain loss, forgetting, LoRA sparsity."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from hivemind.data.collate import DistillCollator
from hivemind.data.teacher_cache import TeacherLogitsCache
from hivemind.evaluation import (
    EvaluationConfig,
    ForgettingTracker,
    run_evaluation,
)
from hivemind.stores.retrieval import RetrievalStore
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel


def _build_student(vocab):
    cfg = StudentConfig(
        vocab_size=vocab, dim=32, num_layers=2, heads=4, max_seq_len=48,
        lora=LoRAConfig(rank=4, alpha=8.0,
                        target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    return StudentModel(cfg)


def test_run_evaluation_produces_per_domain_metrics(whitespace_tokenizer, hf_sample_rows):
    student = _build_student(whitespace_tokenizer.vocab_size)
    cache = TeacherLogitsCache(root=None)
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=48, logits_cache=cache
    )

    code_rows = [r for r in hf_sample_rows if r["domain"] == "code"]
    math_rows = [r for r in hf_sample_rows if r["domain"] == "math"]
    loaders = {
        "code": DataLoader(code_rows, batch_size=2, shuffle=False, collate_fn=collator),
        "math": DataLoader(math_rows, batch_size=2, shuffle=False, collate_fn=collator),
    }

    config = EvaluationConfig(enabled=True, max_batches=4, track_forgetting=True)
    tracker = ForgettingTracker()
    out = run_evaluation(
        student=student,
        val_loaders=loaders,
        device=torch.device("cpu"),
        config=config,
        forgetting=tracker,
    )

    for domain in ("code", "math"):
        assert f"eval/{domain}/loss" in out
        assert f"eval/{domain}/ppl" in out
        assert out[f"eval/{domain}/tokens"] > 0
    assert "eval/macro_loss" in out


def test_forgetting_delta_appears_on_second_pass(whitespace_tokenizer, hf_sample_rows):
    student = _build_student(whitespace_tokenizer.vocab_size)
    cache = TeacherLogitsCache(root=None)
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=32, logits_cache=cache
    )
    code_rows = [r for r in hf_sample_rows if r["domain"] == "code"]
    loaders = {"code": DataLoader(code_rows, batch_size=2, shuffle=False, collate_fn=collator)}
    config = EvaluationConfig(enabled=True, max_batches=2, track_forgetting=True)
    tracker = ForgettingTracker()

    first = run_evaluation(student=student, val_loaders=loaders, device=torch.device("cpu"),
                           config=config, forgetting=tracker)
    assert "eval/code/forgetting_delta" not in first  # baseline only

    # Nudge the model so the second pass has a different loss
    tokens = torch.randint(0, whitespace_tokenizer.vocab_size, (2, 16))
    opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        student(tokens).sum().backward()
        opt.step()

    second = run_evaluation(student=student, val_loaders=loaders, device=torch.device("cpu"),
                            config=config, forgetting=tracker)
    assert "eval/code/forgetting_delta" in second


def test_retrieval_hit_rate_reports_zero_for_empty_store(whitespace_tokenizer, hf_sample_rows):
    student = _build_student(whitespace_tokenizer.vocab_size)
    cache = TeacherLogitsCache(root=None)
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=16, logits_cache=cache
    )
    loaders = {
        "code": DataLoader(
            [r for r in hf_sample_rows if r["domain"] == "code"][:2],
            batch_size=1, collate_fn=collator,
        )
    }
    out = run_evaluation(
        student=student, val_loaders=loaders, device=torch.device("cpu"),
        config=EvaluationConfig(enabled=True, max_batches=2),
        r_store=RetrievalStore(max_size=4),
        probe_embeddings=torch.randn(2, 32),
    )
    assert out["eval/retrieval/buffer_size"] == 0
    assert out["eval/retrieval/hit_rate"] == 0.0
