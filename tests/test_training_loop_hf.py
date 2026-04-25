"""End-to-end HF-path smoke: dict batches through every pipeline stage.

Bypasses network + tokenizer downloads by building a dataloader directly
from the fixtures and calling the training step pieces. Confirms CE-only
training produces finite loss when the teacher logits cache misses
everywhere (the day-one default).
"""

from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader

from hivemind.controller.config import ControllerConfig
from hivemind.controller.module_index import build_module_index
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import SignalComputer
from hivemind.data.collate import DistillCollator
from hivemind.data.teacher_cache import TeacherLogitsCache
from hivemind.distillation import DistillationConfig, compute_distillation_objective
from hivemind.embedding import SharedEmbedding
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel
from hivemind.teacher_registry import TeacherRegistry, create_hf_cache_teachers
from hivemind.teacher_router import MetadataRouter, batch_teacher_forward
from hivemind.training import _bucket_to_int, _unpack_batch


def test_hf_batch_runs_full_step(whitespace_tokenizer, hf_sample_rows):
    torch.manual_seed(0)
    V = whitespace_tokenizer.vocab_size
    config = StudentConfig(
        vocab_size=V, dim=32, num_layers=2, heads=4, max_seq_len=48,
        lora=LoRAConfig(rank=4, alpha=8.0,
                        target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    student = StudentModel(config)
    device = torch.device("cpu")

    teachers = create_hf_cache_teachers(
        domains=["code", "math"],
        teacher_ids={
            "code": "code_teacher_deepseek_r1",
            "math": "math_teacher_deepseek_r1_1p5b",
        },
        vocab_size=V,
        device=device,
    )
    registry = TeacherRegistry(teachers)
    # compute_prototypes needs SOMETHING per teacher; wire up a tiny seed
    from hivemind.data import SyntheticTeacherSeedData
    seed = SyntheticTeacherSeedData(2, 2, V, 16)
    raw = seed.get_seed_data()
    named = {t.name: raw[f"teacher_{i}"] for i, t in enumerate(registry.teachers)}
    embed = SharedEmbedding(student.embed)
    registry.compute_prototypes(embed, named)

    router = MetadataRouter(
        teacher_names=[t.name for t in registry.teachers],
        domain_to_index={"code": 0, "math": 1},
    )

    cache = TeacherLogitsCache(root=None)  # always miss → CE-only
    collator = DistillCollator(
        tokenizer=whitespace_tokenizer, max_seq_len=48, logits_cache=cache
    )
    loader = DataLoader(
        hf_sample_rows, batch_size=3, shuffle=False, collate_fn=collator
    )

    optimizer = MaskedAdamW(student.parameters(), lr=1e-3)
    ctrl = ControllerConfig()
    module_index = build_module_index(student)
    signals = SignalComputer(ctrl, module_index)
    policy = RFPPolicy(ctrl.policy)
    dist_cfg = DistillationConfig(tau=4.0, lambda_kd=0.0, lambda_ce=1.0, lambda_reg=0.0)

    student.train()
    losses: list[float] = []

    for step, raw_batch in enumerate(loader):
        if step >= 3:
            break
        batch = _unpack_batch(raw_batch, device)
        tokens = batch["input_ids"]

        h_t = embed(tokens)
        teacher_indices = router.route(batch["metadata"], device=device)

        teacher_logits = batch_teacher_forward(
            tokens, teacher_indices, registry.teachers, V,
            cached_logits=batch["teacher_logits"],
            cached_logits_mask=batch["teacher_logits_mask"],
        )

        optimizer.zero_grad()
        student_logits = student(tokens)
        loss, metrics = compute_distillation_objective(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            targets=tokens,
            config=dist_cfg,
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
            teacher_logits_mask=batch["teacher_logits_mask"],
        )
        loss.backward()

        bucket_id = _bucket_to_int(batch["metadata"]["bucket_id"][0], 0)
        signals.compute_all(optimizer, module_index, bucket_id, h_t[0])

        optimizer.step()
        losses.append(loss.item())
        assert torch.isfinite(loss)

    assert len(losses) == 3
    assert all(l > 0 for l in losses)


def test_unpack_synthetic_path_preserves_legacy_contract():
    device = torch.device("cpu")
    tokens = torch.randint(0, 64, (3, 8))
    batch = _unpack_batch(tokens, device)
    assert batch["input_ids"].shape == (3, 8)
    assert (batch["labels"] == batch["input_ids"]).all()
    assert batch["teacher_logits"] is None
    assert batch["metadata"] is None
