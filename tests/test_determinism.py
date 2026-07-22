"""Tests for reproducibility / determinism."""

import torch

from hivemind.data import SyntheticTeacherSeedData
from hivemind.distillation import DistillationConfig, compute_distillation_objective
from hivemind.embedding import SharedEmbedding
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel
from hivemind.teacher_registry import TeacherRegistry, create_synthetic_teachers
from hivemind.teacher_router import TeacherRouter, batch_teacher_forward


def _run_n_steps(seed: int, n: int = 3) -> list[float]:
    """Run n training steps with a given seed, return losses."""
    torch.manual_seed(seed)

    config = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        max_seq_len=16,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    student = StudentModel(config)
    teachers = create_synthetic_teachers(2, config, torch.device("cpu"))
    registry = TeacherRegistry(teachers)
    routing_embed = SharedEmbedding(student.embed)
    seed_data = SyntheticTeacherSeedData(2, 5, 64, 16)
    registry.compute_prototypes(routing_embed, seed_data.get_seed_data())
    router = TeacherRouter()
    optimizer = MaskedAdamW(student.parameters(), lr=1e-3)
    dist_config = DistillationConfig(tau=4.0, lambda_kd=1.0, lambda_ce=0.5, lambda_reg=0.0)

    losses = []
    student.train()
    gen = torch.Generator().manual_seed(seed + 100)

    for _ in range(n):
        tokens = torch.randint(0, 64, (2, 16), generator=gen)
        h_t = routing_embed(tokens)
        teacher_indices, _ = router.route(h_t, registry.prototypes)
        teacher_logits = batch_teacher_forward(tokens, teacher_indices, teachers, 64)
        optimizer.zero_grad()
        student_logits = student(tokens)
        loss, _ = compute_distillation_objective(
            teacher_logits, student_logits, tokens, dist_config
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def test_same_seed_same_losses():
    """Two runs with the same seed produce identical losses."""
    losses_1 = _run_n_steps(seed=42, n=5)
    losses_2 = _run_n_steps(seed=42, n=5)

    for i, (l1, l2) in enumerate(zip(losses_1, losses_2)):
        assert abs(l1 - l2) < 1e-6, f"Step {i}: {l1} != {l2}"


def test_different_seed_different_losses():
    """Two runs with different seeds produce different losses."""
    losses_1 = _run_n_steps(seed=42, n=3)
    losses_2 = _run_n_steps(seed=123, n=3)

    # At least one loss should differ
    assert any(abs(l1 - l2) > 1e-6 for l1, l2 in zip(losses_1, losses_2))
