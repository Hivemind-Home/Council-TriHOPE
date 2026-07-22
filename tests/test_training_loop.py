"""Integration test for the training loop."""

import torch

from hivemind.controller.config import ControllerConfig
from hivemind.controller.module_index import build_module_index
from hivemind.controller.policy import RFPPolicy
from hivemind.controller.signals import SignalComputer
from hivemind.data import SyntheticTeacherSeedData
from hivemind.distillation import DistillationConfig, compute_distillation_objective
from hivemind.embedding import SharedEmbedding
from hivemind.optim.masked_adamw import MaskedAdamW
from hivemind.student.config import LoRAConfig, StudentConfig
from hivemind.student.model import StudentModel
from hivemind.teacher_registry import TeacherRegistry, create_synthetic_teachers
from hivemind.teacher_router import TeacherRouter, batch_teacher_forward


def test_three_step_smoke():
    """Run 3 training steps and verify loss is finite."""
    torch.manual_seed(1337)

    # Build student
    config = StudentConfig(
        vocab_size=64, dim=32, num_layers=2, heads=4,
        max_seq_len=16,
        lora=LoRAConfig(
            rank=4, alpha=8.0, target_modules=["q", "k", "v", "o", "up", "gate", "down"]
        ),
    )
    student = StudentModel(config)

    # Build teachers
    teachers = create_synthetic_teachers(2, config, torch.device("cpu"))
    registry = TeacherRegistry(teachers)

    # Routing
    routing_embed = SharedEmbedding(student.embed)
    seed_data = SyntheticTeacherSeedData(2, 5, 64, 16)
    registry.compute_prototypes(routing_embed, seed_data.get_seed_data())
    router = TeacherRouter()

    # Optimizer
    optimizer = MaskedAdamW(student.parameters(), lr=1e-3)

    # Controller
    ctrl_config = ControllerConfig()
    module_index = build_module_index(student)
    signal_computer = SignalComputer(ctrl_config, module_index)
    policy = RFPPolicy(ctrl_config.policy)

    # Distillation config
    dist_config = DistillationConfig(tau=4.0, lambda_kd=1.0, lambda_ce=0.5, lambda_reg=0.0)

    losses = []
    student.train()

    for step in range(3):
        tokens = torch.randint(0, 64, (2, 16))

        # Route
        h_t = routing_embed(tokens)
        teacher_indices, _ = router.route(h_t, registry.prototypes)

        # Teacher forward
        teacher_logits = batch_teacher_forward(tokens, teacher_indices, teachers, 64)

        # Student forward + loss
        optimizer.zero_grad()
        student_logits = student(tokens)
        loss, metrics = compute_distillation_objective(
            teacher_logits, student_logits, tokens, dist_config
        )

        # Backward
        loss.backward()

        # Compute signals
        bucket_id = teacher_indices[0].item()
        module_signals = signal_computer.compute_all(
            optimizer, module_index, bucket_id, h_t[0]
        )

        # Policy
        policy.decide(module_signals)

        # Step
        optimizer.step()

        losses.append(loss.item())
        assert torch.isfinite(loss), f"Loss is not finite at step {step}"

    # Verify we got 3 finite losses
    assert len(losses) == 3
    assert all(loss > 0 for loss in losses)


def test_metrics_keys():
    """Training step produces expected metric keys."""
    torch.manual_seed(42)

    config = StudentConfig(
        vocab_size=64, dim=32, num_layers=1, heads=4,
        max_seq_len=16,
        lora=LoRAConfig(rank=4, target_modules=["q", "k", "v", "o", "up", "gate", "down"]),
    )
    student = StudentModel(config)

    teacher_logits = torch.randn(2, 16, 64)
    student_logits = student(torch.randint(0, 64, (2, 16)))

    dist_config = DistillationConfig(tau=4.0, lambda_kd=1.0, lambda_ce=0.5, lambda_reg=0.01)
    reg_loss = torch.tensor(0.1)

    _, metrics = compute_distillation_objective(
        teacher_logits, student_logits, torch.randint(0, 64, (2, 16)),
        dist_config, reg_loss,
    )

    assert "loss/kd" in metrics
    assert "loss/ce" in metrics
    assert "loss/reg" in metrics
    assert "loss/total" in metrics
