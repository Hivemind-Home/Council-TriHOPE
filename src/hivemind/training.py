"""Main training loop orchestration.

Implements the full pipeline:
1. Embed → 2. Route teacher → 3. Teacher forward → 4. Student forward →
5. Loss → 6. Backward → 7. Read Adam states → 8. Compute signals →
9. R/F/P policy → 10. Execute writes → 11. Optimizer step → 12. Consolidation → 13. Log
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .config_utils import unwrap_config
from .controller.config import (
    ConsolidationConfig,
    ControllerConfig,
    PolicyConfig,
    RepetitionConfig,
    StabilityConfig,
    SurpriseConfig,
    WriterConfig,
)
from .controller.consolidation import ConsolidationScheduler
from .controller.module_index import ModuleId, ModuleInfo, build_module_index
from .controller.policy import RFPPolicy
from .controller.signals import SignalComputer
from .controller.writer import WriteExecutor
from .data import SyntheticTeacherSeedData, build_dataloader
from .distillation import DistillationConfig, compute_distillation_objective
from .embedding import SharedEmbedding
from .logging_utils import BaseLogger, init_logger
from .optim.factory import build_optimizer
from .optim.masked_adamw import MaskedAdamW
from .regularization import RegularizationConfig, compute_total_regularization
from .stores.fast import FastStore
from .stores.permanent import PermanentStore
from .stores.retrieval import RetrievalStore
from .student.config import LoRAConfig, StudentConfig
from .student.model import StudentModel
from .teacher_registry import TeacherRegistry, create_synthetic_teachers
from .teacher_router import TeacherRouter, batch_teacher_forward


def _seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_student(cfg: DictConfig, device: torch.device) -> StudentModel:
    """Build student model from config."""
    model_cfg = cfg.model
    lora_cfg = LoRAConfig(
        rank=model_cfg.lora.get("rank", 8),
        alpha=model_cfg.lora.get("alpha", 16.0),
        dropout=model_cfg.lora.get("dropout", 0.0),
        target_modules=list(model_cfg.lora.get("target_modules", ["q", "k", "v", "o", "up", "gate", "down"])),
    )
    student_cfg = StudentConfig(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        heads=model_cfg.heads,
        ffn_hidden_multiplier=model_cfg.get("ffn_hidden_multiplier", 4),
        activation=model_cfg.get("activation", "silu"),
        max_seq_len=model_cfg.get("max_seq_len", 2048),
        dropout=model_cfg.get("dropout", 0.0),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        lora=lora_cfg,
    )
    return StudentModel(student_cfg).to(device)


def _build_controller_config(cfg: DictConfig) -> ControllerConfig:
    """Build controller config from hydra config."""
    ctrl = cfg.get("controller", {})
    return ControllerConfig(
        surprise=SurpriseConfig(**OmegaConf.to_container(ctrl.get("surprise", {}), resolve=True)),
        stability=StabilityConfig(**OmegaConf.to_container(ctrl.get("stability", {}), resolve=True)),
        repetition=RepetitionConfig(**OmegaConf.to_container(ctrl.get("repetition", {}), resolve=True)),
        policy=PolicyConfig(**OmegaConf.to_container(ctrl.get("policy", {}), resolve=True)),
        consolidation=ConsolidationConfig(**OmegaConf.to_container(ctrl.get("consolidation", {}), resolve=True)),
        writer=WriterConfig(**OmegaConf.to_container(ctrl.get("writer", {}), resolve=True)),
    )


def run_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
    distributed: bool = False,
) -> dict[str, float]:
    """Main training loop with multi-teacher distillation and R/F/P routing.

    Args:
        cfg: full Hydra config.
        device: torch device.
        distributed: whether running distributed.

    Returns:
        Final metrics dict.
    """
    train_cfg = cfg.train
    seed = train_cfg.get("seed", 1337)
    _seed_everything(seed)

    # --- Build components ---
    student = _build_student(cfg, device)
    print(f"Student model: {sum(p.numel() for p in student.parameters()):,} params")

    # Teachers
    teachers_cfg = cfg.get("teachers", {})
    num_teachers = teachers_cfg.get("num_teachers", 2)
    teachers = create_synthetic_teachers(num_teachers, student.config, device)
    registry = TeacherRegistry(teachers)

    # Shared embedding for routing
    routing_embed = SharedEmbedding(student.embed)

    # Compute teacher prototypes from synthetic seed data
    seed_data = SyntheticTeacherSeedData(
        num_teachers=num_teachers,
        samples_per_teacher=teachers_cfg.get("prototype_seed_samples", 10),
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.data.get("seq_len", 32),
    )
    registry.compute_prototypes(routing_embed, seed_data.get_seed_data())

    # Router
    router = TeacherRouter()

    # Optimizer
    optimizer = build_optimizer(student, OmegaConf.to_container(cfg.optim, resolve=True))

    # Distillation config
    dist_cfg = cfg.get("distillation", {})
    distill_config = DistillationConfig(
        tau=dist_cfg.get("tau", 4.0),
        lambda_kd=dist_cfg.get("lambda_kd", 1.0),
        lambda_ce=dist_cfg.get("lambda_ce", 0.5),
        lambda_reg=dist_cfg.get("lambda_reg", 0.01),
    )

    # Regularization config
    reg_dict = cfg.get("regularization", {})
    reg_config = RegularizationConfig(
        weight_decay=reg_dict.get("weight_decay", 0.01),
        trust_region_weight=reg_dict.get("trust_region_weight", 0.0),
        anti_forgetting_weight=reg_dict.get("anti_forgetting_weight", 0.0),
        lora_sparsity_weight=reg_dict.get("lora_sparsity_weight", 0.0),
    )

    # Controller
    ctrl_config = _build_controller_config(cfg)
    module_index = build_module_index(student)
    module_map = {m.id: m for m in module_index}
    signal_computer = SignalComputer(ctrl_config, module_index)
    policy = RFPPolicy(ctrl_config.policy)

    # Stores
    r_store = RetrievalStore(max_size=ctrl_config.repetition.buffer_size)
    f_store = FastStore(
        top_k_granularity=ctrl_config.writer.top_k_granularity,
        top_k_fraction=ctrl_config.writer.top_k_fraction,
    )
    p_store = PermanentStore()

    writer = WriteExecutor(student, optimizer, r_store, f_store, p_store, ctrl_config.writer)
    consolidator = ConsolidationScheduler(ctrl_config.consolidation, student, p_store)

    # Data
    dataloader = build_dataloader(OmegaConf.to_container(cfg.data, resolve=True), distributed)

    # Logger
    log_cfg = OmegaConf.to_container(cfg.get("logging", {}), resolve=True)
    logger = init_logger(log_cfg)

    # Mixed precision
    mp_cfg = train_cfg.get("mixed_precision", {})
    use_amp = mp_cfg.get("enabled", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if mp_cfg.get("dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
    )

    # --- Training loop ---
    steps = train_cfg.get("steps", 100)
    log_interval = train_cfg.get("log_interval", 10)
    data_iter = iter(dataloader)
    final_metrics: dict[str, float] = {}

    student.train()
    pbar = tqdm(range(steps), desc="Training")

    for step in pbar:
        # Get batch
        try:
            tokens = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            tokens = next(data_iter)

        tokens = tokens.to(device)  # [B, T]

        # === Step 1: Embed for routing ===
        h_t = routing_embed(tokens)  # [B, d]

        # === Step 2: Route to teacher ===
        teacher_indices, scores = router.route(h_t, registry.prototypes.to(device))

        # === Step 3: Teacher forward (frozen, no grad) ===
        teacher_logits = batch_teacher_forward(
            tokens, teacher_indices, registry.teachers, cfg.model.vocab_size
        )

        # === Step 4-6: Student forward + loss + backward ===
        optimizer.zero_grad()

        with autocast_ctx:
            student_logits = student(tokens)  # [B, T, V]

            # Regularization
            reg_loss = compute_total_regularization(student, reg_config)

            # Combined distillation objective
            loss, loss_metrics = compute_distillation_objective(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                targets=tokens,  # self-supervised: predict next token
                config=distill_config,
                reg_loss=reg_loss,
            )

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # === Step 7-8: Compute controller signals ===
        # Use first teacher index in batch as bucket_id
        bucket_id = teacher_indices[0].item() if teacher_indices.numel() > 0 else 0
        embedding = h_t[0] if h_t.numel() > 0 else None

        module_signals = signal_computer.compute_all(
            optimizer=optimizer,
            modules=module_index,
            bucket_id=bucket_id,
            embedding=embedding,
        )

        # === Step 9: R/F/P routing policy ===
        actions = policy.decide(module_signals)

        # === Step 10: Execute writes ===
        write_metrics = writer.execute(
            actions=actions,
            module_map=module_map,
            step=step,
            embedding=embedding,
            bucket_id=bucket_id,
        )

        # === Step 11: Optimizer step (updates remaining params) ===
        scaler.step(optimizer)
        scaler.update()

        # === Step 12: Consolidation check ===
        consolidated = []
        if consolidator.should_check(step):
            consolidated = consolidator.consolidate(module_signals)

        # === Step 13: Logging ===
        if step % log_interval == 0 or step == steps - 1:
            # Aggregate signal stats
            surprises = [s.surprise for s in module_signals.values()]
            repetitions = [s.repetition for s in module_signals.values()]
            stabilities = [s.stability_C for s in module_signals.values()]

            metrics = {
                **loss_metrics,
                "write/r_count": write_metrics["r_count"],
                "write/f_count": write_metrics["f_count"],
                "write/p_count": write_metrics["p_count"],
                "signals/surprise_mean": sum(surprises) / max(len(surprises), 1),
                "signals/repetition_mean": sum(repetitions) / max(len(repetitions), 1),
                "signals/stability_C_mean": sum(stabilities) / max(len(stabilities), 1),
                "retrieval/buffer_size": r_store.size,
                "consolidation/count": len(consolidated),
            }

            pbar.set_postfix(
                loss=f"{loss_metrics['loss/total']:.4f}",
                R=write_metrics["r_count"],
                F=write_metrics["f_count"],
                P=write_metrics["p_count"],
            )
            logger.log(metrics, step)
            final_metrics = metrics

    logger.close()
    print(f"\nTraining complete. Final loss: {final_metrics.get('loss/total', 'N/A')}")
    return final_metrics
