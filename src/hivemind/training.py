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
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .checkpoint import CheckpointConfig, CheckpointManager
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
from .evaluation import EvaluationConfig, ForgettingTracker, run_evaluation
from .logging_utils import BaseLogger, init_logger
from .optim.factory import build_optimizer
from .optim.masked_adamw import MaskedAdamW
from .regularization import RegularizationConfig, compute_total_regularization
from .stores.fast import FastStore
from .stores.permanent import PermanentStore
from .stores.retrieval import RetrievalStore
from .tracing import EventTrace, ModuleLedger
from .student.config import LoRAConfig, StudentConfig
from .student.model import StudentModel
from .student.hf_backbone import HFStudent
from .student.unsloth_backbone import UnslothStudent
from .teacher_hf import create_hf_live_teachers, parse_hf_teacher_specs
from .teacher_registry import (
    TeacherRegistry,
    create_hf_cache_teachers,
    create_synthetic_teachers,
)
from .teacher_router import MetadataRouter, TeacherRouter, batch_teacher_forward


def _seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_student(cfg: DictConfig, device: torch.device) -> nn.Module:
    """Build student model from config.

    Supports two backbones:
    - ``in_repo`` (default): the small hand-rolled StudentModel, used by
      the existing CPU smoke test.
    - ``hf``: wraps a pretrained ``AutoModelForCausalLM`` (Qwen / Llama
      family) and injects LoRA in-place on the target projections.
    """
    model_cfg = cfg.model
    lora_cfg = LoRAConfig(
        rank=model_cfg.lora.get("rank", 8),
        alpha=model_cfg.lora.get("alpha", 16.0),
        dropout=model_cfg.lora.get("dropout", 0.0),
        target_modules=list(model_cfg.lora.get("target_modules", ["q", "k", "v", "o", "up", "gate", "down"])),
    )

    backbone = str(model_cfg.get("backbone", "in_repo"))
    if backbone == "hf":
        pretrained = model_cfg.get("pretrained_name")
        if not pretrained:
            raise ValueError("model.backbone=hf requires model.pretrained_name")
        student = HFStudent(
            pretrained_name=str(pretrained),
            lora_config=lora_cfg,
            dtype=str(model_cfg.get("dtype", "float32")),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            cache_dir=model_cfg.get("hf_cache_dir"),
        )
        return student.to(device)

    if backbone == "unsloth":
        pretrained = model_cfg.get("pretrained_name")
        if not pretrained:
            raise ValueError("model.backbone=unsloth requires model.pretrained_name")
        # UnslothStudent calls FastLanguageModel which already places the
        # model on GPU per its own config — but be explicit for safety.
        student = UnslothStudent(
            pretrained_name=str(pretrained),
            lora_config=lora_cfg,
            max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
            dtype=str(model_cfg.get("dtype", "bfloat16")),
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            load_in_16bit=bool(model_cfg.get("load_in_16bit", True)),
            full_finetuning=bool(model_cfg.get("full_finetuning", False)),
            use_gradient_checkpointing=model_cfg.get(
                "use_gradient_checkpointing", "unsloth"
            ),
            cache_dir=model_cfg.get("hf_cache_dir"),
        )
        return student

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


def _bucket_to_int(bucket: Any, fallback: int) -> int:
    """Stable int id for a bucket string (non-negative, fits in python int).

    Controller repetition state keys are ints; the HF dataset gives strings.
    We use python ``hash`` modded into a large-but-fixed space. Collisions
    are rare and harmless — worst case two buckets share a repetition
    slot, which just couples their statistics.
    """
    if bucket is None or bucket == "":
        return int(fallback)
    return abs(hash(str(bucket))) % (2**31 - 1)


def _unpack_batch(raw: Any, device: torch.device) -> dict[str, Any]:
    """Normalize a batch into the shape the training step expects.

    Synthetic path: ``raw`` is a ``[B, T]`` token tensor. We expand into
    a dict where ``labels = input_ids`` (full-sequence CE) and teacher
    supervision is absent (set by caller from live teacher forward).

    HF path: ``raw`` is already a dict from ``DistillCollator``; we just
    move tensors to device.
    """
    if isinstance(raw, torch.Tensor):
        input_ids = raw.to(device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids,  # synthetic next-token loss
            "teacher_logits": None,
            "teacher_logits_mask": None,
            "sample_ids": None,
            "metadata": None,
        }

    # dict path — move tensors, pass through metadata
    out: dict[str, Any] = {}
    for key in (
        "input_ids",
        "attention_mask",
        "labels",
        "teacher_logits",
        "teacher_logits_mask",
        "teacher_confidence",
        "teacher_entropy",
    ):
        v = raw.get(key)
        if isinstance(v, torch.Tensor):
            out[key] = v.to(device)
        else:
            out[key] = v
    out["sample_ids"] = raw.get("sample_ids")
    out["metadata"] = raw.get("metadata")
    return out


def _sub_config(node: Any, key: str) -> dict[str, Any]:
    """Fetch a config sub-dict, tolerating missing keys and plain dicts."""
    val = node.get(key) if node is not None else None
    if val is None:
        return {}
    if OmegaConf.is_config(val):
        return OmegaConf.to_container(val, resolve=True)  # type: ignore[return-value]
    return dict(val)


def _build_controller_config(cfg: DictConfig) -> ControllerConfig:
    """Build controller config from hydra config."""
    ctrl = cfg.get("controller", None)
    return ControllerConfig(
        surprise=SurpriseConfig(**_sub_config(ctrl, "surprise")),
        stability=StabilityConfig(**_sub_config(ctrl, "stability")),
        repetition=RepetitionConfig(**_sub_config(ctrl, "repetition")),
        policy=PolicyConfig(**_sub_config(ctrl, "policy")),
        consolidation=ConsolidationConfig(**_sub_config(ctrl, "consolidation")),
        writer=WriterConfig(**_sub_config(ctrl, "writer")),
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

    # Data source decides the teacher kind + router kind.
    data_source = cfg.data.get("source", "synthetic")
    router_kind = cfg.data.get("router", "metadata" if data_source == "hf" else "cosine")

    # Teachers
    teachers_cfg = cfg.get("teachers", {})
    teacher_mode = str(teachers_cfg.get("mode", "cache" if data_source == "hf" else "synthetic"))
    if data_source == "hf" and teacher_mode == "live":
        raw_specs = teachers_cfg.get("pretrained", {})
        specs = parse_hf_teacher_specs(raw_specs)
        if not specs:
            raise ValueError(
                "teachers.mode=live requires teachers.pretrained "
                "(map of domain → pretrained_name, or a list of specs)."
            )
        teachers = create_hf_live_teachers(
            specs,
            vocab_size=cfg.model.vocab_size,
            device=device,
            dtype=str(teachers_cfg.get("dtype", "float32")),
            trust_remote_code=bool(teachers_cfg.get("trust_remote_code", False)),
            cache_dir=teachers_cfg.get("hf_cache_dir"),
        )
    elif data_source == "hf":
        domains = list(cfg.data.get("domains", ["code"]))
        teacher_ids = dict(teachers_cfg.get("teacher_ids", {}))
        teachers = create_hf_cache_teachers(
            domains=domains,
            teacher_ids=teacher_ids,
            vocab_size=cfg.model.vocab_size,
            device=device,
        )
    else:
        num_teachers = teachers_cfg.get("num_teachers", 2)
        teachers = create_synthetic_teachers(num_teachers, student.config, device)
    registry = TeacherRegistry(teachers)

    # Shared embedding for routing
    routing_embed = SharedEmbedding(student.embed)

    # Compute teacher prototypes.
    # - synthetic: random seed data, as before.
    # - hf: random seed until a future pass plumbs real input_text samples
    #   through the embedder. Metadata routing is the default for hf so
    #   prototypes are only used by the cosine-router ablation.
    seed_data = SyntheticTeacherSeedData(
        num_teachers=len(teachers),
        samples_per_teacher=teachers_cfg.get("prototype_seed_samples", 10),
        vocab_size=cfg.model.vocab_size,
        seq_len=cfg.data.get("seq_len", 32) if data_source == "synthetic" else 32,
    )
    # Re-key seed_data to match registered teacher names.
    named_seed: dict[str, list[torch.Tensor]] = {}
    raw_seed = seed_data.get_seed_data()
    for idx, t in enumerate(registry.teachers):
        named_seed[t.name] = raw_seed.get(f"teacher_{idx}", [])
    registry.compute_prototypes(routing_embed, named_seed)

    # Router
    if router_kind == "metadata":
        domain_to_index = {
            getattr(t.model, "domain", ""): i for i, t in enumerate(registry.teachers)
        }
        router: TeacherRouter | MetadataRouter = MetadataRouter(
            teacher_names=[t.name for t in registry.teachers],
            domain_to_index=domain_to_index,
        )
    else:
        router = TeacherRouter()

    # Optimizer
    optim_cfg = OmegaConf.to_container(cfg.optim, resolve=True)
    optimizer = build_optimizer(student, optim_cfg)
    # Exact write masking (paper Theorem 1): controller-indexed params are
    # default-closed and only update when an R/F/P action opens them.
    # ``optim.masked: false`` is an escape hatch that leaves every param
    # open — MaskedAdamW then reduces exactly to plain AdamW (Lemma 1).
    masked_updates = bool(optim_cfg.get("masked", True))

    # Distillation config
    dist_cfg = cfg.get("distillation", {})
    distill_config = DistillationConfig(
        tau=dist_cfg.get("tau", 4.0),
        lambda_kd=dist_cfg.get("lambda_kd", 1.0),
        lambda_ce=dist_cfg.get("lambda_ce", 0.5),
        lambda_reg=dist_cfg.get("lambda_reg", 0.01),
        use_teacher_confidence=dist_cfg.get("use_teacher_confidence", False),
        confidence_floor=dist_cfg.get("confidence_floor", 0.0),
        confidence_power=dist_cfg.get("confidence_power", 1.0),
    )

    # Regularization config
    reg_dict = cfg.get("regularization", {})
    reg_config = RegularizationConfig(
        weight_decay=reg_dict.get("weight_decay", 0.01),
        trust_region_weight=reg_dict.get("trust_region_weight", 0.0),
        anti_forgetting_weight=reg_dict.get("anti_forgetting_weight", 0.0),
        lora_sparsity_weight=reg_dict.get("lora_sparsity_weight", 0.0),
        consistency_weight=reg_dict.get("consistency_weight", 0.0),
    )

    # Controller
    ctrl_config = _build_controller_config(cfg)
    module_index = build_module_index(student)
    module_map = {m.id: m for m in module_index}
    signal_computer = SignalComputer(ctrl_config, module_index)
    policy = RFPPolicy(ctrl_config.policy)

    if masked_updates:
        # Controller-indexed params (attn/ffn base + LoRA) default to closed;
        # shared params (embeddings, norms, lm_head) stay ordinary-AdamW.
        optimizer.register_controller_params(
            p for m in module_index for p in m.params
        )

    # Stores
    r_store = RetrievalStore(max_size=ctrl_config.repetition.buffer_size)
    f_store = FastStore(
        top_k_granularity=ctrl_config.writer.top_k_granularity,
        top_k_fraction=ctrl_config.writer.top_k_fraction,
    )
    p_store = PermanentStore()

    consolidator = ConsolidationScheduler(
        ctrl_config.consolidation, student, p_store, optimizer=optimizer
    )
    writer = WriteExecutor(
        student,
        optimizer,
        r_store,
        f_store,
        p_store,
        ctrl_config.writer,
        consolidator=consolidator,
    )

    # Checkpointing
    ckpt_cfg_raw = cfg.get("checkpoint", {}) or {}
    ckpt_cfg = CheckpointConfig(
        enabled=bool(ckpt_cfg_raw.get("enabled", False)),
        dir=str(ckpt_cfg_raw.get("dir", "checkpoints")),
        save_every=int(ckpt_cfg_raw.get("save_every", 1000)),
        keep_last=int(ckpt_cfg_raw.get("keep_last", 3)),
        resume_from=ckpt_cfg_raw.get("resume_from"),
    )
    ckpt_manager = CheckpointManager(ckpt_cfg)

    # Data
    dataloader = build_dataloader(OmegaConf.to_container(cfg.data, resolve=True), distributed)

    # Per-domain validation loaders (HF path only; synthetic skips eval).
    eval_cfg_raw = cfg.get("eval", {}) or {}
    eval_config = EvaluationConfig(
        enabled=bool(eval_cfg_raw.get("enabled", False)),
        interval=int(eval_cfg_raw.get("interval", 500)),
        max_batches=int(eval_cfg_raw.get("max_batches", 32)),
        track_forgetting=bool(eval_cfg_raw.get("track_forgetting", True)),
        lora_sparsity_threshold=float(eval_cfg_raw.get("lora_sparsity_threshold", 1e-4)),
    )
    val_loaders: dict[str, Any] = {}
    if eval_config.enabled and data_source == "hf":
        base_data = OmegaConf.to_container(cfg.data, resolve=True)
        for domain in list(cfg.data.get("domains", [])):
            per_domain = dict(base_data)
            per_domain["split"] = eval_cfg_raw.get("split", "validation")
            per_domain["domains"] = [domain]
            per_domain["shuffle"] = False
            per_domain["drop_last"] = False
            per_domain["batch_size"] = int(eval_cfg_raw.get("batch_size", base_data.get("batch_size", 4)))
            per_domain["max_rows_per_domain"] = int(
                eval_cfg_raw.get("max_rows_per_domain", per_domain.get("max_rows_per_domain") or 256)
            )
            try:
                val_loaders[domain] = build_dataloader(per_domain, distributed=False)
            except RuntimeError as exc:
                print(f"eval: skipping domain {domain}: {exc}")
    forgetting_tracker = ForgettingTracker() if eval_config.track_forgetting else None

    # Logger
    log_cfg = OmegaConf.to_container(cfg.get("logging", {}), resolve=True) or {}
    logger = init_logger(log_cfg)

    # Event trace ("when and why parameters changed") + per-module ledger.
    trace_enabled = bool(log_cfg.get("enabled", False)) and bool(
        log_cfg.get("events_enabled", True)
    )
    event_trace = EventTrace(
        log_cfg.get("events_path", "logs/events.jsonl"), enabled=trace_enabled
    )
    ledger = ModuleLedger()

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

    # Resume if configured
    start_step = 0
    resume_path = ckpt_manager.resolve_resume_path()
    if resume_path is not None:
        meta = ckpt_manager.load(
            resume_path,
            student=student,
            optimizer=optimizer,
            signal_computer=signal_computer,
            r_store=r_store,
            consolidator=consolidator,
            forgetting=forgetting_tracker,
            ledger=ledger,
            map_location=str(device),
        )
        start_step = int(meta.get("step", 0)) + 1
        event_trace.emit(
            {"type": "resume", "step": start_step, "from": str(resume_path)}
        )
        print(f"Resumed from {resume_path} at step {start_step}.")

    # One self-describing config event per (re)start so the trace can be
    # analyzed without the Hydra config at hand.
    event_trace.emit(
        {
            "type": "run_config",
            "start_step": start_step,
            "top_m": ctrl_config.policy.top_m_modules,
            "top_k_fraction": ctrl_config.writer.top_k_fraction,
            "thresholds": {
                "surprise_high": ctrl_config.policy.surprise_high,
                "repetition_low": ctrl_config.policy.repetition_low,
                "repetition_medium": ctrl_config.policy.repetition_medium,
                "stability_high_C": ctrl_config.policy.stability_high_C,
                "stability_low_V": ctrl_config.policy.stability_low_V,
            },
            "consolidation": {
                "period": ctrl_config.consolidation.period,
                "min_repetition": ctrl_config.consolidation.min_repetition,
                "min_stability_C": ctrl_config.consolidation.min_stability_C,
                "strategy": ctrl_config.consolidation.merge_strategy,
            },
        }
    )

    student.train()
    pbar = tqdm(range(start_step, steps), desc="Training", initial=start_step, total=steps)

    # Track the last completed step so a SIGINT (Ctrl+C) or unexpected
    # exit can save a checkpoint at exactly where we stopped. The handler
    # for SIGINT lets one Ctrl+C flush state and exit cleanly; a second
    # one terminates immediately if the save itself hangs.
    last_step = start_step - 1

    def _emergency_save(reason: str) -> None:
        if not ckpt_cfg.enabled or last_step < 0:
            return
        try:
            print(f"\n[hivemind] {reason} — saving checkpoint at step={last_step}…")
            event_trace.flush()
            ckpt_manager.save(
                step=last_step,
                student=student,
                optimizer=optimizer,
                signal_computer=signal_computer,
                r_store=r_store,
                consolidator=consolidator,
                forgetting=forgetting_tracker,
                ledger=ledger,
                extra={"interrupted": reason},
            )
            print(f"[hivemind] checkpoint saved. Resume with checkpoint.resume_from=latest")
        except Exception as exc:  # noqa: BLE001 — best-effort on shutdown
            print(f"[hivemind] emergency save failed: {exc}")

    import signal

    _interrupted = {"flag": False}

    def _sigint_handler(_signum, _frame):
        if _interrupted["flag"]:
            # Second Ctrl+C — give up on graceful shutdown.
            print("\n[hivemind] second interrupt — exiting hard.")
            raise KeyboardInterrupt
        _interrupted["flag"] = True
        print("\n[hivemind] interrupt received — finishing current step then saving.")

    _prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    for step in pbar:
        # Get batch
        try:
            raw = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            raw = next(data_iter)

        batch = _unpack_batch(raw, device)
        tokens = batch["input_ids"]  # [B, T]

        # === Step 1: Embed for routing ===
        h_t = routing_embed(tokens)  # [B, d]

        # === Step 2: Route to teacher ===
        if isinstance(router, MetadataRouter):
            if batch["metadata"] is None:
                raise RuntimeError(
                    "router='metadata' requires HF-style batches with a metadata dict."
                )
            teacher_indices = router.route(batch["metadata"], device=device)
            scores = None
        else:
            teacher_indices, scores = router.route(
                h_t, registry.prototypes.to(device)
            )

        # === Step 3: Teacher forward (cache fast-path if batch carries logits) ===
        cached_logits = batch.get("teacher_logits")
        cached_mask = batch.get("teacher_logits_mask")
        teacher_logits = batch_teacher_forward(
            tokens,
            teacher_indices,
            registry.teachers,
            cfg.model.vocab_size,
            cached_logits=cached_logits,
            cached_logits_mask=cached_mask,
        )

        # === Step 4-6: Student forward + loss + backward ===
        optimizer.zero_grad()

        with autocast_ctx:
            student_logits = student(tokens)  # [B, T, V]

            # Regularization (incl. dropout-based consistency penalty if enabled)
            consistency_forward = (
                (lambda: student(tokens)) if reg_config.consistency_weight > 0 else None
            )
            reg_loss = compute_total_regularization(
                student,
                reg_config,
                consistency_forward=consistency_forward,
                consistency_attention_mask=batch.get("attention_mask"),
                consistency_labels=batch.get("labels"),
            )

            # Combined distillation objective
            loss, loss_metrics = compute_distillation_objective(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                targets=tokens,
                config=distill_config,
                reg_loss=reg_loss,
                labels=batch.get("labels"),
                attention_mask=batch.get("attention_mask"),
                teacher_logits_mask=cached_mask,
                teacher_confidence=batch.get("teacher_confidence"),
            )

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # === Step 7-8: Compute controller signals ===
        # Prefer the dataset-provided bucket when present (Theory 101 §4C);
        # fall back to the teacher index for the synthetic path.
        meta = batch.get("metadata")
        if meta and meta.get("bucket_id"):
            bucket_id = _bucket_to_int(meta["bucket_id"][0], fallback=0)
        else:
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

        # Resolve the teacher info for the representative (first) sample —
        # mirrors the choice already made for ``embedding`` and ``bucket_id``
        # above so the R-store entry is consistent. Theory 101 §7 lists
        # teacher_id and teacher soft targets among the R payload.
        teacher_idx_repr = (
            int(teacher_indices[0].item()) if teacher_indices.numel() > 0 else 0
        )
        teacher_name_repr = ""
        if 0 <= teacher_idx_repr < len(registry.teachers):
            teacher_name_repr = str(registry.teachers[teacher_idx_repr].name)
        teacher_text_repr = ""
        if meta and meta.get("teacher_output_text"):
            teacher_text_repr = str(meta["teacher_output_text"][0] or "")

        # === Step 10: Execute writes (build open-authorization masks) ===
        write_metrics, write_masks = writer.execute(
            actions=actions,
            module_map=module_map,
            step=step,
            embedding=embedding,
            bucket_id=bucket_id,
            teacher_id=teacher_idx_repr,
            teacher_name=teacher_name_repr,
            teacher_output_text=teacher_text_repr,
        )

        # Trace every routing decision + update the per-module ledger.
        action_by_module = {str(a.module_id): a for a in actions}
        for det in write_metrics["actions"]:
            ledger.record_action(
                det["module"], det["store"], step, det["coords_opened"]
            )
            act = action_by_module.get(det["module"])
            if act is None:
                continue
            event_trace.emit(
                {
                    "type": "decision",
                    "step": step,
                    "module": det["module"],
                    "action": det["store"],
                    "coords_opened": det["coords_opened"],
                    "bucket_id": bucket_id,
                    "teacher": teacher_name_repr,
                    "signals": {
                        "S": round(act.surprise, 6),
                        "C": round(act.stability_C, 6),
                        "V": round(act.stability_V, 6),
                        "R": round(act.repetition, 6),
                        "R_mom": round(act.repetition_components.get("mom", 0.0), 6),
                        "R_hash": round(act.repetition_components.get("hash", 0.0), 6),
                        "R_ret": round(act.repetition_components.get("ret", 0.0), 6),
                        "stability_adam": round(act.stability_adam, 6),
                        "grad_norm": round(act.grad_norm, 6),
                    },
                }
            )

        # === Step 11: Masked optimizer step ===
        # Coordinates not opened by an action stay bit-identical (params,
        # moments, weight decay, bias-correction clocks — Theorem 1).
        if masked_updates:
            optimizer.set_masks(write_masks)
        try:
            scaler.step(optimizer)
            scaler.update()
        finally:
            # A GradScaler inf/nan skip must not leak masks into the next step.
            optimizer.clear_masks()

        # === Step 12: Consolidation check ===
        # Record the batch for distill-based consolidation replay (no-op
        # for the direct strategy).
        consolidator.record_batch(tokens)
        consolidated = []
        if consolidator.should_check(step):
            consolidated = consolidator.consolidate(module_signals)
            for mid in consolidated:
                ledger.record_consolidation(str(mid), step)
                sig = module_signals.get(mid)
                event_trace.emit(
                    {
                        "type": "consolidation",
                        "step": step,
                        "module": str(mid),
                        "strategy": ctrl_config.consolidation.merge_strategy,
                        "pre_signals": (
                            {
                                "C": round(sig.stability_C, 6),
                                "R": round(sig.repetition, 6),
                            }
                            if sig is not None
                            else {}
                        ),
                        "forced": False,
                        "success": True,
                    }
                )

        # === Step 13: Logging ===
        if step % log_interval == 0 or step == steps - 1:
            # Aggregate signal stats
            surprises = [s.surprise for s in module_signals.values()]
            repetitions = [s.repetition for s in module_signals.values()]
            stabilities = [s.stability_C for s in module_signals.values()]

            adam_stabilities = [s.stability_adam for s in module_signals.values()]
            metrics = {
                **loss_metrics,
                "write/r_count": write_metrics["r_count"],
                "write/f_count": write_metrics["f_count"],
                "write/p_count": write_metrics["p_count"],
                "write/coords_opened": write_metrics["coords_opened"],
                "signals/surprise_mean": sum(surprises) / max(len(surprises), 1),
                "signals/repetition_mean": sum(repetitions) / max(len(repetitions), 1),
                "signals/stability_C_mean": sum(stabilities) / max(len(stabilities), 1),
                "signals/stability_adam_mean": (
                    sum(adam_stabilities) / max(len(adam_stabilities), 1)
                ),
                "retrieval/buffer_size": r_store.size,
                "consolidation/count": len(consolidated),
            }
            if isinstance(router, MetadataRouter):
                metrics["router/miss_count"] = router.miss_count

            pbar.set_postfix(
                loss=f"{loss_metrics['loss/total']:.4f}",
                R=write_metrics["r_count"],
                F=write_metrics["f_count"],
                P=write_metrics["p_count"],
            )
            logger.log(metrics, step)
            final_metrics = metrics

        # Evaluation
        if (
            eval_config.enabled
            and val_loaders
            and eval_config.interval > 0
            and step > 0
            and (step % eval_config.interval == 0 or step == steps - 1)
        ):
            eval_metrics = run_evaluation(
                student=student,
                val_loaders=val_loaders,
                device=device,
                config=eval_config,
                forgetting=forgetting_tracker,
                r_store=r_store,
                probe_embeddings=h_t.detach() if h_t.numel() > 0 else None,
            )
            logger.log(eval_metrics, step)
            # Surface key eval numbers on the progress bar
            macro = eval_metrics.get("eval/macro_loss")
            if macro is not None:
                pbar.set_postfix_str(f"val_loss={macro:.3f}", refresh=False)

        # Checkpoint
        if (
            ckpt_cfg.enabled
            and ckpt_cfg.save_every > 0
            and step > 0
            and (step % ckpt_cfg.save_every == 0 or step == steps - 1)
        ):
            event_trace.flush()
            ckpt_manager.save(
                step=step,
                student=student,
                optimizer=optimizer,
                signal_computer=signal_computer,
                r_store=r_store,
                consolidator=consolidator,
                forgetting=forgetting_tracker,
                ledger=ledger,
                extra={"final_metrics": final_metrics},
            )

        # Mark this step as fully complete (signals + writes + optimizer
        # step + maybe checkpoint all finished) so the SIGINT handler's
        # emergency save resumes from the right place.
        last_step = step
        if _interrupted["flag"]:
            break

    # Restore the prior signal handler before any more code runs.
    signal.signal(signal.SIGINT, _prev_handler)

    if _interrupted["flag"]:
        _emergency_save("SIGINT")

    event_trace.close()
    logger.close()
    if _interrupted["flag"]:
        print(f"\nTraining interrupted at step {last_step}. Resume with checkpoint.resume_from=latest")
    else:
        print(f"\nTraining complete. Final loss: {final_metrics.get('loss/total', 'N/A')}")
    return final_metrics
