"""Main training loop orchestration.

Implements the full pipeline:
1. Embed → 2. Route teacher → 3. Teacher forward → 4. Student forward →
5. Loss → 6. Backward → 7. Read Adam states → 8. Compute signals →
9. R/F/P policy → 10. Execute writes → 11. Optimizer step → 12. Consolidation → 13. Log
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .checkpoint import CheckpointConfig, CheckpointManager
from .controller.config import (
    AblationConfig,
    ConsolidationConfig,
    ControllerConfig,
    DebugConfig,
    PolicyConfig,
    RepetitionConfig,
    RetrievalConfig,
    StabilityConfig,
    SurpriseConfig,
    WriterConfig,
)
from .controller.consolidation import ConsolidationScheduler
from .controller.module_index import ModuleId, build_module_index
from .controller.policy import RFPPolicy, StoreAction
from .controller.signals import ModuleSignals, SignalComputer
from .controller.writer import WriteExecutor
from .data import SyntheticTeacherSeedData, build_dataloader
from .data.collate import EmptyBatchError
from .data.hf_loader import domain_names
from .distillation import DistillationConfig, compute_distillation_objective
from .distributed import DistContext
from .embedding import SharedEmbedding
from .evaluation import (
    EvaluationConfig,
    ForgettingTracker,
    PhaseEvalTracker,
    exact_match_eval,
    run_evaluation,
)
from .logging_utils import init_logger
from .optim.factory import build_optimizer
from .profiling import RunProfiler, write_run_summary
from .regularization import RegularizationConfig, compute_total_regularization
from .stores.fast import FastStore
from .stores.permanent import PermanentStore
from .stores.retrieval import RetrievalStore
from .student.config import LoRAConfig, StudentConfig
from .student.hf_backbone import HFStudent
from .student.model import StudentModel
from .student.unsloth_backbone import UnslothStudent
from .teacher_hf import create_hf_live_teachers, parse_hf_teacher_specs
from .teacher_registry import (
    TeacherRegistry,
    create_hf_cache_teachers,
    create_synthetic_teachers,
)
from .teacher_router import MetadataRouter, TeacherRouter, batch_teacher_forward
from .tracing import EventTrace, ModuleLedger


def _as_container(node: Any) -> dict:
    """OmegaConf node or plain dict -> plain dict."""
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)  # type: ignore[return-value]
    return dict(node)


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
    lora_rank = int(model_cfg.lora.get("rank", 8))
    lora_cfg = LoRAConfig(
        rank=max(1, lora_rank),
        alpha=model_cfg.lora.get("alpha", 16.0),
        dropout=model_cfg.lora.get("dropout", 0.0),
        # rank <= 0 disables LoRA entirely (full-FT baseline: no adapters,
        # no F-store — plain base model).
        target_modules=(
            []
            if lora_rank <= 0
            else list(
                model_cfg.lora.get(
                    "target_modules", ["q", "k", "v", "o", "up", "gate", "down"]
                )
            )
        ),
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
    Uses blake2s (not python ``hash``, which is PYTHONHASHSEED-randomized
    per process) so bucket ids — and the repetition state keyed on them —
    survive restarts and resumes. Collisions are rare and harmless — worst
    case two buckets share a repetition slot, which just couples their
    statistics.
    """
    if bucket is None or bucket == "":
        return int(fallback)
    import hashlib

    digest = hashlib.blake2s(str(bucket).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


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
    enabled = True
    if ctrl is not None:
        enabled = bool(ctrl.get("enabled", True))
    return ControllerConfig(
        enabled=enabled,
        surprise=SurpriseConfig(**_sub_config(ctrl, "surprise")),
        stability=StabilityConfig(**_sub_config(ctrl, "stability")),
        repetition=RepetitionConfig(**_sub_config(ctrl, "repetition")),
        policy=PolicyConfig(**_sub_config(ctrl, "policy")),
        consolidation=ConsolidationConfig(**_sub_config(ctrl, "consolidation")),
        writer=WriterConfig(**_sub_config(ctrl, "writer")),
        ablation=AblationConfig(**_sub_config(ctrl, "ablation")),
        debug=DebugConfig(**_sub_config(ctrl, "debug")),
        retrieval=RetrievalConfig(**_sub_config(ctrl, "retrieval")),
    )


def _capture_replay_sample(
    batch: dict[str, Any], dist: DistContext, device: torch.device
) -> dict[str, Any]:
    """Representative row (row 0) of the batch, for R-store replay.

    Mirrors the choice already made for ``bucket_id`` / ``embedding``: row 0
    of the global batch. Under DDP that row lives on rank 0 only (contiguous
    sharding), and each rank pads to its own shard's length, so the row is
    broadcast from rank 0 — unconditionally, every step, so the collective
    can never depend on a per-rank decision. Tensors are returned on CPU;
    the R-store keeps them there.

    ``kd_row_mask`` is ``None`` on the synthetic path (the replay computes KD
    against the live teacher, as the main step does) and ``0.0`` for HF
    batches: cached logits are not stored, so a replayed HF row is CE-only.
    """
    ids = batch["input_ids"]
    labels = batch.get("labels")
    conf = batch.get("teacher_confidence")
    kd_row_mask = None if batch.get("teacher_logits") is None else 0.0
    if dist.enabled:
        T = int(ids.shape[1]) if dist.is_main else 0
        has_labels = labels is not None
        conf_val = None
        if dist.is_main and isinstance(conf, torch.Tensor) and conf.numel() > 0:
            conf_val = float(conf[0].item())
        T, has_labels, kd_row_mask, conf_val = dist.broadcast_obj(
            (T, has_labels, kd_row_mask, conf_val), src=0
        )
        coll = dist._coll_device()
        row_ids = (
            ids[0].detach().to(coll)
            if dist.is_main
            else torch.empty(T, dtype=torch.long, device=coll)
        )
        dist.broadcast_tensor_(row_ids)
        row_labels = None
        if has_labels:
            row_labels = (
                labels[0].detach().to(coll)
                if dist.is_main
                else torch.empty(T, dtype=torch.long, device=coll)
            )
            dist.broadcast_tensor_(row_labels)
        return {
            "input_ids": row_ids.cpu(),
            "labels": None if row_labels is None else row_labels.cpu(),
            "kd_row_mask": kd_row_mask,
            "teacher_confidence": conf_val,
        }
    conf_val = None
    if isinstance(conf, torch.Tensor) and conf.numel() > 0:
        conf_val = float(conf[0].item())
    return {
        "input_ids": ids[0].detach().cpu(),
        "labels": None if labels is None else labels[0].detach().cpu(),
        "kd_row_mask": kd_row_mask,
        "teacher_confidence": conf_val,
    }


def run_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
    distributed: bool = False,
    dist: DistContext | None = None,
) -> dict[str, float]:
    """Main training loop with multi-teacher distillation and R/F/P routing.

    Args:
        cfg: full Hydra config.
        device: torch device.
        distributed: deprecated alias kept for the CLI and existing tests;
            prefer ``dist``.
        dist: the process-group handle from
            :func:`hivemind.distributed.init_distributed`. Inert at
            ``world_size == 1``, which keeps the single-GPU path identical.

    Returns:
        Final metrics dict.

    Under DDP exactly ONE call site uses the wrapped module — the student
    forward. Everything else (module index, optimizer, stores, consolidator,
    eval, checkpoints) must see the raw module: DDP has no ``__getattr__``
    forwarding, so ``.blocks`` / ``.embed`` / ``.get_param_groups()`` raise
    through it, and the parameter objects are shared either way.
    """
    dist = dist if dist is not None else DistContext()
    is_main = dist.is_main
    train_cfg = cfg.train
    seed = train_cfg.get("seed", 1337)
    # The same seed on every rank is deliberate: it makes the initial
    # weights, and every later reset_lora() draw, identical without a
    # collective. All shipped configs run dropout=0, so the RNG streams do
    # not desynchronize; consolidation broadcasts lora_a anyway.
    _seed_everything(seed)

    # --- Build components ---
    student = _build_student(cfg, device)
    if is_main:
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
        # ``data.domains`` entries are DomainSpec dicts, not bare strings —
        # passing them through raw stringifies the whole dict into the
        # teacher id and makes both MetadataRouter lookups miss.
        domains = domain_names(
            OmegaConf.to_container(cfg.data.domains, resolve=True)
            if "domains" in cfg.data
            else ["code"]
        )
        teacher_ids = dict(teachers_cfg.get("teacher_ids", {}))
        missing_ids = [d for d in domains if d not in teacher_ids]
        if missing_ids:
            print(
                f"[warn] teachers.teacher_ids has no entry for {missing_ids}; "
                "those domains can only route by domain name."
            )
        # Corrupted-teacher stream (T4): register the tag as its own cache
        # teacher (same domain, distinct name) so MetadataRouter resolves it
        # by name and every decision / R entry is attributed to it.
        corrupt_raw = _as_container(cfg.data.get("corrupt_teacher")) or {}
        extra_teachers = (
            [(str(corrupt_raw.get("tag", "math_teacher_corrupted")),
              str(corrupt_raw.get("domain", "math")))]
            if corrupt_raw.get("enabled")
            else []
        )
        teachers = create_hf_cache_teachers(
            domains=domains,
            teacher_ids=teacher_ids,
            vocab_size=cfg.model.vocab_size,
            device=device,
            extra=extra_teachers,
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
        # First teacher wins per domain: the corrupted tag shares a domain
        # with the clean teacher, and a row with an unknown teacher_id must
        # fall back to the CLEAN one.
        domain_to_index: dict[str, int] = {}
        for i, t in enumerate(registry.teachers):
            domain_to_index.setdefault(getattr(t.model, "domain", ""), i)
        # Live teachers: a routing miss silently distils a domain through the
        # wrong expert, so refuse. Cache teachers all return zeros, so a miss
        # cannot change the loss — keep the tolerant fallback there.
        router: TeacherRouter | MetadataRouter = MetadataRouter(
            teacher_names=[t.name for t in registry.teachers],
            domain_to_index=domain_to_index,
            strict=bool(
                cfg.data.get("router_strict", teacher_mode == "live")
            ),
        )
    else:
        router = TeacherRouter()

    # Baseline trainability: "all" (default) or "lora" (freeze base+shared —
    # the plain-LoRA baseline). Must run before the optimizer is built.
    trainable = str(train_cfg.get("trainable", "all"))
    if trainable == "lora":
        groups = student.get_param_groups()
        for p in groups.get("P", []):
            p.requires_grad = False
        for p in groups.get("shared", []):
            p.requires_grad = False
    elif trainable != "all":
        raise ValueError(f"train.trainable must be 'all' or 'lora', got {trainable!r}")

    # DDP wrap. AFTER the requires_grad mutation above (DDP freezes the
    # trainable set at construction) and BEFORE the optimizer, whose groups
    # must be built from the raw module. DDP's construction-time parameter
    # broadcast also gives the replication invariant its base case for free.
    # ``student`` stays the raw module everywhere below; only the forward
    # goes through ``model_fwd``.
    model_fwd = dist.wrap(student)

    # Optimizer
    optim_cfg = OmegaConf.to_container(cfg.optim, resolve=True)
    optimizer = build_optimizer(student, optim_cfg)
    # Exact write masking (paper Theorem 1): controller-indexed params are
    # default-closed and only update when an R/F/P action opens them.
    # ``optim.masked: false`` is an escape hatch that leaves every param
    # open — MaskedAdamW then reduces exactly to plain AdamW (Lemma 1).
    # A disabled controller implies unmasked (baseline) behavior.
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
        ce_confidence_weighting=dist_cfg.get("ce_confidence_weighting", False),
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
    policy = RFPPolicy(
        ctrl_config.policy,
        ablation=ctrl_config.ablation,
        override=ctrl_config.debug.policy_override,
        debug=ctrl_config.debug,
        seed=int(seed),
    )
    controller_enabled = ctrl_config.enabled
    masked_updates = masked_updates and controller_enabled
    p_store_disabled = "P" in ctrl_config.ablation.disable_stores
    force_steps = set(ctrl_config.debug.force_consolidate_steps)
    replay_cfg = ctrl_config.retrieval
    replay_enabled = bool(replay_cfg.replay_on_hit) and controller_enabled

    if masked_updates:
        # Controller-indexed params (attn/ffn base + LoRA) default to closed;
        # shared params (embeddings, norms, lm_head) stay ordinary-AdamW.
        optimizer.register_controller_params(
            p for m in module_index for p in m.params
        )

    # Budget accounting (T7). ``indexed_coords`` is the denominator of the
    # active fraction; with masking off (controller disabled, or
    # optim.masked=false) every TRAINABLE indexed coordinate is open every
    # step and every trainable BASE coordinate is a permanent write.
    indexed_coords = sum(m.num_params for m in module_index)
    unmasked_open_coords = (
        0
        if masked_updates
        else sum(p.numel() for m in module_index for p in m.params if p.requires_grad)
    )
    unmasked_base_coords = (
        0
        if masked_updates
        else sum(
            p.numel()
            for m in module_index
            if m.id.param_type == "P"
            for p in m.params
            if p.requires_grad
        )
    )

    # Stores
    r_store = RetrievalStore(max_size=ctrl_config.repetition.buffer_size)
    f_store = FastStore(
        top_k_granularity=ctrl_config.writer.top_k_granularity,
        top_k_fraction=ctrl_config.writer.top_k_fraction,
    )
    p_store = PermanentStore()

    # Merges mutate weights outside the optimizer, so the scheduler needs the
    # process group: it broadcasts the reinitialized lora_a and the distill
    # replay batch (see ConsolidationScheduler._sync_after_merge/_pick_replay).
    # ``student`` here is the RAW module — the reducer guard uses the wrapper.
    consolidator = ConsolidationScheduler(
        ctrl_config.consolidation, student, p_store, optimizer=optimizer, dist=dist
    )
    consolidator._ddp_handle = model_fwd if model_fwd is not student else None
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
        keep_tagged=int(ckpt_cfg_raw.get("keep_tagged", 4)),
        resume_from=ckpt_cfg_raw.get("resume_from"),
        allow_partial_load=bool(ckpt_cfg_raw.get("allow_partial_load", False)),
        save_before_phases=[
            str(p) for p in (_as_container(ckpt_cfg_raw.get("save_before_phases")) or [])
        ],
    )
    # ``train.skip_step_ranges: [[lo, hi], ...]`` — steps consumed but not
    # trained on (E5's full-restore counterfactual). The stream digest does
    # not cover it on purpose: the batch sequence is unchanged.
    skip_ranges: list[tuple[int, int]] = [
        (int(lo), int(hi))
        for lo, hi in (_as_container(train_cfg.get("skip_step_ranges")) or [])
    ]
    # Only rank 0 writes: the tmp->rename->symlink sequence is not safe to
    # run concurrently against one directory. Every rank still loads.
    ckpt_manager = CheckpointManager(ckpt_cfg, is_main=is_main)

    # Data — the train loader uses a StatefulSampler so its (epoch, offset)
    # cursor checkpoints alongside everything else (bit-exact resume). When a
    # phased stream is configured, a deterministic StreamSchedule drives
    # batching instead (resumable by step index).
    from .data.stream import parse_stream_config

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    # Preflight cross-checks teacher ids and EM probe availability against
    # the real rows, so it needs those blocks alongside the data config.
    # Tests pass plain dicts, so only convert genuine OmegaConf nodes.
    data_cfg["_teachers_cfg"] = _as_container(cfg.get("teachers", {}))
    data_cfg["_eval_cfg"] = _as_container(cfg.get("eval", {}))
    data_cfg.setdefault("shuffle_seed", int(train_cfg.get("seed", 42)))
    stream_raw = cfg.get("stream")
    stream_cfg = parse_stream_config(
        OmegaConf.to_container(stream_raw, resolve=True)
        if stream_raw is not None
        else None
    )
    stream_schedule = None
    data_sampler = None
    if stream_cfg.enabled:
        dataloader, stream_schedule = build_dataloader(
            data_cfg,
            distributed,
            stream=stream_cfg,
            rank=dist.rank,
            world_size=dist.world_size,
        )
        cfg_steps = train_cfg.get("steps")
        if cfg_steps not in (None, stream_schedule.total_steps):
            raise ValueError(
                f"train.steps={cfg_steps} conflicts with the stream's total of "
                f"{stream_schedule.total_steps} steps; set train.steps: null "
                "or make them equal."
            )
    else:
        if dist.enabled:
            raise ValueError(
                "distributed.enabled requires a stream schedule (stream.enabled=true). "
                "The non-stream StatefulSampler has no rank-aware sharding and its "
                "(epoch, offset) cursor is what makes resume bit-exact."
            )
        dataloader, data_sampler = build_dataloader(data_cfg, distributed, stateful=True)
    if int(data_cfg.get("num_workers", 0)) > 0:
        print(
            "[hivemind] warning: num_workers > 0 breaks exact dataloader resume "
            "(worker prefetch advances the sampler ahead of consumed batches)."
        )

    # Per-domain validation loaders (HF path only; synthetic skips eval).
    eval_cfg_raw = cfg.get("eval", {}) or {}
    em_raw = eval_cfg_raw.get("exact_match", {}) or {}
    eval_config = EvaluationConfig(
        enabled=bool(eval_cfg_raw.get("enabled", False)),
        interval=int(eval_cfg_raw.get("interval", 500)),
        max_batches=int(eval_cfg_raw.get("max_batches", 32)),
        track_forgetting=bool(eval_cfg_raw.get("track_forgetting", True)),
        lora_sparsity_threshold=float(eval_cfg_raw.get("lora_sparsity_threshold", 1e-4)),
        at_phase_boundaries=bool(eval_cfg_raw.get("at_phase_boundaries", True)),
        on_consolidation=bool(eval_cfg_raw.get("on_consolidation", False)),
        interval_by_phase={
            str(k): int(v)
            for k, v in (_as_container(eval_cfg_raw.get("interval_by_phase")) or {}).items()
        },
        exact_match_enabled=bool(em_raw.get("enabled", False)),
        exact_match_domains=list(em_raw.get("domains", ["math"])),
        exact_match_samples=int(em_raw.get("num_samples", 64)),
        exact_match_max_new_tokens=int(em_raw.get("max_new_tokens", 64)),
        exact_match_max_scan_rows=int(em_raw.get("max_scan_rows", 20000)),
    )
    val_loaders: dict[str, Any] = {}
    if eval_config.enabled and data_source == "hf":
        base_data = OmegaConf.to_container(cfg.data, resolve=True)
        domain_specs = list(base_data.get("domains", []))
        for spec, name in zip(domain_specs, domain_names(domain_specs)):
            per_domain = dict(base_data)
            per_domain["split"] = eval_cfg_raw.get("split", "validation")
            per_domain["domains"] = [spec]
            per_domain["shuffle"] = False
            per_domain["drop_last"] = False
            per_domain["batch_size"] = int(
                eval_cfg_raw.get("batch_size", base_data.get("batch_size", 4))
            )
            per_domain["max_rows_per_domain"] = int(
                eval_cfg_raw.get(
                    "max_rows_per_domain", per_domain.get("max_rows_per_domain") or 256
                )
            )
            try:
                val_loaders[name] = build_dataloader(per_domain, distributed=False)
            except RuntimeError as exc:
                print(f"eval: skipping domain {name}: {exc}")
    # The eval block is entered on a predicate that includes `val_loaders`.
    # A transient per-rank load failure would therefore make ranks disagree,
    # and whoever entered would block forever at the barrier. Agree instead.
    if dist.enabled:
        agreed = dist.all_reduce_max_int(0 if val_loaders else 1)
        if agreed and val_loaders:
            print("[warn] a rank failed to build its eval loaders; disabling eval.")
            val_loaders = {}
    forgetting_tracker = ForgettingTracker() if eval_config.track_forgetting else None
    phase_tracker = PhaseEvalTracker() if eval_config.enabled else None

    # Gold-label exact-match probes: fixed rows from the validation split.
    em_probes: dict[str, Any] = {}
    em_tokenizer = None
    if eval_config.enabled and eval_config.exact_match_enabled and val_loaders:
        from .evaluation import build_gold_probes

        for name, loader in val_loaders.items():
            if name not in eval_config.exact_match_domains:
                continue
            # Probes need their OWN dataset. The val loaders are capped at
            # eval.max_rows_per_domain (a couple hundred rows) for speed,
            # and gold rows are sparse — math has 379 in 20 000, so scanning
            # the eval cap yields ~3 probes and quantizes EM to thirds.
            probe_dataset = loader.dataset
            scan = eval_config.exact_match_max_scan_rows
            if scan > int(eval_cfg_raw.get("max_rows_per_domain", 256) or 256):
                probe_cfg = dict(base_data)
                probe_cfg["split"] = eval_cfg_raw.get("split", "validation")
                probe_cfg["domains"] = [
                    s for s in base_data.get("domains", [])
                    if domain_names([s])[0] == name
                ]
                probe_cfg["max_rows_per_domain"] = scan
                probe_cfg["shuffle"] = False
                try:
                    probe_dataset = build_dataloader(
                        probe_cfg, distributed=False
                    ).dataset
                except RuntimeError as exc:
                    print(f"eval: exact_match probe scan for {name} failed: {exc}")
            probes = build_gold_probes(
                probe_dataset,
                [name],
                num_samples=eval_config.exact_match_samples,
                seed=int(train_cfg.get("seed", 42)),
            )
            em_probes.update(probes)
            if em_tokenizer is None:
                em_tokenizer = getattr(loader.collate_fn, "tokenizer", None)
        if em_probes and em_tokenizer is None:
            print("eval: exact_match enabled but no tokenizer found; disabling.")
            em_probes = {}

    # Logger
    log_cfg = OmegaConf.to_container(cfg.get("logging", {}), resolve=True) or {}
    # One writer per JSONL file: all ranks appending to the same fd
    # produces interleaved, torn records.
    if not is_main:
        log_cfg = dict(log_cfg, enabled=False)
    logger = init_logger(log_cfg)

    # Event trace ("when and why parameters changed") + per-module ledger.
    trace_enabled = (
        bool(log_cfg.get("enabled", False))
        and bool(log_cfg.get("events_enabled", True))
        and is_main
    )
    event_trace = EventTrace(
        log_cfg.get("events_path", "logs/events.jsonl"), enabled=trace_enabled
    )
    ledger = ModuleLedger()

    # Profiling (memory + wall-clock; the paper reports both).
    prof_raw = cfg.get("profiling", None)
    prof_cfg = (
        OmegaConf.to_container(prof_raw, resolve=True)
        if OmegaConf.is_config(prof_raw)
        else dict(prof_raw or {})
    )
    profiler = RunProfiler(
        device=device,
        enabled=bool(prof_cfg.get("enabled", True)),
        sync_cuda=bool(prof_cfg.get("sync_cuda", False)),
        ema_beta=float(prof_cfg.get("ema_beta", 0.98)),
    )
    # run_summary.json destination: explicit run.dir, else next to the
    # metrics file when logging is on, else skipped entirely.
    run_raw = cfg.get("run", None)
    run_dir = run_raw.get("dir") if run_raw is not None else None
    if run_dir is None and log_cfg.get("enabled"):
        run_dir = str(Path(str(log_cfg.get("path", "logs/metrics.jsonl"))).parent)

    # Mixed precision
    mp_cfg = train_cfg.get("mixed_precision", {})
    use_amp = mp_cfg.get("enabled", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if mp_cfg.get("dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
    )

    # --- Training loop ---
    if stream_schedule is not None:
        steps = stream_schedule.total_steps
    else:
        steps = train_cfg.get("steps", 100)
    log_interval = train_cfg.get("log_interval", 10)
    final_metrics: dict[str, float] = {}

    # Resume if configured. Must happen BEFORE the data iterator is created:
    # the load restores RNG state and the sampler cursor, and creating the
    # iterator first would consume them out of order.
    start_step = 0
    # Every rank must resolve to the SAME directory: the `latest` symlink can
    # be mid-update on rank 0 while another rank reads it.
    _resolved = str(ckpt_manager.resolve_resume_path() or "") if dist.is_main else ""
    _resolved = dist.broadcast_obj(_resolved, src=0)
    resume_path = Path(_resolved) if _resolved else None
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
            sampler=data_sampler,
            phase_eval=phase_tracker,
            map_location=str(device),
        )
        start_step = int(meta.get("step", 0)) + 1
        if stream_schedule is not None:
            saved_digest = (meta.get("extra") or {}).get("stream_digest")
            if saved_digest and saved_digest != stream_schedule.config_digest():
                raise RuntimeError(
                    "Stream config changed since the checkpoint was written "
                    f"(digest {saved_digest} != {stream_schedule.config_digest()}). "
                    "Resuming would silently change the data order."
                )
            dataloader.batch_sampler.start_step = start_step
            saved_skip = (meta.get("extra") or {}).get("skip_step_ranges")
            if saved_skip is not None and [list(r) for r in skip_ranges] != [
                list(r) for r in saved_skip
            ]:
                print(
                    f"[warn] train.skip_step_ranges changed since the checkpoint "
                    f"({saved_skip} -> {skip_ranges}); the stream order is unchanged "
                    "but the trained steps differ."
                )
        saved_ws = int((meta.get("extra") or {}).get("world_size", 1))
        if saved_ws != dist.world_size and not dist.cfg.allow_world_size_change:
            raise RuntimeError(
                f"checkpoint was written at world_size={saved_ws} but this run has "
                f"{dist.world_size}. The global batch composition is preserved, but "
                "the float reduction order changes, so the resume is not bit-exact. "
                "Set distributed.allow_world_size_change=true to proceed anyway."
            )
        event_trace.emit(
            {"type": "resume", "step": start_step, "from": str(resume_path)}
        )
        if is_main:
            print(f"Resumed from {resume_path} at step {start_step}.")

        # Selective rollback (T5): a blocked teacher's *tentative* evidence
        # must not survive the restore either — a later, clean P action
        # would merge the whole adapter, corrupted rows included. Every
        # F-module whose pending attribution is dominated by a blocked
        # teacher is reset (adapter re-initialised, its optimizer state
        # zeroed, its consolidation flag dropped, its attribution cleared).
        blocked_teachers = list(ctrl_config.debug.block_p_for_teachers)
        if blocked_teachers:
            for mid in consolidator.all_f_modules():
                name = str(mid)
                share = sum(
                    ledger.attribution_share(name, t) for t in blocked_teachers
                )
                if share < ctrl_config.debug.block_min_share:
                    continue
                attribution = ledger.clear_pending(name)
                adapters = consolidator._get_lora_adapters(mid)
                for ad in adapters:
                    ad.reset_lora()
                optimizer.reset_state_for_params(
                    [p for ad in adapters for p in ad.lora_params]
                )
                consolidator.unflag(mid)
                consolidator._sync_after_merge(mid)
                event_trace.emit(
                    {
                        "type": "rollback_reset",
                        "step": start_step,
                        "module": name,
                        "attribution": attribution,
                        "blocked_teachers": blocked_teachers,
                    }
                )

    # One self-describing config event per (re)start so the trace can be
    # analyzed without the Hydra config at hand.
    event_trace.emit(
        {
            "type": "run_config",
            "start_step": start_step,
            "top_m": (
                ctrl_config.ablation.top_m_override
                if ctrl_config.ablation.top_m_override is not None
                else ctrl_config.policy.top_m_modules
            ),
            "top_k_fraction": ctrl_config.writer.top_k_fraction,
            "ablation": {
                "disable_signals": list(ctrl_config.ablation.disable_signals),
                "disable_stores": list(ctrl_config.ablation.disable_stores),
                "use_teacher_confidence": ctrl_config.ablation.use_teacher_confidence,
            },
            "policy_mode": ctrl_config.policy.mode,
            "policy_override": ctrl_config.debug.policy_override,
            "adam_score_high": ctrl_config.policy.adam_score_high,
            "thresholds": {
                "surprise_high": ctrl_config.policy.surprise_high,
                "repetition_low": ctrl_config.policy.repetition_low,
                "repetition_medium": ctrl_config.policy.repetition_medium,
                "stability_high_C": ctrl_config.policy.stability_high_C,
                "stability_low_V": ctrl_config.policy.stability_low_V,
                "stability_source": ctrl_config.policy.stability_source,
            },
            "consolidation": {
                "period": ctrl_config.consolidation.period,
                "min_repetition": ctrl_config.consolidation.min_repetition,
                "min_stability_C": ctrl_config.consolidation.min_stability_C,
                "stability_mode": ctrl_config.consolidation.stability_mode,
                "strategy": ctrl_config.consolidation.merge_strategy,
                "trigger": ctrl_config.consolidation.trigger,
                "plateau_window": ctrl_config.consolidation.plateau_window,
                "plateau_tolerance": ctrl_config.consolidation.plateau_tolerance,
            },
            "retrieval": {
                "replay_on_hit": replay_enabled,
                "hit_threshold": replay_cfg.hit_threshold,
                "replay_batches": replay_cfg.replay_batches,
            },
            "corrupt_teacher": (
                {
                    **stream_schedule.corruption.digest_payload(),
                    "corrupted_rows": len(stream_schedule.corrupted_indices),
                    "corrupted_rows_by_phase": dict(
                        stream_schedule.corrupted_rows_by_phase
                    ),
                }
                if stream_schedule is not None and stream_schedule.corruption is not None
                else None
            ),
        }
    )

    if ctrl_config.debug.block_p_for_teachers:
        _blocked = list(ctrl_config.debug.block_p_for_teachers)
        _min_share = ctrl_config.debug.block_min_share

        def _is_blocked(mid: ModuleId) -> bool:
            return (
                sum(ledger.attribution_share(str(mid), t) for t in _blocked) >= _min_share
            )

        consolidator.blocked = _is_blocked

    if stream_schedule is not None and skip_ranges:
        dataloader.batch_sampler.skip_ranges = list(skip_ranges)

    data_iter = iter(dataloader)

    student.train()
    pbar = tqdm(
        range(start_step, steps),
        desc="Training",
        initial=start_step,
        total=steps,
        disable=not is_main,
    )

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
                sampler=data_sampler,
                phase_eval=phase_tracker,
                extra={
                    "interrupted": reason,
                    **(
                        {
                            "stream_digest": stream_schedule.config_digest(),
                            "world_size": dist.world_size,
                        }
                        if stream_schedule is not None
                        else {}
                    ),
                },
            )
            if is_main:
                print(
                    "[hivemind] checkpoint saved. "
                    "Resume with checkpoint.resume_from=latest"
                )
        except Exception as exc:  # noqa: BLE001 — best-effort on shutdown
            print(f"[hivemind] emergency save failed: {exc}")

    def _pre_merge_save(at_step: int) -> None:
        """Tagged rollback checkpoint taken right before a merge.

        Resuming from it continues at ``at_step + 1`` WITHOUT the merge —
        i.e. the counterfactual run in which the promotion never happened.
        """
        if not ckpt_cfg.enabled:
            return
        event_trace.flush()
        ckpt_manager.save(
            step=at_step,
            student=student,
            optimizer=optimizer,
            signal_computer=signal_computer,
            r_store=r_store,
            consolidator=consolidator,
            forgetting=forgetting_tracker,
            ledger=ledger,
            sampler=data_sampler,
            phase_eval=phase_tracker,
            tag="pre_merge",
            extra={
                "reason": "pre_merge",
                "pending_modules": sorted(str(m) for m in consolidator.pending_p),
                "attribution": {
                    str(m): ledger.pending_attribution(str(m))
                    for m in sorted(consolidator.pending_p, key=str)
                },
                **(
                    {
                        "stream_digest": stream_schedule.config_digest(),
                        "world_size": dist.world_size,
                    }
                    if stream_schedule is not None
                    else {}
                ),
            },
        )
        # Rank 0 serializes the whole model here; without this the other
        # ranks race straight into the merge and the next gradient
        # all-reduce. p_study_small sets checkpoint_before_merge on four runs.
        dist.barrier()

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

    current_phase: str | None = (
        stream_schedule.phase_at(start_step).name
        if stream_schedule is not None and start_step < steps
        else None
    )

    for step in pbar:
        profiler.step_start()

        # Stream phase bookkeeping (boundaries feed the trace + analysis).
        if stream_schedule is not None:
            phase_name = stream_schedule.phase_at(step).name
            if phase_name != current_phase or step == start_step:
                if phase_name != current_phase and current_phase is not None:
                    event_trace.emit(
                        {"type": "phase_end", "step": step - 1, "phase": current_phase}
                    )
                    mem = profiler.on_phase_end(current_phase)
                    if mem:
                        event_trace.emit(
                            {"type": "phase_memory", "step": step - 1, **mem}
                        )
                event_trace.emit(
                    {
                        "type": "phase_start",
                        "step": step,
                        "phase": phase_name,
                        "corrupted_rows": int(
                            stream_schedule.corrupted_rows_by_phase.get(phase_name, 0)
                        ),
                    }
                )
                profiler.on_phase_start(phase_name)
                current_phase = phase_name

        # Steps inside train.skip_step_ranges are consumed by the sampler
        # (not yielded) and not trained on: no forward, no eval, no
        # checkpoint — the phase simply never happened to this model.
        if stream_schedule is not None and dataloader.batch_sampler.is_skipped(step):
            event_trace.emit(
                {"type": "skipped_step", "step": step, "reason": "skip_range"}
            )
            profiler.step_end(tokens_in_step=0)
            last_step = step
            if dist.all_reduce_max_int(1 if _interrupted["flag"] else 0):
                _interrupted["flag"] = True
                break
            continue

        # Get batch
        raw = None
        collate_failed = False
        with profiler.timer.section("data"):
            try:
                raw = next(data_iter)
            except StopIteration:
                if stream_schedule is not None:
                    # The stream is finite by construction; running past it
                    # means a step-accounting bug, not an epoch boundary.
                    raise RuntimeError(
                        f"Stream exhausted at step {step} (< {steps})"
                    ) from None
                data_iter = iter(dataloader)
                raw = next(data_iter)
            except EmptyBatchError:
                # Only the collator's own "every row was unusable" signal —
                # a bare `except ValueError` here would silently reinterpret
                # a dataset/tokenizer/cache failure as an empty shard.
                # Single-process this is fatal; under DDP it would kill one
                # rank while the others block in the reducer, i.e. a hang,
                # which is strictly worse than a crash. Agree on it first.
                collate_failed = True

        if dist.all_reduce_max_int(1 if collate_failed else 0):
            if dist.cfg.on_empty_batch == "fail":
                raise RuntimeError(
                    f"step {step}: a rank produced an empty batch after collation "
                    "(every row was missing input_text / response text). Set "
                    "distributed.on_empty_batch=skip to continue instead."
                )
            optimizer.zero_grad()
            event_trace.emit({"type": "skipped_step", "step": step})
            # Close the step properly rather than `continue`-ing past the
            # tail: otherwise the profiler never sees it, `last_step` lags
            # (so a SIGINT save resumes a step early), and a skip on the
            # FINAL step would lose the final checkpoint entirely.
            profiler.step_end(tokens_in_step=0)
            last_step = step
            if step == steps - 1 and ckpt_cfg.enabled:
                event_trace.flush()
                ckpt_manager.save(
                    step=step, student=student, optimizer=optimizer,
                    signal_computer=signal_computer, r_store=r_store,
                    consolidator=consolidator, forgetting=forgetting_tracker,
                    ledger=ledger, sampler=data_sampler, phase_eval=phase_tracker,
                    extra={"skipped_final_step": True, "world_size": dist.world_size},
                )
                dist.barrier()
            if dist.all_reduce_max_int(1 if _interrupted["flag"] else 0):
                _interrupted["flag"] = True
                break
            continue

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
            if step == start_step and router.miss_count:
                # A silent 100% miss rate still trains (every cache teacher
                # returns zeros) — surface it on the very first step.
                print(
                    f"[warn] MetadataRouter missed {router.miss_count}/"
                    f"{router.routed_count} samples on the first step. "
                    "teachers.teacher_ids does not match the dataset's "
                    "teacher_id values; routing is falling back to index 0."
                )
        else:
            teacher_indices, _scores = router.route(
                h_t, registry.prototypes.to(device)
            )

        # === Step 3: Teacher forward (cache fast-path if batch carries logits) ===
        cached_logits = batch.get("teacher_logits")
        cached_mask = batch.get("teacher_logits_mask")
        with profiler.timer.section("teacher"):
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

        with profiler.timer.section("forward"), autocast_ctx:
            # The ONLY site that goes through the DDP wrapper — this is what
            # engages the gradient all-reduce during backward.
            student_logits = model_fwd(tokens)  # [B, T, V]

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
        with profiler.timer.section("backward"):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

        # === Step 7-8: Compute controller signals ===
        # Prefer the dataset-provided bucket when present (Theory 101 §4C);
        # fall back to the teacher index for the synthetic path.
        meta = batch.get("metadata")
        stream_bucket = (
            stream_schedule.bucket_override(step) if stream_schedule is not None else None
        )
        if stream_bucket is not None:
            # Novel-phase batches carry a unique synthetic bucket so the
            # repetition signal sees them as isolated one-offs.
            bucket_id = _bucket_to_int(stream_bucket, fallback=0)
        elif meta and meta.get("bucket_id"):
            bucket_id = _bucket_to_int(meta["bucket_id"][0], fallback=0)
        else:
            bucket_id = teacher_indices[0].item() if teacher_indices.numel() > 0 else 0
        embedding = h_t[0] if h_t.numel() > 0 else None
        batch_conf = batch.get("teacher_confidence")
        if isinstance(batch_conf, torch.Tensor) and batch_conf.numel() > 0:
            conf_sum = float(batch_conf.float().sum().item())
            conf_count = int(batch_conf.numel())
            # Passed through unchanged when world_size == 1, so the
            # single-GPU objective stays bit-identical to before DDP existed.
            conf_local_mean = float(batch_conf.float().mean().item())
        else:
            conf_sum, conf_count, conf_local_mean = 0.0, 0, None

        # Under DDP these three are the ONLY controller inputs still derived
        # from the rank-local batch; everything downstream is a pure function
        # of them plus the (already all-reduced) gradients. One fused
        # collective makes them global, and the replication invariant in
        # hivemind.distributed then holds by induction. Placed after
        # scaler.unscale_ so DDP's own reduction has completed.
        #
        # Fidelity note: bucket_id/embedding come from rank 0's first row,
        # which is the global batch's first row only when the collator kept
        # it. DistillCollator silently drops malformed rows, so a drop on
        # rank 0 shifts which row is representative. Harmless — recurrent
        # batches are bucket-pure, and random/mixed batches have no
        # privileged row — but it is a real difference from single-GPU.
        bucket_id, embedding, conf_mean = dist.sync_controller_inputs(
            bucket_id=bucket_id,
            embedding=embedding,
            conf_sum=conf_sum,
            conf_count=conf_count,
            conf_local_mean=conf_local_mean,
        )
        # Representative row for R-store replay (T2). Captured every step
        # when replay is on — under DDP it is a collective, and a collective
        # must not sit behind a branch on the (not yet computed) actions.
        replay_sample = (
            _capture_replay_sample(batch, dist, device) if replay_enabled else None
        )

        with profiler.timer.section("signals"):
            module_signals = (
                signal_computer.compute_all(
                    optimizer=optimizer,
                    modules=module_index,
                    bucket_id=bucket_id,
                    embedding=embedding,
                )
                if controller_enabled
                else {}
            )

        # Resolve the teacher info for the representative (first) sample —
        # mirrors the choice already made for ``embedding`` and ``bucket_id``
        # above so the R-store entry is consistent. Theory 101 §7 lists
        # teacher_id and teacher soft targets among the R payload.
        teacher_idx_repr = (
            int(teacher_indices[0].item()) if teacher_indices.numel() > 0 else 0
        )
        # Rank 0's first row is the global batch's first row; the index feeds
        # the (rank-0-checkpointed) ledger, so every rank must carry it.
        teacher_idx_repr = dist.broadcast_int(teacher_idx_repr)
        teacher_name_repr = ""
        if 0 <= teacher_idx_repr < len(registry.teachers):
            teacher_name_repr = str(registry.teachers[teacher_idx_repr].name)
        teacher_text_repr = ""
        if meta and meta.get("teacher_output_text"):
            teacher_text_repr = str(meta["teacher_output_text"][0] or "")

        # === Step 9: R/F/P routing policy ===
        policy_section = profiler.timer.section("policy_write")
        policy_section.__enter__()
        actions = (
            policy.decide(
                module_signals,
                teacher_confidence=conf_mean,
                step=step,
                phase=current_phase,
                teacher_name=teacher_name_repr if teacher_name_repr else None,
                teacher_index=teacher_idx_repr,
                num_teachers=len(registry.teachers),
            )
            if controller_enabled
            else []
        )

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
            sample=replay_sample,
        )

        # Trace every routing decision + update the per-module ledger.
        action_by_module = {str(a.module_id): a for a in actions}
        for det in write_metrics["actions"]:
            ledger.record_action(
                det["module"],
                det["store"],
                step,
                det["coords_opened"],
                teacher=teacher_name_repr,
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
                        "C_bar": round(act.stability_C_sustained, 6),
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

        policy_section.__exit__(None, None, None)

        # Tier 1: prove the replicas still agree on WHICH coordinates open.
        # Divergence here has no symptom of its own — DDP synchronizes
        # gradients, never parameters — so the replicas would simply drift
        # apart and the run would finish looking healthy.
        dist.assert_rank_consistent(
            (
                step,
                bucket_id,
                teacher_name_repr,
                None if conf_mean is None else round(conf_mean, 12),
                tuple(
                    (d["module"], d["store"], d["coords_opened"])
                    for d in write_metrics["actions"]
                ),
            ),
            what="routing decisions",
        )

        # === Step 11: Masked optimizer step ===
        # Coordinates not opened by an action stay bit-identical (params,
        # moments, weight decay, bias-correction clocks — Theorem 1).
        with profiler.timer.section("optimizer"):
            if masked_updates:
                optimizer.set_masks(write_masks)
            try:
                scaler.step(optimizer)
                scaler.update()
            finally:
                # A GradScaler inf/nan skip must not leak masks into the
                # next step.
                optimizer.clear_masks()

        # === Step 11b: R-store replay (deferral, not deletion) ===
        # When the current bucket has been parked in R on enough EARLIER
        # steps and the controller now routes it somewhere other than R (the
        # pattern has started recurring), the parked representative rows are
        # replayed into the fast store as extra F micro-steps INSIDE this
        # step index. Running them here rather than substituting the next
        # stream batch keeps ``step == stream index``, which the phase map,
        # eval schedule, force_consolidate_steps, checkpoints and DDP
        # sharding all rely on. The queue is drained within the step, so no
        # replay state survives a step boundary except the R-store itself
        # (already checkpointed). Trigger inputs are all rank-synced.
        replayed: list[tuple[int, tuple[tuple[str, int], ...]]] = []
        replay_coords = 0
        if replay_enabled and write_metrics["actions"]:
            with profiler.timer.section("replay"):
                eligible = [
                    s for s in r_store.distinct_steps_for_bucket(bucket_id) if s < step
                ]
                any_non_r = any(d["store"] != "R" for d in write_metrics["actions"])
                if len(eligible) >= replay_cfg.hit_threshold and any_non_r:
                    for entry in r_store.pop_for_bucket(
                        bucket_id, replay_cfg.replay_batches
                    ):
                        if entry.input_ids is None:
                            continue  # parked before replay was enabled
                        rep_tokens = entry.input_ids.to(device).unsqueeze(0)
                        rep_labels = (
                            entry.labels.to(device).unsqueeze(0)
                            if entry.labels is not None
                            else None
                        )
                        rep_tidx = torch.tensor([int(entry.teacher_id)], device=device)
                        rep_mask = (
                            None
                            if entry.kd_row_mask is None
                            else torch.tensor([float(entry.kd_row_mask)], device=device)
                        )
                        rep_conf = (
                            None
                            if entry.teacher_confidence is None
                            else torch.tensor(
                                [float(entry.teacher_confidence)], device=device
                            )
                        )
                        rep_teacher_logits = batch_teacher_forward(
                            rep_tokens, rep_tidx, registry.teachers, cfg.model.vocab_size
                        )
                        optimizer.zero_grad()
                        with autocast_ctx:
                            rep_logits = model_fwd(rep_tokens)
                            rep_reg = compute_total_regularization(
                                student, reg_config, consistency_forward=None
                            )
                            rep_loss, _rep_loss_metrics = compute_distillation_objective(
                                teacher_logits=rep_teacher_logits,
                                student_logits=rep_logits,
                                targets=rep_tokens,
                                config=distill_config,
                                reg_loss=rep_reg,
                                labels=rep_labels,
                                teacher_logits_mask=rep_mask,
                                teacher_confidence=rep_conf,
                            )
                        scaler.scale(rep_loss).backward()
                        scaler.unscale_(optimizer)
                        # Top-M F-type modules by raw gradient norm — a pure
                        # function of the grads; no stateful tracker sees
                        # the replay. Base (P-type) modules are excluded:
                        # replay_target is the fast store.
                        rep_signals: dict[ModuleId, ModuleSignals] = {}
                        for m in module_index:
                            if m.id.param_type != "F":
                                continue
                            grads = [
                                p.grad.detach().flatten()
                                for p in m.params
                                if p.grad is not None
                            ]
                            if not grads:
                                continue
                            rep_signals[m.id] = ModuleSignals(
                                module_id=m.id,
                                grad_norm=torch.cat(grads).norm().item(),
                            )
                        rep_actions = [
                            StoreAction(module_id=s.module_id, store="F", grad_norm=s.grad_norm)
                            for s in policy.select_top_m(rep_signals)
                        ]
                        rep_write_metrics, rep_masks = writer.execute(
                            actions=rep_actions, module_map=module_map, step=step
                        )
                        if masked_updates:
                            optimizer.set_masks(rep_masks)
                        try:
                            scaler.step(optimizer)
                            scaler.update()
                        finally:
                            optimizer.clear_masks()
                        replay_coords += int(rep_write_metrics["coords_opened"])
                        for det in rep_write_metrics["actions"]:
                            ledger.record_action(
                                det["module"],
                                det["store"],
                                step,
                                det["coords_opened"],
                                replay=True,
                                teacher=entry.teacher_name,
                            )
                            # A distinct event type: ``decision`` records stay
                            # one-per-Top-M-module-per-step and the action
                            # shares (the source of random_routing's budget)
                            # stay uncontaminated.
                            event_trace.emit(
                                {
                                    "type": "replay",
                                    "step": step,
                                    "origin_step": int(entry.step),
                                    "module": det["module"],
                                    "action": det["store"],
                                    "coords_opened": det["coords_opened"],
                                    "bucket_id": bucket_id,
                                    "teacher": entry.teacher_name,
                                }
                            )
                        replayed.append(
                            (
                                int(entry.step),
                                tuple(
                                    (d["module"], d["coords_opened"])
                                    for d in rep_write_metrics["actions"]
                                ),
                            )
                        )
            dist.assert_rank_consistent(
                (step, "replay", tuple(replayed)), what="replay decisions"
            )
        replay_count = len(replayed)

        # Gradient-routing ablation (T6): kill one teacher's rank slice.
        # After the optimizer step and before this step's checkpoint, so a
        # save at this step already contains it and a resume never repeats
        # it. No RNG, identical on every rank.
        ablate = ctrl_config.debug.ablate_teacher_slice
        if ablate is not None and step == ablate["step"]:
            names = [t.name for t in registry.teachers]
            if ablate["teacher"] not in names:
                raise ValueError(
                    f"ablate_teacher_slice.teacher={ablate['teacher']!r} is not a "
                    f"registered teacher ({names})"
                )
            t_idx = names.index(ablate["teacher"])
            zeroed = 0
            with torch.no_grad():
                for mid in consolidator.all_f_modules():
                    for ad in consolidator._get_lora_adapters(mid):
                        mask_a, mask_b = f_store.compute_slice_masks(
                            ad, t_idx, len(registry.teachers)
                        )
                        ad.lora_a.masked_fill_(mask_a, 0)
                        ad.lora_b.masked_fill_(mask_b, 0)
                        optimizer.reset_state_for_coords(ad.lora_a, mask_a)
                        optimizer.reset_state_for_coords(ad.lora_b, mask_b)
                        zeroed += int(mask_a.sum().item()) + int(mask_b.sum().item())
            event_trace.emit(
                {
                    "type": "ablation",
                    "step": step,
                    "teacher": ablate["teacher"],
                    "teacher_index": t_idx,
                    "coords_zeroed": zeroed,
                }
            )
        ledger.record_step(
            write_metrics["coords_opened"] + replay_coords + unmasked_open_coords,
            unmasked_base_coords,
        )

        # Tier 2: catch drift that has not yet flipped a discrete decision.
        every = dist.cfg.assert_rank_consistency
        if every and step % every == 0:
            dist.assert_state_consistent(student, optimizer, step=step)

        # === Step 12: Consolidation check ===
        # Record the batch for distill-based consolidation replay (no-op
        # for the direct strategy).
        consolidation_section = profiler.timer.section("consolidation")
        consolidation_section.__enter__()
        consolidator.record_batch(tokens)
        consolidated = []

        def _trace_merges(
            mids: list[ModuleId], forced: bool, trigger: str = "signals"
        ) -> None:
            for mid in mids:
                # A merge rewrites the block's base weights wholesale: its
                # permanent-write cost is the base numel of the paired P
                # module (Figure 1's budget axis).
                base_mod = module_map.get(ModuleId(mid.layer, mid.block_type, "P"))
                merged_coords = base_mod.num_params if base_mod is not None else 0
                attribution = ledger.record_consolidation(
                    str(mid), step, merged_coords=merged_coords
                )
                attr_total = sum(attribution.values())
                sig = module_signals.get(mid)
                event_trace.emit(
                    {
                        "type": "consolidation",
                        "step": step,
                        "module": str(mid),
                        "trigger": trigger,
                        "merged_coords": merged_coords,
                        "attribution": attribution,
                        "attribution_share": (
                            {t: c / attr_total for t, c in attribution.items()}
                            if attr_total
                            else {}
                        ),
                        "strategy": ctrl_config.consolidation.merge_strategy,
                        "pre_signals": (
                            {
                                "C": round(sig.stability_C, 6),
                                "C_bar": round(sig.stability_C_sustained, 6),
                                "R": round(sig.repetition, 6),
                            }
                            if sig is not None
                            else {}
                        ),
                        "forced": forced,
                        "success": True,
                    }
                )

        # ``disable_stores: [P]`` must also stop the periodic sweep — with an
        # empty pending set the scheduler would otherwise merge unconditionally.
        if (
            controller_enabled
            and not p_store_disabled
            and consolidator.should_check(step)
        ):
            if (
                ctrl_config.consolidation.checkpoint_before_merge
                and consolidator.pending_p
            ):
                _pre_merge_save(step)
            consolidated = consolidator.consolidate(module_signals)
            _trace_merges(consolidated, forced=False)

        # Forced (deliberately mistimed) consolidation — the incorrect-
        # consolidation experiment. Bypasses all threshold validation.
        if controller_enabled and step in force_steps:
            if ctrl_config.consolidation.checkpoint_before_merge:
                _pre_merge_save(step)
            forced_merged = consolidator.force_consolidate()
            _trace_merges(forced_merged, forced=True, trigger="forced")
            consolidated = consolidated + forced_merged

        # Loss-plateau trigger (Online-LoRA-style baseline). Deliberately
        # outside the ``controller_enabled`` gates: the `plateau_trigger`
        # spec runs with the controller off and every adapter open. Every
        # rank feeds its own shard loss to the detector, but only rank 0's
        # verdict counts (per-rank losses differ), agreed through one int
        # collective, and the window is reset everywhere on a fire.
        if ctrl_config.consolidation.trigger == "plateau" and not p_store_disabled:
            fired_local = consolidator.plateau_fired(
                float(loss_metrics.get("loss/total", 0.0)), step
            )
            fire = dist.all_reduce_max_int(1 if (fired_local and is_main) else 0)
            if fire:
                if consolidator.plateau is not None:
                    consolidator.plateau.reset(step)
                if ctrl_config.consolidation.checkpoint_before_merge:
                    _pre_merge_save(step)
                plateau_merged = consolidator.force_consolidate()
                _trace_merges(plateau_merged, forced=False, trigger="plateau")
                consolidated = consolidated + plateau_merged
        consolidation_section.__exit__(None, None, None)

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
                "write/replay_count": replay_count,
                "signals/surprise_mean": sum(surprises) / max(len(surprises), 1),
                "signals/repetition_mean": sum(repetitions) / max(len(repetitions), 1),
                "signals/stability_C_mean": sum(stabilities) / max(len(stabilities), 1),
                "signals/stability_adam_mean": (
                    sum(adam_stabilities) / max(len(adam_stabilities), 1)
                ),
                "retrieval/buffer_size": r_store.size,
                "consolidation/count": len(consolidated),
            }
            metrics.update(profiler.step_metrics())
            if current_phase is not None:
                metrics["phase"] = current_phase
            if isinstance(router, MetadataRouter):
                metrics["router/miss_count"] = router.miss_count
                metrics["router/miss_rate"] = router.miss_rate

            pbar.set_postfix(
                loss=f"{loss_metrics['loss/total']:.4f}",
                R=write_metrics["r_count"],
                F=write_metrics["f_count"],
                P=write_metrics["p_count"],
            )
            logger.log(metrics, step)
            final_metrics = metrics

        # Evaluation — interval, phase-boundary, and post-merge triggers.
        boundary_phase = None
        if stream_schedule is not None and eval_config.at_phase_boundaries:
            next_differs = (
                step + 1 < steps
                and stream_schedule.phase_at(step + 1).name != current_phase
            )
            if next_differs or step == steps - 1:
                boundary_phase = stream_schedule.phase_at(step)
        eval_interval = eval_config.interval_by_phase.get(
            current_phase or "", eval_config.interval
        )
        interval_due = (
            eval_interval > 0
            and step > 0
            and (step % eval_interval == 0 or step == steps - 1)
        )
        consolidation_due = bool(consolidated) and eval_config.on_consolidation
        if (
            eval_config.enabled
            and val_loaders
            and (interval_due or boundary_phase is not None or consolidation_due)
        ):
            # Rank 0 only. ForgettingTracker and PhaseEvalTracker are
            # CHECKPOINTED mutable state that run_evaluation mutates in
            # place, so running eval everywhere would make rank 0's
            # checkpoint stop describing every rank. Cost is negligible:
            # eval.max_batches batches against hundreds of training steps.
            # Barriers on both sides keep the others from racing ahead into
            # the next backward's all-reduce (and tripping the NCCL watchdog
            # during a long exact_match generation).
            dist.barrier()
            eval_metrics: dict = {}
            eval_failed = 0
            if is_main:
                try:
                    eval_metrics = run_evaluation(
                        student=student,
                        val_loaders=val_loaders,
                        device=device,
                        config=eval_config,
                        forgetting=forgetting_tracker,
                        r_store=r_store,
                        probe_embeddings=h_t.detach() if h_t.numel() > 0 else None,
                    )

                    # Gold exact-match probes only at phase boundaries (generation
                    # is expensive).
                    if em_probes and em_tokenizer is not None and boundary_phase is not None:
                        for name, probe in em_probes.items():
                            eval_metrics[f"eval/{name}/exact_match"] = exact_match_eval(
                                student,
                                em_tokenizer,
                                probe,
                                device,
                                max_new_tokens=eval_config.exact_match_max_new_tokens,
                            )

                    # Phase-resolved retention accounting (the paper's forgetting
                    # numbers segment by phase, not just first-seen baselines).
                    if boundary_phase is not None and phase_tracker is not None:
                        phase_domains = (
                            list(boundary_phase.domains)
                            if boundary_phase.domains
                            else ([boundary_phase.domain] if boundary_phase.domain else [])
                        )
                        domain_loss = {
                            d: eval_metrics[f"eval/{d}/loss"]
                            for d in val_loaders
                            if f"eval/{d}/loss" in eval_metrics
                        }
                        domain_em = {
                            d: eval_metrics[f"eval/{d}/exact_match"]
                            for d in val_loaders
                            if f"eval/{d}/exact_match" in eval_metrics
                        }
                        deltas = phase_tracker.record(
                            phase=boundary_phase.name,
                            step=step,
                            phase_domains=phase_domains,
                            domain_loss=domain_loss,
                            domain_em=domain_em or None,
                        )
                        for d, delta in deltas.items():
                            eval_metrics[f"eval/{d}/retention_delta"] = delta
                        event_trace.emit(
                            {
                                "type": "phase_eval",
                                "step": step,
                                "phase": boundary_phase.name,
                                "loss": domain_loss,
                                "em": domain_em,
                                "retention_delta": deltas,
                            }
                        )

                except Exception as exc:  # noqa: BLE001 — see below
                    # An eval failure on rank 0 (OOM during exact-match
                    # generation is the realistic one) would otherwise unwind
                    # here while every other rank sits in the broadcast below
                    # until the NCCL watchdog fires. Record it and let the
                    # all-reduce turn it into a simultaneous crash.
                    eval_failed = 1
                    print(f"[error] evaluation failed at step {step}: {exc!r}")
            if consolidation_due:
                eval_metrics["eval/trigger"] = "consolidation"
            # Turn a rank-0 eval failure into a crash on every rank rather
            # than a 30-minute NCCL-watchdog hang on ranks 1..N-1.
            if dist.all_reduce_max_int(eval_failed):
                raise RuntimeError(
                    f"evaluation failed on rank 0 at step {step}; aborting all ranks."
                )
            # Share the canonical numbers so every rank's view agrees.
            eval_metrics = dist.broadcast_obj(eval_metrics, src=0)
            logger.log(eval_metrics, step)
            # Surface key eval numbers on the progress bar
            macro = eval_metrics.get("eval/macro_loss")
            if macro is not None:
                pbar.set_postfix_str(f"val_loss={macro:.3f}", refresh=False)
            dist.barrier()

        # Tagged ``pre_phase_<name>`` save at the last step before a listed
        # phase (after the boundary eval, so the tracker state is complete).
        if (
            ckpt_cfg.enabled
            and ckpt_cfg.save_before_phases
            and stream_schedule is not None
            and step + 1 < steps
        ):
            next_phase = stream_schedule.phase_at(step + 1).name
            if next_phase != current_phase and next_phase in ckpt_cfg.save_before_phases:
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
                    sampler=data_sampler,
                    phase_eval=phase_tracker,
                    tag=f"pre_phase_{next_phase}",
                    extra={
                        "reason": "pre_phase",
                        "phase": next_phase,
                        "stream_digest": stream_schedule.config_digest(),
                        "world_size": dist.world_size,
                        "skip_step_ranges": [list(r) for r in skip_ranges],
                    },
                )
                dist.barrier()

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
                sampler=data_sampler,
                phase_eval=phase_tracker,
                extra={
                    "final_metrics": final_metrics,
                    **(
                        {
                            "stream_digest": stream_schedule.config_digest(),
                            "world_size": dist.world_size,
                            "skip_step_ranges": [list(r) for r in skip_ranges],
                        }
                        if stream_schedule is not None
                        else {}
                    ),
                },
            )
            # Ranks > 0 must not race past a save: rank 0 is still writing,
            # and at the final step they would otherwise exit mid-write.
            dist.barrier()

        # Mark this step as fully complete (signals + writes + optimizer
        # step + maybe checkpoint all finished) so the SIGINT handler's
        # emergency save resumes from the right place.
        profiler.step_end(tokens_in_step=int(tokens.numel()))
        last_step = step
        # SIGINT does not land on every rank at the same instant. If one
        # rank breaks and the others do not, the survivors block forever
        # in the next gradient all-reduce — so agree on it first.
        if dist.all_reduce_max_int(1 if _interrupted["flag"] else 0):
            _interrupted["flag"] = True
            break

    # Restore the prior signal handler before any more code runs.
    signal.signal(signal.SIGINT, _prev_handler)

    if _interrupted["flag"]:
        _emergency_save("SIGINT")

    if current_phase is not None:
        profiler.on_phase_end(current_phase)
    if run_dir is not None and is_main:
        totals = ledger.totals()
        permanent_writes = {
            "p_action_coords": int(totals.get("p_coords_opened", 0)),
            "merged_coords": int(totals.get("merged_coords", 0)),
            "unmasked_base_coords_total": int(totals.get("unmasked_base_coords", 0)),
        }
        permanent_writes["total"] = sum(permanent_writes.values())
        trained_steps = int(totals.get("steps", 0))
        write_run_summary(
            run_dir,
            profiler=profiler,
            final_metrics=final_metrics,
            retention=phase_tracker.retention_table() if phase_tracker else None,
            ledger_totals={**totals, "teachers": ledger.teacher_summary()},
            permanent_writes=permanent_writes,
            indexed_coords=indexed_coords,
            active_fraction_mean=(
                totals.get("coords_opened", 0) / (trained_steps * indexed_coords)
                if trained_steps and indexed_coords
                else 0.0
            ),
            interrupted=bool(_interrupted["flag"]),
            extra={
                **(
                    {
                        "stream_digest": stream_schedule.config_digest(),
                        "world_size": dist.world_size,
                    }
                    if stream_schedule is not None
                    else {}
                ),
                "retrieval": {
                    "replay_on_hit": replay_enabled,
                    "replayed_total": int(r_store.replayed_total),
                    "buffer_size": int(r_store.size),
                },
                **(
                    {
                        "corrupt_teacher": {
                            **stream_schedule.corruption.digest_payload(),
                            "corrupted_rows": len(stream_schedule.corrupted_indices),
                            "corrupted_rows_by_phase": dict(
                                stream_schedule.corrupted_rows_by_phase
                            ),
                        }
                    }
                    if stream_schedule is not None
                    and stream_schedule.corruption is not None
                    else {}
                ),
            },
        )

    event_trace.close()
    logger.close()
    if _interrupted["flag"]:
        print(
            f"\nTraining interrupted at step {last_step}. "
            "Resume with checkpoint.resume_from=latest"
        )
    else:
        print(f"\nTraining complete. Final loss: {final_metrics.get('loss/total', 'N/A')}")
    return final_metrics
