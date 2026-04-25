"""Teacher registry: manages K frozen teacher models and their prototypes.

Each teacher has a precomputed prototype vector c_k ∈ R^d, computed as
the mean routing embedding of seed examples associated with that teacher.

Two teacher kinds are supported:

* ``StudentModel``-shaped teachers (synthetic / in-memory) — run live via
  ``batch_teacher_forward``; useful for the CPU smoke test and ablations.
* ``CacheBackedTeacher`` — a metadata-only stub used when supervision was
  pre-generated (Layer B + teacher_logits NPZ). The logits travel with
  the batch via ``DistillCollator``, so this class doesn't need to do
  real inference; it exists to give the router/prototypes a stable
  target and to keep the ``TeacherInfo.model`` slot type-uniform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .student.config import StudentConfig
from .student.model import StudentModel


@dataclass
class TeacherInfo:
    """Information about a single teacher."""

    name: str
    model: nn.Module  # frozen teacher
    prototype: torch.Tensor | None = None  # c_k ∈ R^d, set after compute_prototypes


class TeacherRegistry:
    """Manages K teachers and their prototype vectors."""

    def __init__(self, teachers: list[TeacherInfo]) -> None:
        self.teachers = teachers
        self._prototypes: torch.Tensor | None = None

        # Freeze all teachers
        for t in self.teachers:
            t.model.eval()
            for p in t.model.parameters():
                p.requires_grad = False

    @property
    def num_teachers(self) -> int:
        return len(self.teachers)

    @property
    def prototypes(self) -> torch.Tensor:
        """[K, d] stacked and L2-normalized prototype vectors."""
        if self._prototypes is None:
            raise RuntimeError("Prototypes not computed yet. Call compute_prototypes() first.")
        return self._prototypes

    def compute_prototypes(
        self,
        embedding_fn: nn.Module,
        seed_data: dict[str, list[torch.Tensor]],
    ) -> None:
        """Compute teacher prototypes from seed examples.

        c_k = mean(f_emb(x_i^(k))) for each teacher k.

        Args:
            embedding_fn: SharedEmbedding module.
            seed_data: {teacher_name: [tensor of token_ids [T], ...]}
        """
        device = next(embedding_fn.parameters()).device
        proto_list = []

        for teacher in self.teachers:
            examples = seed_data.get(teacher.name, [])
            if len(examples) == 0:
                # Random prototype if no seed data
                proto = torch.randn(embedding_fn.dim, device=device)
                proto = F.normalize(proto, p=2, dim=0)
            else:
                # Stack and embed seed examples
                batch = torch.stack(examples).to(device)  # [N_k, T]
                embeddings = embedding_fn(batch)  # [N_k, d]
                proto = embeddings.mean(dim=0)  # [d]
                proto = F.normalize(proto, p=2, dim=0)
            teacher.prototype = proto
            proto_list.append(proto)

        self._prototypes = torch.stack(proto_list)  # [K, d]

    def get_teacher(self, index: int) -> TeacherInfo:
        return self.teachers[index]


def create_synthetic_teachers(
    num_teachers: int,
    config: StudentConfig,
    device: torch.device,
) -> list[TeacherInfo]:
    """Create synthetic teacher models for testing/development.

    Each teacher is a small randomly-initialized StudentModel (frozen).
    """
    teachers = []
    for i in range(num_teachers):
        teacher_cfg = StudentConfig(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=max(1, config.num_layers // 2),  # smaller than student
            heads=config.heads,
            ffn_hidden_multiplier=config.ffn_hidden_multiplier,
            max_seq_len=config.max_seq_len,
            lora=config.lora,
        )
        model = StudentModel(teacher_cfg).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(TeacherInfo(name=f"teacher_{i}", model=model))
    return teachers


class CacheBackedTeacher(nn.Module):
    """Metadata-only teacher.

    Wraps a ``teacher_id`` / ``domain`` pair. In HF-cache training the
    actual teacher logits are already in the batch (put there by the
    collator via ``TeacherLogitsCache``), so this module's ``forward``
    just returns an empty tensor and the training loop uses the batch
    tensor directly. Making it an ``nn.Module`` keeps the ``TeacherInfo``
    contract simple and lets it live on a device.
    """

    def __init__(self, teacher_id: str, domain: str, vocab_size: int) -> None:
        super().__init__()
        self.teacher_id = teacher_id
        self.domain = domain
        self.vocab_size = vocab_size
        # A dummy buffer so ``.to(device)`` has a handle to move.
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        B, T = tokens.shape
        return torch.zeros(B, T, self.vocab_size, device=self._dummy.device)

    def extra_repr(self) -> str:
        return f"teacher_id={self.teacher_id!r}, domain={self.domain!r}"


def create_hf_cache_teachers(
    domains: list[str],
    teacher_ids: dict[str, str] | None,
    vocab_size: int,
    device: torch.device,
) -> list[TeacherInfo]:
    """One ``CacheBackedTeacher`` per domain.

    ``teacher_ids`` maps domain → teacher_id. When absent we fall back to
    ``"{domain}_teacher"`` which matches nothing in the Layer B data but
    is fine because metadata routing looks up by domain when the
    teacher_id is unknown.
    """
    teacher_ids = teacher_ids or {}
    teachers: list[TeacherInfo] = []
    for domain in domains:
        tid = teacher_ids.get(domain, f"{domain}_teacher")
        model = CacheBackedTeacher(teacher_id=tid, domain=domain, vocab_size=vocab_size).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(TeacherInfo(name=tid, model=model))
    return teachers


def compute_prototypes_from_texts(
    registry: "TeacherRegistry",
    embed_tokens: list[list[int]],
    embedding_fn: nn.Module,
    domain_assignments: list[str],
    device: torch.device,
) -> None:
    """Prototype helper for HF mode.

    ``embed_tokens[i]`` is a token-id list for sample ``i``; it is assigned
    to a teacher via ``domain_assignments[i]`` matching ``teacher.name``
    (or stripped to domain). Called by the training entry point when the
    router is ``cosine``; for metadata routing this is unnecessary but
    we still populate prototypes for logging.
    """
    assert len(embed_tokens) == len(domain_assignments)
    by_teacher: dict[str, list[torch.Tensor]] = {t.name: [] for t in registry.teachers}
    for ids, label in zip(embed_tokens, domain_assignments):
        if label not in by_teacher:
            # match by domain suffix if user passed bare domain
            for t in registry.teachers:
                if getattr(t.model, "domain", None) == label:
                    label = t.name
                    break
        if label in by_teacher:
            by_teacher[label].append(torch.tensor(ids, dtype=torch.long, device=device))

    registry.compute_prototypes(embedding_fn, by_teacher)
