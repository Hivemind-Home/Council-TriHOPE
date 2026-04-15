"""Teacher registry: manages K frozen teacher models and their prototypes.

Each teacher has a precomputed prototype vector c_k ∈ R^d, computed as
the mean routing embedding of seed examples associated with that teacher.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
