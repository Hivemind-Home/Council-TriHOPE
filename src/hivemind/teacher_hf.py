"""Live HF-backed teachers for on-the-fly distillation.

Alternative to ``CacheBackedTeacher`` / precomputed NPZ logits. Use when:

* Supervision hasn't been materialised yet and you'd rather stream it
  from the teacher than generate + upload a Layer B.
* You're swapping the teacher model (e.g. Qwen3-Coder → DeepSeek R1) and
  want to compare without regenerating the dataset.

Vocab alignment is a hard requirement for logit-level KD: the teacher's
``config.vocab_size`` MUST match the student's. We check this at
construction and raise a clear error otherwise — silently mismatched
vocabs produce garbage KD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .teacher_registry import TeacherInfo


@dataclass
class HFTeacherSpec:
    domain: str
    pretrained_name: str
    teacher_id: Optional[str] = None  # defaults to "{domain}_teacher_hf"


class HFTeacher(nn.Module):
    """Frozen ``AutoModelForCausalLM`` exposing ``forward(tokens) -> logits``."""

    def __init__(
        self,
        pretrained_name: str,
        expected_vocab_size: int,
        *,
        dtype: str = "float32",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        torch_dtype = getattr(torch, dtype, torch.float32)
        hfc = AutoConfig.from_pretrained(
            pretrained_name, trust_remote_code=trust_remote_code, cache_dir=cache_dir
        )
        if int(hfc.vocab_size) != int(expected_vocab_size):
            raise ValueError(
                f"HFTeacher vocab mismatch: teacher={pretrained_name} has vocab_size="
                f"{hfc.vocab_size}, student expects {expected_vocab_size}. "
                "Logit-level KD requires aligned vocabs. Either pick a teacher with "
                "the matching tokenizer family, or disable KD (lambda_kd=0) and run "
                "text-level CE on teacher_output_text."
            )

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            pretrained_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.vocab_size = int(hfc.vocab_size)
        self.pretrained_name = pretrained_name

        # Freeze + eval
        self.hf_model.eval()
        for p in self.hf_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(input_ids=tokens, use_cache=False, return_dict=True)
        return out.logits


def create_hf_live_teachers(
    specs: list[HFTeacherSpec],
    *,
    vocab_size: int,
    device: torch.device,
    dtype: str = "float32",
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
) -> list[TeacherInfo]:
    """One ``HFTeacher`` per spec, registered under ``teacher_id`` name."""
    out: list[TeacherInfo] = []
    for spec in specs:
        name = spec.teacher_id or f"{spec.domain}_teacher_hf"
        model = HFTeacher(
            pretrained_name=spec.pretrained_name,
            expected_vocab_size=vocab_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        ).to(device)
        # Attach domain attribute so MetadataRouter's domain fallback works.
        model.domain = spec.domain  # type: ignore[attr-defined]
        out.append(TeacherInfo(name=name, model=model))
    return out


def parse_hf_teacher_specs(raw: dict[str, str] | list[dict], /) -> list[HFTeacherSpec]:
    """Normalise config into a list of ``HFTeacherSpec``.

    Accepts either:
    * ``{"code": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "math": "..."}``
    * ``[{"domain": "code", "pretrained_name": "...", "teacher_id": "..."}, ...]``
    """
    out: list[HFTeacherSpec] = []
    if isinstance(raw, dict):
        for domain, pretrained in raw.items():
            out.append(HFTeacherSpec(domain=str(domain), pretrained_name=str(pretrained)))
    else:
        for entry in raw:
            out.append(
                HFTeacherSpec(
                    domain=str(entry["domain"]),
                    pretrained_name=str(entry["pretrained_name"]),
                    teacher_id=entry.get("teacher_id"),
                )
            )
    return out
