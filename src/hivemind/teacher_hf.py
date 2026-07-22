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
    # Optional load-time quantization. ``None`` keeps the checkpoint dtype
    # (or whatever quantization config is already baked into the
    # checkpoint, e.g. native FP8 weights). Accepted: ``"int8"`` /
    # ``"8bit"``, ``"nf4"`` / ``"4bit"``, ``"fp4"``. Requires
    # ``bitsandbytes`` to be installed.
    quantization: Optional[str] = None


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
        device: Optional[torch.device] = None,
        quantization: Optional[str] = None,
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

        # FP8 (and other quantized) checkpoints must be placed on the
        # accelerator at load time — a post-hoc ``.to(device)`` leaves
        # quantizer state on CPU and triggers the HF FP8 warning. Pass
        # ``device_map`` so accelerate streams shards directly to the
        # target device. Falls back to a plain CPU load when no device.
        from_pretrained_kwargs: dict = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )

        if quantization is not None:
            from transformers import BitsAndBytesConfig

            q = quantization.lower()
            if q in ("int8", "8bit", "load_in_8bit"):
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif q in ("nf4", "4bit"):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            elif q == "fp4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            else:
                raise ValueError(
                    f"Unknown teacher quantization {quantization!r}; "
                    "expected one of: int8, nf4, fp4."
                )
            from_pretrained_kwargs["quantization_config"] = bnb_config

        if device is not None and device.type in ("cuda", "xpu"):
            from_pretrained_kwargs["device_map"] = {"": device}

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            pretrained_name,
            **from_pretrained_kwargs,
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
            device=device,
            quantization=spec.quantization,
        )
        # When device_map placed the model already, .to() is a no-op;
        # keep it for the CPU/fallback path so non-accelerated loads
        # still land on the requested device. bnb quantized models
        # cannot be moved with .to(); skip the move in that case.
        if (
            device is not None
            and device.type not in ("cuda", "xpu")
            and spec.quantization is None
        ):
            model = model.to(device)
        # Attach domain attribute so MetadataRouter's domain fallback works.
        model.domain = spec.domain  # type: ignore[attr-defined]
        out.append(TeacherInfo(name=name, model=model))
    return out


def parse_hf_teacher_specs(raw, /) -> list[HFTeacherSpec]:
    """Normalise config into a list of ``HFTeacherSpec``.

    Accepts either:
    * ``{"code": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "math": "..."}``
    * ``[{"domain": "code", "pretrained_name": "...", "teacher_id": "..."}, ...]``

    OmegaConf ``DictConfig`` / ``ListConfig`` are accepted too and
    converted to plain Python containers up front so the type checks
    below behave as expected (``isinstance(DictConfig, dict)`` is False
    in current OmegaConf releases).
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(raw, (DictConfig, ListConfig)):
            raw = OmegaConf.to_container(raw, resolve=True)
    except ImportError:  # pragma: no cover — omegaconf always present in deps
        pass

    out: list[HFTeacherSpec] = []
    if isinstance(raw, dict):
        for domain, value in raw.items():
            # Two dict-form shapes: ``{domain: "Org/Model"}`` (plain
            # name) or ``{domain: {name: "Org/Model", quantization:
            # "int8"}}`` (per-teacher options).
            if isinstance(value, dict):
                pretrained = str(value.get("name") or value.get("pretrained_name"))
                if not pretrained or pretrained == "None":
                    raise ValueError(
                        f"teachers.pretrained[{domain}] is missing 'name'."
                    )
                out.append(
                    HFTeacherSpec(
                        domain=str(domain),
                        pretrained_name=pretrained,
                        teacher_id=value.get("teacher_id"),
                        quantization=value.get("quantization"),
                    )
                )
            else:
                out.append(
                    HFTeacherSpec(domain=str(domain), pretrained_name=str(value))
                )
    elif isinstance(raw, list):
        for entry in raw:
            out.append(
                HFTeacherSpec(
                    domain=str(entry["domain"]),
                    pretrained_name=str(entry["pretrained_name"]),
                    teacher_id=entry.get("teacher_id"),
                    quantization=entry.get("quantization"),
                )
            )
    else:
        raise TypeError(
            f"teachers.pretrained must be a dict {{domain: name}} or a list of "
            f"specs; got {type(raw).__name__}"
        )
    return out
