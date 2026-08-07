"""Device resolution utilities."""

from __future__ import annotations

import torch


def resolve_device(device_str: str, local_rank: int | None = None) -> torch.device:
    """Resolve a device string to a torch.device, falling back to CPU if unavailable.

    ``local_rank`` pins this process to its own GPU. That is mandatory
    under DDP, not cosmetic: ``torch.amp.autocast("cuda")`` and
    ``GradScaler("cuda")`` bind to the *current* CUDA device, so without
    ``set_device`` every rank would autocast on ``cuda:0``. It also turns
    an index-less ``cuda`` into an explicit ``cuda:{local_rank}``.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            idx = 0 if local_rank is None else int(local_rank)
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
        return torch.device("cpu")
    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else (local_rank or 0)
        torch.cuda.set_device(int(idx))
        return torch.device(f"cuda:{int(idx)}")
    return dev
