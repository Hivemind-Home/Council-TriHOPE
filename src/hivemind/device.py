"""Device resolution utilities."""

from __future__ import annotations

import torch


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device, falling back to CPU if unavailable."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev
