"""Data subsystem.

Exports kept stable so prior imports (``from hivemind.data import
SyntheticTeacherSeedData, build_dataloader``) continue to work after the
split from a single ``data.py`` into a package.
"""

from .dataloader import build_dataloader
from .synthetic import (
    SyntheticDataConfig,
    SyntheticTeacherSeedData,
    SyntheticTextDataset,
)

__all__ = [
    "build_dataloader",
    "SyntheticDataConfig",
    "SyntheticTextDataset",
    "SyntheticTeacherSeedData",
]
