"""Hydra single-GPU training entry point."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from hivemind.config_utils import unwrap_config
from hivemind.device import resolve_device
from hivemind.training import run_training_loop


@hydra.main(config_path="configs", config_name="pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    device = resolve_device(cfg.train.device)
    run_training_loop(cfg, device=device, distributed=False)


if __name__ == "__main__":
    main()
