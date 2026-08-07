"""Hydra training entry point (single-GPU, or one DDP rank).

Single GPU:
    python train.py --config-name stream_small

Multi-GPU (see src/hivemind/distributed.py for why DDP and only DDP):
    python -m torch.distributed.run --standalone --nproc_per_node=4 \
        train.py --config-name stream_small ++distributed.enabled=true
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from hivemind.config_utils import unwrap_config
from hivemind.device import resolve_device
from hivemind.distributed import init_distributed, reject_unsupported_features, shutdown
from hivemind.training import run_training_loop


@hydra.main(config_path="configs", config_name="pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    # Without this the library's logger.info/warning calls (checkpoint saves,
    # loader fallbacks) go nowhere, which is how a silent resume-onto-nothing
    # stayed invisible.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    dist_ctx = init_distributed(cfg)
    reject_unsupported_features(cfg, dist_ctx)
    device = resolve_device(cfg.train.device, local_rank=dist_ctx.local_rank)
    try:
        run_training_loop(cfg, device=device, dist=dist_ctx)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
