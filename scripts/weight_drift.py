#!/usr/bin/env python
"""Split a run's total weight change into routed base / adapters / shared (paper T3).

Compares the final checkpoint's student weights with the pretrained
Qwen3-0.6B weights they started from. Routed = the attention and
feed-forward projection weights the controller indexes (LoRALinear.base);
shared = everything else in base.pt (tied embedding / LM head, norms), which
trains with plain AdamW in every trainable=all arm; adapters = lora.pt.

For ``frozen_blocks`` the routed part must be bit-identical to the
pretrained weights: an at-scale check of Theorem 1 (closed coordinates are
untouched). The script exits 1 if ``--expect-routed-unchanged`` is given
and any routed coordinate moved.

    python scripts/weight_drift.py runs/priority_s3_v1_live/frozen_blocks-seed1337 --expect-routed-unchanged
    python scripts/weight_drift.py runs/baselines_small_v1_live/trihope-seed1337 runs/...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROUTED_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def final_checkpoint(run: Path) -> Path | None:
    root = run / "checkpoints"
    if not root.exists():
        return None
    cands = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("step_") and "pre_merge" not in p.name
    )
    return cands[-1] if cands else None


def hf_key(student_key: str) -> str:
    k = student_key.removeprefix("hf_model.")
    return k.replace(".base.weight", ".weight").replace(".base.bias", ".bias")


def is_routed(student_key: str) -> bool:
    return ".base." in student_key and any(
        f".{s}.base." in student_key for s in ROUTED_SUFFIXES
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--expect-routed-unchanged", action="store_true")
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    ref_sd = {k: v.detach().cpu() for k, v in ref.state_dict().items()}
    del ref

    failed = False
    for run in args.runs:
        ckpt = final_checkpoint(run)
        if ckpt is None:
            print(f"{run}: no final checkpoint, skipped")
            continue
        base = torch.load(ckpt / "base.pt", map_location="cpu", weights_only=False)
        lora = torch.load(ckpt / "lora.pt", map_location="cpu", weights_only=False)
        acc = {
            g: {"sq_norm_delta": 0.0, "changed_coords": 0, "coords": 0, "max_abs": 0.0}
            for g in ("routed", "shared")
        }
        unmatched = []
        for k, v in base.items():
            if not torch.is_floating_point(v):
                continue
            hk = hf_key(k)
            if hk not in ref_sd:
                unmatched.append(k)
                continue
            d = v.float() - ref_sd[hk].float()
            g = acc["routed" if is_routed(k) else "shared"]
            g["sq_norm_delta"] += float((d * d).sum())
            g["changed_coords"] += int((d != 0).sum())
            g["coords"] += d.numel()
            g["max_abs"] = max(g["max_abs"], float(d.abs().max()) if d.numel() else 0.0)
        adapters = {"modules": 0, "nonzero_B": 0, "sum_frob_BA": 0.0}
        a_keys = [k for k in lora if k.endswith("lora_A")]
        for ka in a_keys:
            kb = ka[: -len("lora_A")] + "lora_B"
            if kb not in lora:
                continue
            A, B = lora[ka].float(), lora[kb].float()
            adapters["modules"] += 1
            if bool((B != 0).any()):
                adapters["nonzero_B"] += 1
                adapters["sum_frob_BA"] += float(torch.linalg.matrix_norm(B @ A))
        out = {
            "run": str(run),
            "checkpoint": ckpt.name,
            "routed": {**acc["routed"], "l2_delta": acc["routed"]["sq_norm_delta"] ** 0.5},
            "shared": {**acc["shared"], "l2_delta": acc["shared"]["sq_norm_delta"] ** 0.5},
            "adapters": adapters,
            "unmatched_keys": unmatched[:20],
        }
        (run / "weight_drift.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out))
        if args.expect_routed_unchanged and acc["routed"]["changed_coords"] != 0:
            print(f"{run}: FAIL — {acc['routed']['changed_coords']} routed coords changed")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
