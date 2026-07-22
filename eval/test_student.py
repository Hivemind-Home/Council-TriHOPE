"""Quick inference / sanity check for the student model.

Run a list of prompts through the student. Use it twice: once before
training (no ``--checkpoint``) for a baseline, and once after training
(pointing at a saved step dir) to see how distillation moved outputs.

Examples
--------
Baseline (vanilla Qwen3-4B + freshly initialised LoRA = identity):

    python eval/test_student.py --config-name pilot_hf_gpu

After training, load latest checkpoint:

    python eval/test_student.py --config-name pilot_hf_gpu \\
        --checkpoint checkpoints/pilot_hf_gpu/latest

Specific step + your own prompt file:

    python eval/test_student.py --config-name pilot_hf_gpu \\
        --checkpoint checkpoints/pilot_hf_gpu/step_00002000 \\
        --prompts eval/prompts.txt

Hydra-style overrides go at the end (positional), e.g.:

    python eval/test_student.py --config-name pilot_hf_gpu \\
        model.dtype=bfloat16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hivemind.training import _build_student  # noqa: E402

DEFAULT_PROMPTS = [
    "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Handle empty input, unsorted input, and touching intervals like [1, 3] and [3, 5]. Explain the time complexity.",

    "Three people, Alice, Bob, and Carol, each either always tell the truth or always lie. Alice says: 'Bob is a liar.' Bob says: 'Carol is a liar.' Carol says: 'Alice and Bob are different types.' Determine who is truthful and who is lying, and explain why.",

    "Debug this Python function. What is wrong, why does it happen, and how would you fix it?\n\n```python\ndef add_item(item, items=[]):\n    items.append(item)\n    return items\n```"
]


def load_config(name: str, overrides: list[str]) -> DictConfig:
    cfg_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name=name, overrides=overrides)


def resolve_checkpoint_dir(spec: str) -> Path:
    """Accept ``checkpoints/<run>/latest``, a step dir, or a step number."""
    p = Path(spec)
    if p.exists() and p.is_symlink():
        return p.resolve()
    if p.exists() and p.is_dir():
        # if it's the run root, pick the highest step_*
        steps = sorted(c for c in p.iterdir() if c.is_dir() and c.name.startswith("step_"))
        if steps and not (p / "base.pt").exists():
            return steps[-1]
        return p
    raise FileNotFoundError(f"checkpoint path not found: {spec}")


def load_checkpoint_weights(
    student: torch.nn.Module, ckpt_dir: Path, device: torch.device
) -> None:
    base = torch.load(ckpt_dir / "base.pt", map_location=str(device), weights_only=False)
    lora = torch.load(ckpt_dir / "lora.pt", map_location=str(device), weights_only=False)
    missing, unexpected = student.load_state_dict({**base, **lora}, strict=False)
    print(f"loaded checkpoint: {ckpt_dir}")
    if missing:
        print(f"  missing keys (first 8): {list(missing)[:8]}")
    if unexpected:
        print(f"  unexpected keys (first 8): {list(unexpected)[:8]}")


def format_prompt(tokenizer, prompt: str) -> str:
    """Apply chat template if the tokenizer ships one, else raw text."""
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", required=True, help="hydra config name (no .yaml)")
    ap.add_argument("--checkpoint", default=None, help="checkpoint step dir, run root, or 'latest' symlink")
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line")
    ap.add_argument("--max-new-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda | cpu",
    )
    ap.add_argument("overrides", nargs="*", help="hydra overrides, e.g. model.dtype=bfloat16")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    cfg = load_config(args.config_name, list(args.overrides))

    print(f"building student: {cfg.model.pretrained_name} dtype={cfg.model.dtype} on {device}")
    student = _build_student(cfg, device)
    student.eval()
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  params: {n_params:,}")

    if args.checkpoint:
        ckpt_dir = resolve_checkpoint_dir(args.checkpoint)
        load_checkpoint_weights(student, ckpt_dir, device)
        tag = f"TRAINED ({ckpt_dir.name})"
    else:
        tag = "BASELINE (no checkpoint)"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.prompts:
        prompts = [
            ln.strip()
            for ln in Path(args.prompts).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        prompts = DEFAULT_PROMPTS

    print(f"\n=== {tag} — {len(prompts)} prompts ===\n")
    for i, prompt in enumerate(prompts, start=1):
        text = format_prompt(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = student.hf_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs.input_ids.shape[-1] :]
        gen = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"[{i}] {prompt}")
        print(f"  -> {gen}\n")


if __name__ == "__main__":
    main()
