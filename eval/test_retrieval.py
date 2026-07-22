"""Retrieval-aware inference for the trained student.

Loads a checkpoint AND its R-store, then for each prompt:

  1. Embeds the prompt with the same SharedEmbedding used in training.
  2. Queries the R-store for the top-k nearest neighbors.
  3. Generates an answer in one of three modes:

     * ``rag``    — prepend retrieved (question, teacher_answer) pairs as
                    in-context examples, then generate with the student.
     * ``direct`` — if top similarity ≥ threshold, return the stored
                    teacher answer verbatim (R as a long-tail cache).
                    Falls back to plain generation otherwise.
     * ``both``   — print plain generation, then RAG generation, then
                    direct lookup. Side-by-side comparison.

Compare against ``eval/test_student.py`` (non-retrieval) on the same
prompts to see what retrieval is buying you.

Examples
--------
    python eval/test_retrieval.py --config-name pilot_hf_gpu \\
        --checkpoint checkpoints/pilot_hf_gpu/latest

    python eval/test_retrieval.py --config-name pilot_hf_gpu \\
        --checkpoint checkpoints/pilot_hf_gpu/latest \\
        --mode both --top-k 3 --sim-threshold 0.85

A checkpoint with a non-empty R-store is required — without retrieval
entries this script has nothing to retrieve.
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

from hivemind.embedding import SharedEmbedding  # noqa: E402
from hivemind.stores.retrieval import RetrievalStore  # noqa: E402
from hivemind.training import _build_student  # noqa: E402

DEFAULT_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Solve for x: 3x + 5 = 20. Show your steps.",
    "Explain quicksort in one short paragraph.",
    "Compute the derivative of x^2 * sin(x).",
    "Debug: for i in range(len(xs)): print(xs[i+1]) — what is wrong?",
]


def load_config(name: str, overrides: list[str]) -> DictConfig:
    cfg_dir = str(REPO_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name=name, overrides=overrides)


def resolve_checkpoint_dir(spec: str) -> Path:
    p = Path(spec)
    if p.exists() and p.is_symlink():
        return p.resolve()
    if p.exists() and p.is_dir():
        steps = sorted(c for c in p.iterdir() if c.is_dir() and c.name.startswith("step_"))
        if steps and not (p / "base.pt").exists():
            return steps[-1]
        return p
    raise FileNotFoundError(f"checkpoint path not found: {spec}")


def load_checkpoint_weights(student, ckpt_dir: Path, device: torch.device) -> None:
    base = torch.load(ckpt_dir / "base.pt", map_location=str(device), weights_only=False)
    lora = torch.load(ckpt_dir / "lora.pt", map_location=str(device), weights_only=False)
    missing, unexpected = student.load_state_dict({**base, **lora}, strict=False)
    print(f"loaded weights: {ckpt_dir}")
    if missing:
        print(f"  missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"  unexpected keys (first 5): {list(unexpected)[:5]}")


def load_rstore(ckpt_dir: Path) -> RetrievalStore:
    stores_path = ckpt_dir / "stores.pt"
    if not stores_path.exists():
        raise FileNotFoundError(
            f"checkpoint has no stores.pt at {stores_path}. "
            "This script needs a training checkpoint with retrieval entries."
        )
    state = torch.load(stores_path, map_location="cpu", weights_only=False)
    r_store = RetrievalStore(max_size=state.get("max_size", 1000))
    r_store.load_state_dict(state)
    print(f"loaded R-store: {r_store.size} entries")
    return r_store


def format_chat(tokenizer, prompt: str) -> str:
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def build_rag_prompt(tokenizer, prompt: str, neighbors: list, max_chars_per_example: int = 600) -> str:
    """Prepend retrieved (Q, teacher_A) pairs as in-context examples.

    Each ``RetrievalEntry`` only stores a routing-space embedding (no
    original question text), so we mark the retrieved context with the
    teacher_name + bucket_id and use teacher_output_text as the answer.
    The student sees: "Examples from <teacher>: <answer>. Now the user's
    question: ...".
    """
    if not neighbors:
        return format_chat(tokenizer, prompt)

    blocks = ["Reference answers from prior similar questions:"]
    for i, (sim, entry) in enumerate(neighbors, start=1):
        text = (entry.teacher_output_text or "").strip()
        if not text:
            continue
        if len(text) > max_chars_per_example:
            text = text[:max_chars_per_example].rstrip() + "..."
        tag = entry.teacher_name or f"teacher_{entry.teacher_id}"
        blocks.append(f"[{i}] (similarity={sim:.2f}, source={tag})\n{text}")
    blocks.append(f"\nUser question:\n{prompt}")
    composed = "\n\n".join(blocks)
    return format_chat(tokenizer, composed)


def generate(student, tokenizer, text: str, device, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = student.hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir (or 'latest' symlink / run root)")
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line")
    ap.add_argument("--mode", choices=["rag", "direct", "both"], default="both")
    ap.add_argument("--top-k", type=int, default=3, help="neighbors to retrieve / show")
    ap.add_argument("--sim-threshold", type=float, default=0.85,
                    help="min cosine similarity for direct-lookup mode to fire")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    cfg = load_config(args.config_name, list(args.overrides))

    print(f"building student: {cfg.model.pretrained_name} on {device}")
    student = _build_student(cfg, device)
    student.eval()

    ckpt_dir = resolve_checkpoint_dir(args.checkpoint)
    load_checkpoint_weights(student, ckpt_dir, device)
    r_store = load_rstore(ckpt_dir)

    if r_store.size == 0:
        print("\nWARNING: R-store is empty — no R-routing happened during training.")
        print("Retrieval modes will fall back to plain generation. Check your")
        print("controller policy thresholds (surprise_high / repetition_low) if")
        print("you expected R writes.\n")

    routing = SharedEmbedding(student.embed).to(device).eval()

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

    print(f"\n=== retrieval eval — {len(prompts)} prompts, mode={args.mode} ===\n")

    for i, prompt in enumerate(prompts, start=1):
        print(f"[{i}] {prompt}")

        # Embed via SharedEmbedding (same as training-time routing)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        emb = routing(ids)[0]  # [d]

        # Query top-k
        neighbors = r_store.query_nearest(emb.cpu(), k=args.top_k) if r_store.size else []

        if neighbors:
            print("  retrieved:")
            for j, (sim, entry) in enumerate(neighbors, start=1):
                snippet = (entry.teacher_output_text or "").strip().replace("\n", " ")
                if len(snippet) > 90:
                    snippet = snippet[:90] + "..."
                src = entry.teacher_name or f"teacher_{entry.teacher_id}"
                print(f"    {j}. sim={sim:.3f}  src={src}  step={entry.step}  bucket={entry.bucket_id}")
                print(f"       {snippet}")
        else:
            print("  retrieved: <none>")

        if args.mode in ("rag", "both"):
            text = build_rag_prompt(tokenizer, prompt, neighbors)
            answer = generate(
                student, tokenizer, text, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"  [rag]    -> {answer}")

        if args.mode in ("direct", "both"):
            if neighbors and neighbors[0][0] >= args.sim_threshold:
                top_sim, top_entry = neighbors[0]
                cached = (top_entry.teacher_output_text or "").strip()
                if cached:
                    print(f"  [direct] -> (cached, sim={top_sim:.3f}) {cached}")
                else:
                    print(f"  [direct] -> top neighbor has empty teacher_output_text; falling back")
                    answer = generate(
                        student, tokenizer, format_chat(tokenizer, prompt), device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    print(f"  [direct fallback] -> {answer}")
            else:
                top_sim_str = f"{neighbors[0][0]:.3f}" if neighbors else "n/a"
                print(f"  [direct] -> below threshold ({top_sim_str} < {args.sim_threshold}); generating")
                answer = generate(
                    student, tokenizer, format_chat(tokenizer, prompt), device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(f"  [direct fallback] -> {answer}")

        print()


if __name__ == "__main__":
    main()
