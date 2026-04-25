"""
Needle-in-a-haystack: long-context retrieval sanity check.

At each (context_length, depth) pair we:
  1. Build a haystack of "filler" sentences sized to ``ctx_len`` tokens.
  2. Insert a unique "needle" sentence at the specified depth percentile.
  3. Prompt the model to retrieve the needle.
  4. Mark pass/fail by exact-substring containment of the needle's
     secret code in the model output.

Saves a JSON result file and a heatmap PNG.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

from config import NEEDLE_HEATMAP_PATH
from ._utils import generate, load_model, log_wandb, merge_results

LOG = logging.getLogger("npc-fast.eval.context")

CONTEXT_LENGTHS = [16_384, 32_768, 65_536, 131_072]
DEPTHS = [0.10, 0.25, 0.50, 0.75, 0.90]

FILLER = (
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. "
    "It extends from the Arctic Ocean in the north to the Southern Ocean in the south. "
    "Its area is larger than all of Earth's landmasses combined. "
)


def _make_haystack(tokenizer, target_tokens: int, needle: str, depth: float) -> str:
    # Cheap token-to-char ratio estimate, re-tuned empirically when we first
    # measure an over/under-shoot.
    chars_per_tok = 4
    target_chars = target_tokens * chars_per_tok
    filler_copies = max(1, target_chars // len(FILLER))
    text = FILLER * filler_copies
    insert_at = int(len(text) * depth)
    # Snap to a sentence boundary for realism
    while insert_at < len(text) and text[insert_at] != " ":
        insert_at += 1
    haystack = text[:insert_at] + "\n\n" + needle + "\n\n" + text[insert_at:]

    # Trim to exactly target_tokens via tokenizer
    ids = tokenizer(haystack, add_special_tokens=False)["input_ids"]
    if len(ids) > target_tokens:
        # Re-tokenize with trimming, preserving the needle
        prefix_ids = tokenizer(text[:insert_at], add_special_tokens=False)["input_ids"]
        needle_ids = tokenizer("\n\n" + needle + "\n\n", add_special_tokens=False)["input_ids"]
        suffix_budget = target_tokens - len(prefix_ids) - len(needle_ids)
        if suffix_budget <= 0:
            # Needle doesn't fit at this depth for this context — shift left
            prefix_ids = prefix_ids[: max(0, target_tokens - len(needle_ids))]
            ids = prefix_ids + needle_ids
        else:
            suffix_ids = tokenizer(text[insert_at:], add_special_tokens=False)["input_ids"][:suffix_budget]
            ids = prefix_ids + needle_ids + suffix_ids
    return tokenizer.decode(ids, skip_special_tokens=True)


def _heatmap(results: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        LOG.warning("matplotlib unavailable (%s); skipping heatmap.", e)
        return

    grid = np.zeros((len(DEPTHS), len(CONTEXT_LENGTHS)))
    for r in results:
        i = DEPTHS.index(r["depth"])
        j = CONTEXT_LENGTHS.index(r["context_length"])
        grid[i, j] = 1.0 if r["pass"] else 0.0

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(grid, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(CONTEXT_LENGTHS)))
    ax.set_xticklabels([f"{c//1024}K" for c in CONTEXT_LENGTHS])
    ax.set_yticks(range(len(DEPTHS)))
    ax.set_yticklabels([f"{int(d*100)}%" for d in DEPTHS])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Needle depth")
    ax.set_title("Needle-in-haystack — pass (green) / fail (red)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    LOG.info("Saved heatmap → %s", out_path)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--max_context", type=int, default=131_072,
                    help="skip lengths greater than this (for smoke tests)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model, tok = load_model(args.model_path)

    rng = random.Random(0)
    results: list[dict] = []
    for ctx_len in CONTEXT_LENGTHS:
        if ctx_len > args.max_context:
            continue
        for depth in DEPTHS:
            secret = f"NPC-NEEDLE-{rng.randrange(10**8):08d}"
            needle = f"IMPORTANT SECRET CODE: {secret}. Remember this exact code."
            haystack = _make_haystack(tok, ctx_len, needle, depth)
            prompt = (
                "You will be given a long document. Your job is to recall the "
                "secret code embedded inside it.\n\n"
                "=== DOCUMENT BEGIN ===\n"
                f"{haystack}\n"
                "=== DOCUMENT END ===\n\n"
                "What is the exact secret code starting with 'NPC-NEEDLE-'? "
                "Output only the code."
            )
            out = generate(
                model, tok,
                [{"role": "user", "content": prompt}],
                max_new_tokens=32, temperature=0.0,
            )
            passed = secret in out
            LOG.info("ctx=%d depth=%.2f -> pass=%s (model said: %r)",
                     ctx_len, depth, passed, out[:80])
            results.append({
                "context_length": ctx_len,
                "depth": depth,
                "secret": secret,
                "pass": bool(passed),
                "model_answer": out,
            })

    total = max(1, len(results))
    summary = {
        "needle_in_haystack": {
            "n": total,
            "pass_rate": sum(1 for r in results if r["pass"]) / total,
            "per_cell": results,
        }
    }
    merge_results(summary)
    _heatmap(results, Path(NEEDLE_HEATMAP_PATH))
    log_wandb({
        "eval/needle.pass_rate": summary["needle_in_haystack"]["pass_rate"],
    })


if __name__ == "__main__":
    main()
