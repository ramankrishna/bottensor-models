"""
Validation perplexity tracker — overall, per-tag, short vs long context.

Loads the same registry used for training, regenerates the 5% val split
deterministically, tokenizes each example, and computes token-weighted
cross-entropy grouped by tag and by short/long bucket (boundary 32K).
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from typing import Optional

import torch

from config import DATASETS_REGISTRY, TRAINING_CONFIG
from data.loader import load_registry
from data.mixer import mix
from data.preprocessing import tokenize_examples
from ._utils import load_model, log_wandb, merge_results

LOG = logging.getLogger("npc-fast.eval.ppl")


@torch.no_grad()
def _ce_tokens(model, input_ids: torch.Tensor) -> tuple[float, int]:
    """Return (total loss × tokens, token count) for a single sequence."""
    if input_ids.numel() < 2:
        return 0.0, 0
    input_ids = input_ids.to(model.device).unsqueeze(0)
    out = model(input_ids=input_ids, labels=input_ids)
    n_tokens = int(input_ids.numel()) - 1  # causal shift
    # HF returns mean loss per token; re-scale to total.
    return float(out.loss.item()) * n_tokens, n_tokens


def _bucket(seq_len: int) -> str:
    return "short" if seq_len < 4_096 else ("mid" if seq_len < 32_768 else "long")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--max_val", type=int, default=500,
                    help="cap number of val examples for eval time")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model, tok = load_model(args.model_path)

    LOG.info("Loading dataset registry for val split...")
    raw = load_registry(DATASETS_REGISTRY)
    _, val_ex = mix(
        raw,
        seed=TRAINING_CONFIG["seed"],
        val_split=TRAINING_CONFIG["val_split"],
    )
    val_ex = val_ex[: args.max_val]
    LOG.info("Tokenizing %d val examples...", len(val_ex))
    val_tok = tokenize_examples(val_ex, tok)

    totals = defaultdict(lambda: [0.0, 0])   # loss_sum, tokens
    for tex in val_tok:
        ids = torch.tensor(tex.input_ids, dtype=torch.long)
        ls, n = _ce_tokens(model, ids)
        if n == 0:
            continue
        bucket = _bucket(len(tex.input_ids))
        totals["overall"][0] += ls
        totals["overall"][1] += n
        totals[f"len:{bucket}"][0] += ls
        totals[f"len:{bucket}"][1] += n
        for tag in tex.tags:
            totals[f"tag:{tag}"][0] += ls
            totals[f"tag:{tag}"][1] += n

    def _ppl(loss_sum: float, n: int) -> float | None:
        if n <= 0:
            return None
        return math.exp(loss_sum / n)

    summary = {
        "perplexity": {
            key: {"tokens": n, "ppl": _ppl(ls, n)}
            for key, (ls, n) in sorted(totals.items())
        }
    }
    merge_results(summary)
    LOG.info("PPL overall: %.4f", summary["perplexity"]["overall"]["ppl"])
    log_wandb({
        "eval/ppl.overall": summary["perplexity"]["overall"]["ppl"],
        "eval/ppl.short": summary["perplexity"].get("len:short", {}).get("ppl"),
        "eval/ppl.long": summary["perplexity"].get("len:long", {}).get("ppl"),
    })


if __name__ == "__main__":
    main()
