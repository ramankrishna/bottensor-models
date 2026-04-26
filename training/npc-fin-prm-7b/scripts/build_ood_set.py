#!/usr/bin/env python3
"""
Build the out-of-distribution test set for NPC Fin-PRM evaluation.

Pulls reasoning steps from two public benchmarks:
  - GSM8K test split (50 problems → ~134 steps)
  - HuggingFaceH4/MATH-500 (50 problems → ~173 steps)

Each problem's gold solution is decomposed into individual reasoning
steps by sentence-boundary split. Every emitted record has
``gold_correct=True`` because the solutions are gold-correct by
construction; the OOD eval measures how many of these gold-correct
math steps the PRM mis-flags as FLAWED.

This is the reproducible source for ``analysis/ood_steps.jsonl``
shipped alongside.

Usage:
    python3 build_ood_set.py --out analysis/ood_steps.jsonl [--n 50]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset


def decompose_to_steps(solution_text: str) -> list[str]:
    """
    Naive sentence-boundary step decomposition.

    GSM8K:    "She has 3 apples. She buys 2. So 5. #### 5"  → 3 steps
    MATH-500: long LaTeX-ish derivations                    → multi-sentence

    The trailing ``#### N`` answer footer (GSM8K convention) is stripped
    so it doesn't show up as a "step". Sentences shorter than 5 chars
    are dropped.
    """
    body = re.sub(r"####\s*[-\d.,]+\s*$", "", solution_text).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)]
    return [s for s in sents if len(s) > 5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50,
                    help="Problems per benchmark (default 50)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # ── GSM8K test ────────────────────────────────────────────────
    print(f"Loading GSM8K test...")
    gsm = load_dataset("gsm8k", "main", split="test")
    gsm = gsm.shuffle(seed=args.seed).select(range(min(args.n, len(gsm))))
    for ex in gsm:
        steps = decompose_to_steps(ex["answer"])
        for i, step in enumerate(steps):
            prior = "\n".join(f"Step {k+1}: {steps[k]}" for k in range(i))
            records.append({
                "source": "GSM8K",
                "scenario": ex["question"],
                "prior_steps": prior,
                "current_step": step,
                "gold_correct": True,
                "overall_score_truth": 1.0,
            })

    # ── MATH-500 ──────────────────────────────────────────────────
    try:
        print(f"Loading MATH-500...")
        m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        m500 = m500.shuffle(seed=args.seed).select(range(min(args.n, len(m500))))
        for ex in m500:
            steps = decompose_to_steps(ex.get("solution", ""))
            for i, step in enumerate(steps):
                prior = "\n".join(f"Step {k+1}: {steps[k]}" for k in range(i))
                records.append({
                    "source": "MATH-500",
                    "scenario": ex.get("problem", ""),
                    "prior_steps": prior,
                    "current_step": step,
                    "gold_correct": True,
                    "overall_score_truth": 1.0,
                })
    except Exception as e:
        print(f"  MATH-500 unavailable ({e}) — GSM8K-only OOD set")

    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    src = Counter(r["source"] for r in records)
    print(f"\nWrote {out}")
    print(f"  total steps: {len(records)}")
    for s, c in src.items():
        print(f"  {s:<10} {c}")


if __name__ == "__main__":
    sys.exit(main())
