#!/usr/bin/env python3
"""
OOD evaluation: run PRM on math-reasoning steps (GSM8K + MATH-500) and
report degradation vs in-distribution.

Reads ood/ood_steps.jsonl (each row already has scenario, prior_steps,
current_step, gold_correct=True). Wraps each as PRM prompt input,
runs run_prm.py-equivalent inference, scores how the PRM rates known-
correct math reasoning.

Key result for the paper: % of OOD steps that PRM mis-flags as FLAWED
when they're actually gold-correct gold solutions.

Usage:
    python3 eval_ood.py --ood ood/ood_steps.jsonl --out ood/preds.jsonl \
        [--load-4bit]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PRM_SYSTEM = (
    "You are a financial reasoning verifier. Given a crypto/DeFi scenario "
    "and a reasoning step (with context of prior steps), score the step on "
    "four dimensions: factual_accuracy, logical_validity, completeness, "
    "and risk_awareness. Each score is 0.0 to 1.0. Respond with valid "
    "JSON only."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    # Convert OOD format → PRM input format
    in_path = Path(args.ood)
    tmp_path = in_path.with_suffix(".prm_input.jsonl")
    rows = [json.loads(l) for l in open(in_path)]
    if args.n > 0:
        rows = rows[: args.n]

    with open(tmp_path, "w") as f:
        for r in rows:
            user = (
                "Evaluate this reasoning step.\n\n"
                f"SCENARIO:\n{r['scenario']}\n\n"
                f"PRIOR STEPS:\n{r['prior_steps'] or '(none)'}\n\n"
                f"CURRENT STEP TO EVALUATE:\n{r['current_step']}"
            )
            f.write(json.dumps({
                "system": PRM_SYSTEM,
                "user": user,
                "_meta": {
                    "source": r["source"],
                    "gold_correct": r["gold_correct"],
                },
            }) + "\n")

    # Delegate to run_prm.py
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_prm.py"),
        "--input", str(tmp_path),
        "--output", args.out,
        "--batch", "4",
        "--max-new", "512",
        "--temperature", "0.1",
    ]
    if args.load_4bit:
        cmd.append("--load-4bit")

    import subprocess
    subprocess.check_call(cmd)

    # Quick summary
    print()
    print("=== OOD-eval summary ===")
    preds = [json.loads(l) for l in open(args.out)]
    by_src = {}
    for p in preds:
        src = p["_meta"]["source"]
        by_src.setdefault(src, []).append(p)

    for src, rows in by_src.items():
        flagged_flawed = sum(
            1 for r in rows if r.get("pred", {}).get("rating") == "FLAWED"
        )
        ok = [r for r in rows if r.get("pred", {}).get("ok")]
        scores = [r["pred"].get("overall_score", 0) for r in ok if r["pred"].get("overall_score") is not None]
        import statistics
        mean_s = statistics.mean(scores) if scores else 0
        print(f"  {src:<10} n={len(rows):>4}  parsed={len(ok)}  "
              f"mean_overall={mean_s:.3f}  flagged_FLAWED={flagged_flawed}  "
              f"({100*flagged_flawed/len(rows):.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
