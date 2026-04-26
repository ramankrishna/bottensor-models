#!/usr/bin/env python3
"""
Score PRM predictions against gold labels.

Computes:
  1. Spearman correlation (overall_score)
  2. Per-dimension Spearman
  3. Error-detection F1 (binary: is_FLAWED)
  4. Step-level accuracy (rating match)
  5. Mean absolute error
  6. ECE / reliability bins (10 buckets)
  7. JSON parse-failure rate

Usage:
    python3 score_predictions.py preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict


def spearman(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    rx = sorted(range(n), key=lambda i: x[i])
    ry = sorted(range(n), key=lambda i: y[i])
    rank_x = [0] * n
    rank_y = [0] * n
    for i, idx in enumerate(rx):
        rank_x[idx] = i
    for i, idx in enumerate(ry):
        rank_y[idx] = i
    mx = sum(rank_x) / n
    my = sum(rank_y) / n
    num = sum((rank_x[i] - mx) * (rank_y[i] - my) for i in range(n))
    dx = math.sqrt(sum((rank_x[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((rank_y[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx * dy > 0 else 0.0


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r


def ece(predicted_probs, true_labels, n_bins=10):
    """Expected calibration error, binary version (true_labels ∈ {0,1})."""
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(predicted_probs, true_labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    total = sum(len(b) for b in bins)
    e = 0.0
    rows = []
    for i, b in enumerate(bins):
        if not b:
            rows.append((i / n_bins, (i + 1) / n_bins, 0, 0, 0))
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        avg_y = sum(y for _, y in b) / len(b)
        rows.append((i / n_bins, (i + 1) / n_bins, len(b), avg_p, avg_y))
        e += (len(b) / total) * abs(avg_p - avg_y)
    return e, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL with {assistant (gold), pred} per row")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    print(f"loaded {len(rows)} prediction rows")

    # Gold + pred extraction
    gold_overall, pred_overall = [], []
    gold_rating, pred_rating = [], []
    gold_dims = defaultdict(list)
    pred_dims = defaultdict(list)
    parse_fail = 0
    gold_parse_fail = 0

    for r in rows:
        try:
            gold = json.loads(r["assistant"])
        except Exception:
            gold_parse_fail += 1
            continue
        pred = r.get("pred", {})
        if not pred.get("ok"):
            parse_fail += 1
            continue
        gold_overall.append(gold.get("overall_score"))
        pred_overall.append(pred.get("overall_score"))
        gold_rating.append(gold.get("rating"))
        pred_rating.append(pred.get("rating"))
        for d in ["factual_accuracy", "logical_validity",
                  "completeness", "risk_awareness"]:
            g = gold.get("dimensions", {}).get(d)
            g_score = g.get("score") if isinstance(g, dict) else g
            p_score = pred.get("dimensions", {}).get(d)
            if g_score is not None and p_score is not None:
                gold_dims[d].append(g_score)
                pred_dims[d].append(p_score)

    n = len(gold_overall)
    print(f"\nparse failures (pred): {parse_fail}/{len(rows)}  ({100*parse_fail/len(rows):.2f}%)")
    print(f"parse failures (gold): {gold_parse_fail}/{len(rows)}")
    print(f"usable for scoring:    {n}")

    if n == 0:
        print("no usable predictions — abort"); return

    # Overall Spearman
    s_overall = spearman(gold_overall, pred_overall)
    print(f"\n=== Overall Spearman ===")
    print(f"  overall_score: {s_overall:.4f}")

    # Per-dim Spearman
    print(f"\n=== Per-dimension Spearman ===")
    for d in ["factual_accuracy", "logical_validity",
              "completeness", "risk_awareness"]:
        if gold_dims[d] and pred_dims[d]:
            s = spearman(gold_dims[d], pred_dims[d])
            print(f"  {d:<22} n={len(gold_dims[d])}  rho={s:.4f}")

    # MAE on overall
    mae = statistics.mean(abs(g - p) for g, p in zip(gold_overall, pred_overall))
    print(f"\n=== MAE on overall_score ===")
    print(f"  mean abs error: {mae:.4f}")

    # Rating accuracy
    matches = sum(1 for g, p in zip(gold_rating, pred_rating) if g == p)
    print(f"\n=== Rating-label accuracy ===")
    print(f"  exact match: {matches}/{n} = {100*matches/n:.2f}%")
    confusion = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold_rating, pred_rating):
        confusion[g][p] += 1
    print("  confusion matrix (rows=gold, cols=pred):")
    cats = ["STRONG", "ACCEPTABLE_WITH_ISSUES", "FLAWED", None]
    print(f'    {"":<24}' + ''.join(f'{str(c)[:9]:>10}' for c in cats))
    for g in cats:
        if not confusion.get(g):
            continue
        print(f'    {str(g)[:22]:<24}' + ''.join(f'{confusion[g][p]:>10}' for p in cats))

    # Error detection F1 (binary: is_FLAWED)
    tp = fp = fn = tn = 0
    for g, p in zip(gold_rating, pred_rating):
        gf = (g == "FLAWED")
        pf = (p == "FLAWED")
        if gf and pf: tp += 1
        elif not gf and pf: fp += 1
        elif gf and not pf: fn += 1
        else: tn += 1
    f, prec, rec = f1(tp, fp, fn)
    print(f"\n=== Error-detection F1 (FLAWED vs not-FLAWED) ===")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision={prec:.4f}  recall={rec:.4f}  F1={f:.4f}")

    # ECE on FLAWED probability (using overall_score as probability of NOT-flawed)
    # We treat (1 - overall_score) as P(flawed) and check calibration vs binary gold
    p_flawed = [1 - p for p in pred_overall]
    y_flawed = [1 if r == "FLAWED" else 0 for r in gold_rating]
    e, bin_rows = ece(p_flawed, y_flawed)
    print(f"\n=== Calibration (ECE on FLAWED probability) ===")
    print(f"  ECE: {e:.4f}")
    print(f"  bin              n     avg_p_flawed   actual_flawed_rate")
    for lo, hi, n_b, p_avg, y_avg in bin_rows:
        gap = abs(p_avg - y_avg)
        bar = "█" * int(20 * gap)
        print(f"  [{lo:.1f}, {hi:.1f})  {n_b:>6}   {p_avg:>13.3f}   {y_avg:>17.3f}   {bar}")


if __name__ == "__main__":
    sys.exit(main())
