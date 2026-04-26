#!/usr/bin/env python3
"""
Run NPC Fin-PRM on the val set using MLX 4-bit on Apple Silicon.

Usage:
    python3 run_mlx_eval.py --input val.jsonl --output preds.jsonl --n 200

Loads /tmp/finprm-mlx/mlx-q4 (the merged + 4-bit-quantized PRM) via
mlx_lm and runs sequential inference. Single-batch (mlx_lm doesn't
batch yet) but fast on M-series — typically 50-90 tok/s.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from mlx_lm import generate, load


def parse_prm_output(text: str) -> dict:
    """
    Tolerant PRM output parser. Tries (in order):
      1. Strict JSON parse of the {...} block
      2. Regex salvage if the JSON was truncated mid-justification
         (common on small max_new_tokens budgets) — extracts overall_score,
         rating, and per-dimension scores from the partial text.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    first = text.find("{")
    if first < 0:
        return {"ok": False, "mode": "no-brace", "raw": text}

    # ── attempt 1: strict balanced-brace parse ───────────────────────
    depth, end = 0, -1
    in_string = False
    escape = False
    for i, c in enumerate(text[first:], start=first):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end >= 0:
        try:
            obj = json.loads(text[first:end])
            dims_in = obj.get("dimensions", {})
            flat = {}
            for k, v in dims_in.items():
                s = v.get("score") if isinstance(v, dict) else v
                if isinstance(s, (int, float)):
                    flat[k] = s
            return {
                "ok": True,
                "mode": "strict",
                "overall_score": obj.get("overall_score"),
                "rating": obj.get("rating"),
                "dimensions": flat,
                "error_identified": obj.get("error_identified"),
                "raw": text,
            }
        except Exception:
            pass

    # ── attempt 2: regex salvage from truncated JSON ─────────────────
    # The PRM emits a stable shape: overall_score, rating, dimensions.
    # If max_new_tokens cut a justification mid-string, the JSON has no
    # closing brace — but all the *score* values appear early, before the
    # justification strings.
    overall_m = re.search(r'"overall_score"\s*:\s*([-\d.]+)', text)
    rating_m  = re.search(r'"rating"\s*:\s*"([A-Z_]+)"', text)
    err_m     = re.search(r'"error_identified"\s*:\s*(?:"([A-Z_]+)"|null)', text)
    flat = {}
    for d in ("factual_accuracy", "logical_validity", "completeness", "risk_awareness"):
        m = re.search(rf'"{d}"\s*:\s*\{{[^}}]*?"score"\s*:\s*([-\d.]+)', text)
        if m:
            try:
                flat[d] = float(m.group(1))
            except Exception:
                pass

    overall = float(overall_m.group(1)) if overall_m else None
    rating = rating_m.group(1) if rating_m else None
    error = err_m.group(1) if (err_m and err_m.group(1)) else None

    # Salvage is "ok" if we got at least overall_score AND all 4 dim scores
    salvage_ok = overall is not None and len(flat) == 4
    return {
        "ok": salvage_ok,
        "mode": "salvage" if salvage_ok else "fail",
        "overall_score": overall,
        "rating": rating,
        "dimensions": flat,
        "error_identified": error,
        "raw": text,
    }


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/finprm-mlx/mlx-q4")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.1)
    args = ap.parse_args()

    log(f"loading MLX model from {args.model}")
    model, tokenizer = load(args.model)
    log("model loaded")

    rows = [json.loads(l) for l in open(args.input)]
    if args.n > 0:
        rows = rows[: args.n]
    log(f"running PRM on {len(rows)} examples")

    out_f = open(args.output, "w")
    parse_failures = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        try:
            text = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=args.max_new, verbose=False,
            )
        except Exception as e:
            log(f"  [{i}] generate error: {e}")
            text = ""

        pred = parse_prm_output(text)
        if not pred["ok"]:
            parse_failures += 1
        out_f.write(json.dumps({**r, "pred": pred}) + "\n")
        out_f.flush()

        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(rows) - i - 1) / rate if rate > 0 else 0
            log(f"  {i+1}/{len(rows)}  rate={rate:.2f}/s  parse_fail={parse_failures}  ETA={eta:.0f}s")

    out_f.close()
    log(f"done. wrote {args.output}. parse_failures={parse_failures}/{len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
