#!/usr/bin/env python3
"""
NPC Fin-PRM evaluation harness — single-batch inference + parsing.

Loads the PRM (PEFT adapter on Qwen2.5-7B-Instruct), runs it on a batch of
{system, user} prompts, parses the JSON output, returns per-example dicts:

    {
      "ok": True/False,           # JSON parse success
      "overall_score": 0.85,
      "rating": "STRONG",
      "dimensions": {factual_accuracy: 0.9, ...},
      "raw": "<full text>",
    }

Usage:
    python3 run_prm.py --input val_examples.jsonl --output preds.jsonl \
        [--n 200] [--batch 4] [--load-4bit]

Defaults to bf16 on CUDA, fp16 on MPS. Use --load-4bit on CUDA pods
where VRAM is shared with another job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_prm_output(text: str) -> dict:
    """Extract the JSON object from PRM output. Returns {ok, overall_score, ...}."""
    import re
    # Strip leading/trailing junk; PRM outputs valid JSON normally
    # but sometimes wraps in ```json ... ```
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Find first { and matching } (greedy by depth)
    first = text.find("{")
    if first < 0:
        return {"ok": False, "raw": text}
    depth = 0
    end = -1
    for i, c in enumerate(text[first:], start=first):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {"ok": False, "raw": text}
    try:
        obj = json.loads(text[first:end])
    except Exception as e:
        return {"ok": False, "raw": text, "parse_error": str(e)}

    dims = obj.get("dimensions", {})
    flat_dims = {}
    for k, v in dims.items():
        score = v.get("score") if isinstance(v, dict) else v
        if isinstance(score, (int, float)):
            flat_dims[k] = score

    return {
        "ok": True,
        "overall_score": obj.get("overall_score"),
        "rating": obj.get("rating"),
        "dimensions": flat_dims,
        "error_identified": obj.get("error_identified"),
        "raw": text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with {system, user} per row")
    ap.add_argument("--output", required=True, help="Where to write {pred} per row")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="ramankrishna10/npc-fin-prm-7b")
    ap.add_argument("--n", type=int, default=0, help="Limit (0 = all)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--load-4bit", action="store_true",
                    help="Use bnb 4-bit (CUDA only). Default off.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if has_cuda:
        device, dtype = "cuda", torch.bfloat16
    elif has_mps:
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    log(f"device={device}  dtype={dtype}  base={args.base}  adapter={args.adapter}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if args.load_4bit and has_cuda:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        log("loading base in 4-bit NF4")
    log("loading base...")
    base = AutoModelForCausalLM.from_pretrained(args.base, device_map=device, **kw)
    log("attaching PRM adapter...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # ── Run ───────────────────────────────────────────────────────────
    rows = [json.loads(l) for l in open(args.input)]
    if args.n > 0:
        rows = rows[: args.n]
    log(f"running PRM on {len(rows)} examples (batch={args.batch})")

    out_f = open(args.output, "w")
    t0 = time.time()
    parse_failures = 0
    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["user"]},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            for r in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=False if args.temperature == 0 else True,
                temperature=args.temperature,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        for j, r in enumerate(batch):
            new_ids = gen[j][enc["input_ids"].shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True)
            pred = parse_prm_output(text)
            if not pred["ok"]:
                parse_failures += 1
            out = {**r, "pred": pred}
            out_f.write(json.dumps(out) + "\n")
        out_f.flush()
        if (i // args.batch) % 5 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(rows) - done) / rate if rate else 0
            log(f"  {done}/{len(rows)}  rate={rate:.1f}/s  parse_fail={parse_failures}  ETA={eta:.0f}s")
    out_f.close()
    log(f"done. wrote {args.output}. parse_failures={parse_failures}/{len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
