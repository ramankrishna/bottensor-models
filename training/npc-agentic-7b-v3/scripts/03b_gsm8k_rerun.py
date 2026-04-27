"""
Step 4b — GSM8K rerun with fixed answer extractor + larger max_new_tokens.

Rationale:
The first eval used max_new_tokens=400 and a crude "last-number" extractor.
Both hurt the fine-tuned model because it emits long <think>...</think> blocks
(learned from GLM-5.1 reasoning traces) and often doesn't reach the final
answer within 400 tokens. The extractor then picked a random intermediate
number from inside the think block.

This rerun:
  - max_new_tokens=1024 (enough headroom for <think> + final answer)
  - smarter extraction: prefers #### N, then \\boxed{N}, then first number
    AFTER </think>, then last number as fallback.

Does NOT redo the 50 stratified prompts or identity scoring — those already
passed (identity 100%, reasoning samples high quality). We only re-score
GSM8K and update eval_report.json in place.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from unsloth import FastLanguageModel  # must be before transformers
from datasets import load_dataset
from peft import PeftModel

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────
# Fixed answer extractor
# ────────────────────────────────────────────────────────────────────────
_NUMBER_RE = re.compile(r"-?\d[\d,]*")


def extract_gsm_answer(text: str) -> str | None:
    """
    Robust GSM8K answer extraction. Preferences, in order:
      1. `####\\s*N` (gold GSM8K answer format, used by many fine-tuners)
      2. `\\boxed{N}` (R1/reasoning-style final answer)
      3. "answer is N" / "= N" near end
      4. First number AFTER the last `</think>` (strip reasoning scratchpad)
      5. Last number in the whole text (original fallback)
    """
    if not text:
        return None

    # 1. #### marker
    m = re.search(r"####\s*(-?\d[\d,]*)", text)
    if m:
        return m.group(1).replace(",", "")

    # 2. \boxed{N}
    m = re.search(r"\\boxed\{\s*(-?\d[\d,]*)\s*\}", text)
    if m:
        return m.group(1).replace(",", "")

    # 3. After the LAST </think>, take the first number
    think_end = text.rfind("</think>")
    if think_end >= 0:
        post = text[think_end + len("</think>"):]
        # Prefer "answer is N" / "answer: N" / "= N" patterns in post
        for pat in [
            r"(?:final answer|the answer|answer(?:\s+is)?)(?:\s*[:=])?\s*\$?(-?\d[\d,]*)",
            r"=\s*\$?(-?\d[\d,]*)\s*$",
        ]:
            m = re.search(pat, post, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).replace(",", "")
        # Else first number in the post-think text
        m = _NUMBER_RE.search(post)
        if m:
            return m.group(0).replace(",", "")

    # 4. Same "answer is N" patterns on the whole text as fallback
    for pat in [
        r"(?:final answer|the answer|answer(?:\s+is)?)(?:\s*[:=])?\s*\$?(-?\d[\d,]*)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "")

    # 5. Last number (original fallback)
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


# ────────────────────────────────────────────────────────────────────────
# Generation helper
# ────────────────────────────────────────────────────────────────────────
def generate(model, tokenizer, user_text: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "user", "content": user_text + "\nLet's think step by step."}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=config.MAX_SEQ_LEN - max_new_tokens,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # greedy — deterministic for math
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ────────────────────────────────────────────────────────────────────────
# GSM8K sampling + evaluation
# ────────────────────────────────────────────────────────────────────────
def load_gsm8k_samples(n: int) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.shuffle(seed=config.RANDOM_SEED).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        m = re.search(r"####\s*(-?\d[\d,]*)", row["answer"])
        gold = m.group(1).replace(",", "") if m else None
        out.append({"question": row["question"], "gold": gold})
    return out


def eval_gsm8k(model, tokenizer, samples, label: str, dump_path: Path,
               max_new_tokens: int = 1024) -> float:
    correct = 0
    with dump_path.open("w") as f:
        f.write(f"# GSM8K eval — {label}\n\n")
        for i, ex in enumerate(samples):
            reply = generate(model, tokenizer, ex["question"], max_new_tokens)
            got = extract_gsm_answer(reply)
            ok = (got is not None and got == ex["gold"])
            if ok:
                correct += 1
            # Dump first 10 + all wrongs for inspection
            if i < 10 or not ok:
                f.write(f"## [{i}] {'✓' if ok else '✗'} got={got!r} gold={ex['gold']!r}\n\n")
                f.write(f"**Question:** {ex['question']}\n\n")
                f.write(f"**Reply:**\n\n```\n{reply}\n```\n\n---\n\n")
            if i < 5 or i % 25 == 0:
                log(f"  [{label} sample {i}] got={got!r} gold={ex['gold']!r} {'✓' if ok else '✗'}")
    acc = correct / max(1, len(samples))
    log(f"  {label} GSM8K: {correct}/{len(samples)} = {100*acc:.1f}%")
    return acc


# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    log("=" * 64)
    log("GSM8K rerun with fixed extractor + max_new_tokens=1024")
    log("=" * 64)

    samples = load_gsm8k_samples(config.GSM8K_SAMPLES)
    log(f"  samples loaded: {len(samples)}")

    # ── BASE ──────────────────────────────────────────────────────────
    log("")
    log("== Loading BASE Qwen2.5-7B-Instruct in 4-bit ==")
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    FastLanguageModel.for_inference(base_model)
    if base_tok.pad_token_id is None:
        base_tok.pad_token_id = base_tok.eos_token_id

    log("")
    log("== GSM8K (base) ==")
    base_acc = eval_gsm8k(
        base_model, base_tok, samples, label="base",
        dump_path=config.LOG_DIR / "eval_gsm8k_base.md",
    )

    del base_model
    torch.cuda.empty_cache()

    # ── FINE-TUNED ────────────────────────────────────────────────────
    log("")
    log("== Loading FINE-TUNED (base + LoRA) ==")
    ft_model, ft_tok = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_model, str(config.FINAL_ADAPTER_DIR))
    FastLanguageModel.for_inference(ft_model)
    if ft_tok.pad_token_id is None:
        ft_tok.pad_token_id = ft_tok.eos_token_id

    log("")
    log("== GSM8K (fine-tuned) ==")
    ft_acc = eval_gsm8k(
        ft_model, ft_tok, samples, label="fine-tuned",
        dump_path=config.LOG_DIR / "eval_gsm8k_ft.md",
    )

    # ── Update eval_report.json in place ──────────────────────────────
    report_path = config.EVAL_REPORT
    if report_path.exists():
        with report_path.open() as f:
            report = json.load(f)
    else:
        report = {}

    # Store old numbers for reference
    old_gsm = report.get("gsm8k", {})
    report["gsm8k"] = {
        "n_samples": len(samples),
        "base": base_acc,
        "fine_tuned": ft_acc,
        "delta": ft_acc - base_acc,
        "max_new_tokens": 1024,
        "extractor": "fixed (#### → \\boxed → post-</think> → last-number)",
        "previous_run_max_new_tokens": old_gsm.get("max_new_tokens", 400),
        "previous_run_scores": {
            "base": old_gsm.get("base"),
            "fine_tuned": old_gsm.get("fine_tuned"),
        } if old_gsm else None,
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"  updated → {report_path}")

    log("")
    log("=" * 64)
    log("GSM8K RERUN DONE.")
    log(f"  base:        {100*base_acc:.1f}%")
    log(f"  fine-tuned:  {100*ft_acc:.1f}%")
    log(f"  delta:       {100*(ft_acc - base_acc):+.1f}")
    log(f"  dumps:       logs/eval_gsm8k_base.md, logs/eval_gsm8k_ft.md")
    log("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
