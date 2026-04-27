"""
Diagnostic — 20 GSM8K samples with temperature sampling to test whether the
fine-tuned model's GSM8K regression is an inference-config issue (degenerate
greedy loops) or a real training regression.

Config: temp=0.7, top_p=0.9, max_new_tokens=1024 — standard reasoning-model
inference settings that typically avoid greedy-decode attractors.

Uses the fixed extractor from 03b.
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
from unsloth import FastLanguageModel
from datasets import load_dataset
from peft import PeftModel

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Import the fixed extractor from the sibling file
sys.path.insert(0, str(Path(__file__).parent))
# Copy inline to avoid import weirdness
_NUMBER_RE = re.compile(r"-?\d[\d,]*")

def extract_gsm_answer(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"####\s*(-?\d[\d,]*)", text)
    if m: return m.group(1).replace(",", "")
    m = re.search(r"\\boxed\{\s*(-?\d[\d,]*)\s*\}", text)
    if m: return m.group(1).replace(",", "")
    think_end = text.rfind("</think>")
    if think_end >= 0:
        post = text[think_end + len("</think>"):]
        for pat in [
            r"(?:final answer|the answer|answer(?:\s+is)?)(?:\s*[:=])?\s*\$?(-?\d[\d,]*)",
            r"=\s*\$?(-?\d[\d,]*)\s*$",
        ]:
            m = re.search(pat, post, re.IGNORECASE | re.MULTILINE)
            if m: return m.group(1).replace(",", "")
        m = _NUMBER_RE.search(post)
        if m: return m.group(0).replace(",", "")
    m = re.search(r"(?:final answer|the answer|answer(?:\s+is)?)(?:\s*[:=])?\s*\$?(-?\d[\d,]*)", text, re.IGNORECASE)
    if m: return m.group(1).replace(",", "")
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def generate_sampling(model, tokenizer, user_text: str, max_new_tokens: int = 1024,
                      temperature: float = 0.7, top_p: float = 0.9,
                      seed: int = 0) -> str:
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": user_text + "\nLet's think step by step."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=config.MAX_SEQ_LEN - max_new_tokens,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    log("=" * 64)
    log("GSM8K sampling probe — 20 samples, temp=0.7 top_p=0.9")
    log("=" * 64)

    # Load 20 GSM8K samples with the same seed as the main eval, same subset
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.shuffle(seed=config.RANDOM_SEED).select(range(20))
    samples = []
    for row in ds:
        m = re.search(r"####\s*(-?\d[\d,]*)", row["answer"])
        gold = m.group(1).replace(",", "") if m else None
        samples.append({"question": row["question"], "gold": gold})

    log(f"  samples: {len(samples)}")

    log("")
    log("== Loading fine-tuned ==")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(config.FINAL_ADAPTER_DIR))
    FastLanguageModel.for_inference(model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    log("")
    log("== Generating with temperature sampling ==")
    dump = config.LOG_DIR / "eval_gsm8k_ft_sampled.md"
    correct = 0
    with dump.open("w") as f:
        f.write("# GSM8K — fine-tuned w/ temp=0.7 top_p=0.9\n\n")
        for i, ex in enumerate(samples):
            t0 = time.time()
            reply = generate_sampling(model, tok, ex["question"], seed=i)
            got = extract_gsm_answer(reply)
            ok = (got is not None and got == ex["gold"])
            if ok:
                correct += 1
            log(f"  [{i:2d}] ({time.time()-t0:.0f}s) got={got!r} gold={ex['gold']!r} {'✓' if ok else '✗'}")
            f.write(f"## [{i}] {'✓' if ok else '✗'} got={got!r} gold={ex['gold']!r}\n\n")
            f.write(f"**Q:** {ex['question']}\n\n")
            # Detect loops / degenerate output
            degenerate = (len(reply) > 500 and (reply.count("0000") > 3 or
                                                  any(reply.count(s * 8) > 0 for s in "1234567890") or
                                                  reply.count(reply[-200:] if len(reply) >= 200 else reply) > 2))
            if degenerate:
                f.write(f"**⚠️ DEGENERATE OUTPUT DETECTED**\n\n")
            f.write(f"**Reply:**\n\n```\n{reply[:2000]}{'...' if len(reply) > 2000 else ''}\n```\n\n---\n\n")

    acc = correct / len(samples)
    log("")
    log("=" * 64)
    log(f"SAMPLING PROBE DONE: {correct}/20 = {100*acc:.0f}%")
    log(f"  dump: {dump}")
    log(f"  (compare: greedy run was tracking ~25-30% on this model)")
    log("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
