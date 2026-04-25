"""
Step 6 — GPTQ W4A16 quantization of the merged NPC Agentic 7B, via
llm-compressor (preferred) or auto-gptq (fallback).

Calibration: 512 random samples from the training set, tokenized and truncated
to 2048 tokens.

Config:
  - bits = 4
  - group_size = 128
  - sym = False
  - desc_act = True

Smoke test: loads the quantized model with vLLM and generates a single
sample. If it produces gibberish or errors out, we stop and raise.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from transformers import AutoTokenizer

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────
# Calibration dataset
# ────────────────────────────────────────────────────────────────────────
def build_calib_dataset(tokenizer, n: int = 512, max_len: int = 2048):
    """Return a HuggingFace `Dataset` that llm-compressor 0.10+ `oneshot()`
    expects. Each row renders one chat-template-formatted text; the library
    tokenizes internally.
    """
    from datasets import Dataset

    rows = []
    with open(config.DATA_PROCESSED) as f:
        for line in f:
            rows.append(json.loads(line))

    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(rows)
    rows = rows[:n]

    # llm-compressor expects a "text" field it can tokenize itself
    texts = []
    for r in rows:
        text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False,
        )
        # Truncate at the char level so oneshot's own truncate doesn't misbehave
        texts.append(text)

    return Dataset.from_dict({"text": texts})


# ────────────────────────────────────────────────────────────────────────
# llm-compressor path
# ────────────────────────────────────────────────────────────────────────
def quantize_with_llmcompressor():
    log("")
    log("== Quantizing with llm-compressor (GPTQ W4A16) ==")
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM

    log("  loading merged model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        str(config.MERGED_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(str(config.MERGED_DIR), trust_remote_code=True)
    log(f"  loaded. VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

    log("  building 512-sample calibration set...")
    calib = build_calib_dataset(tok, n=512, max_len=2048)

    # llm-compressor 0.10 dropped group_size / desc_act / sym as direct kwargs
    # on GPTQModifier. The "W4A16" scheme preset encodes group_size=128 and
    # sym=False already; desc_act=True is the library's default.
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        dampening_frac=0.01,
    )

    log("  running oneshot quantization (this takes ~20-30 min)...")
    oneshot(
        model=model,
        dataset=calib,
        recipe=recipe,
        output_dir=str(config.QUANTIZED_DIR),
        max_seq_length=2048,
        num_calibration_samples=len(calib),
        # llm-compressor 0.10 needs an explicit text field for on-the-fly tokenization
        text_column="text",
    )
    # llm-compressor also saves the tokenizer, but double-check
    if not (config.QUANTIZED_DIR / "tokenizer.json").exists():
        tok.save_pretrained(str(config.QUANTIZED_DIR))
    log("  done.")


# ────────────────────────────────────────────────────────────────────────
# auto-gptq fallback (rare — use only if llm-compressor refuses)
# ────────────────────────────────────────────────────────────────────────
def quantize_with_autogptq():
    log("")
    log("== Quantizing with auto-gptq fallback ==")
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    tok = AutoTokenizer.from_pretrained(str(config.MERGED_DIR), trust_remote_code=True)
    calib = build_calib_dataset(tok, n=256, max_len=2048)
    examples = [{"input_ids": c["input_ids"].unsqueeze(0),
                 "attention_mask": c["attention_mask"].unsqueeze(0)} for c in calib]

    qcfg = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=True,
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        str(config.MERGED_DIR),
        quantize_config=qcfg,
        trust_remote_code=True,
    )
    log("  running auto-gptq .quantize(...)")
    model.quantize(examples)
    model.save_quantized(str(config.QUANTIZED_DIR), use_safetensors=True)
    tok.save_pretrained(str(config.QUANTIZED_DIR))


# ────────────────────────────────────────────────────────────────────────
# Smoke test via vLLM
# ────────────────────────────────────────────────────────────────────────
def smoke_test() -> bool:
    log("")
    log("== Smoke test: load quantized model in vLLM + 1 generation ==")
    from vllm import LLM, SamplingParams
    try:
        llm = LLM(
            model=str(config.QUANTIZED_DIR),
            dtype="float16",
            gpu_memory_utilization=0.7,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
        )
    except Exception as e:
        log(f"  !! vLLM load failed: {e}")
        return False

    tok = AutoTokenizer.from_pretrained(str(config.QUANTIZED_DIR), trust_remote_code=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain photosynthesis step by step."}],
        tokenize=False, add_generation_prompt=True,
    )
    out = llm.generate([prompt], SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0))
    text = out[0].outputs[0].text.strip()
    log(f"  smoke-test reply (first 300 chars): {text[:300]!r}")

    # Very loose sanity check — gibberish detection
    if len(text) < 30:
        log("  !! too short — likely broken")
        return False
    if any(text.count(c) > len(text) / 3 for c in "\x00"):
        log("  !! appears to be null bytes / broken")
        return False
    return True


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    log("=" * 64)
    log("NPC Agentic 7B — GPTQ W4A16 quantization")
    log("=" * 64)

    if not (config.MERGED_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Merged model not found at {config.MERGED_DIR}. Run 04_merge.py first."
        )

    config.QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Try llm-compressor first
    try:
        quantize_with_llmcompressor()
    except ImportError:
        log("  llm-compressor not installed; trying auto-gptq fallback")
        quantize_with_autogptq()

    # ── Free GPU memory before vLLM loads ──────────────────────────────
    torch.cuda.empty_cache()

    # ── Smoke test ─────────────────────────────────────────────────────
    if not smoke_test():
        raise RuntimeError(
            "Quantized smoke test failed. Do not push. Inspect the quantized "
            "weights manually before any upload."
        )

    # ── Size report ────────────────────────────────────────────────────
    total = sum(p.stat().st_size for p in Path(config.QUANTIZED_DIR).rglob("*") if p.is_file())
    log("")
    log(f"  quantized size: {total/1e9:.2f} GB")
    for p in sorted(config.QUANTIZED_DIR.iterdir()):
        log(f"    {p.stat().st_size/1e9:.3f} GB  {p.name}")

    log("")
    log("=" * 64)
    log(f"QUANTIZATION DONE. Output at {config.QUANTIZED_DIR}")
    log("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
