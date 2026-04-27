"""
Step 5 — merge the LoRA adapter into base Qwen2.5-7B-Instruct, save as FP16.

Loads the base in FP16 (NOT 4-bit — merging requires full precision base),
applies the trained LoRA via PeftModel, calls merge_and_unload(), saves the
full FP16 model plus tokenizer into `merged/`.

VRAM cost: ~14-15 GB base + ~2 GB for the merge op. Safely fits on A40.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 64)
    log("NPC Agentic 7B — merge LoRA → FP16")
    log("=" * 64)

    if not config.FINAL_ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {config.FINAL_ADAPTER_DIR}. "
            "Run 02_train.py first."
        )

    config.MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load base in FP16 ──────────────────────────────────────────────
    log("")
    log(f"== Loading base {config.BASE_MODEL} in FP16 ==")
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    log(f"  base loaded. peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # ── Load LoRA adapter ──────────────────────────────────────────────
    log("")
    log(f"== Loading LoRA adapter from {config.FINAL_ADAPTER_DIR} ==")
    model = PeftModel.from_pretrained(base, str(config.FINAL_ADAPTER_DIR))
    log(f"  adapter loaded. peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # ── Merge ──────────────────────────────────────────────────────────
    log("")
    log("== Merging adapter into base weights ==")
    merged = model.merge_and_unload()
    log(f"  merged. peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    # Ensure fp16 (merge_and_unload preserves dtype)
    merged = merged.to(torch.float16)

    # ── Save merged weights ────────────────────────────────────────────
    log("")
    log(f"== Saving merged model to {config.MERGED_DIR} ==")
    merged.save_pretrained(
        str(config.MERGED_DIR),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Tokenizer: copy from base (the adapter dir may have a smaller or altered copy)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, trust_remote_code=True)
    tok.save_pretrained(str(config.MERGED_DIR))
    # Also copy the chat_template file explicitly if it exists
    try:
        from huggingface_hub import hf_hub_download
        tpl_path = hf_hub_download(config.BASE_MODEL, "chat_template.jinja")
        shutil.copy2(tpl_path, config.MERGED_DIR / "chat_template.jinja")
        log("  copied chat_template.jinja from base")
    except Exception:
        log("  (no standalone chat_template.jinja; tokenizer_config carries it)")

    # ── Report sizes ───────────────────────────────────────────────────
    total_size = sum(p.stat().st_size for p in Path(config.MERGED_DIR).rglob("*") if p.is_file())
    log(f"  wrote {len(list(config.MERGED_DIR.iterdir()))} files  "
        f"total {total_size/1e9:.2f} GB")
    for p in sorted(config.MERGED_DIR.iterdir()):
        log(f"    {p.stat().st_size/1e9:.2f} GB  {p.name}")

    log("")
    log("=" * 64)
    log("MERGE DONE.")
    log(f"  merged model at: {config.MERGED_DIR}")
    log("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
