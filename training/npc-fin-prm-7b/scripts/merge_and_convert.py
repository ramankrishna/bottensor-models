#!/usr/bin/env python3
"""
Mac-native PRM eval prep: merge PEFT adapter into fp16 base on CPU,
then convert to MLX 4-bit for fast Apple Silicon inference.

Steps:
  1. Load Qwen2.5-7B-Instruct in fp16 on CPU (~14 GB RAM, slow)
  2. Apply ramankrishna10/npc-fin-prm-7b adapter
  3. merge_and_unload() to bake the adapter into the weights
  4. Save merged model to /tmp/finprm-mlx/merged
  5. Convert merged → MLX 4-bit at /tmp/finprm-mlx/mlx-q4
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path("/tmp/finprm-mlx")
MERGED = ROOT / "merged"
MLX_DIR = ROOT / "mlx-q4"
ROOT.mkdir(parents=True, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] CPU fp16 merge...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print(f"[{time.strftime('%H:%M:%S')}] base loaded")

m = PeftModel.from_pretrained(base, "ramankrishna10/npc-fin-prm-7b")
print(f"[{time.strftime('%H:%M:%S')}] adapter attached")

m = m.merge_and_unload()
print(f"[{time.strftime('%H:%M:%S')}] merged")

m.save_pretrained(str(MERGED), safe_serialization=True)
tok.save_pretrained(str(MERGED))
print(f"[{time.strftime('%H:%M:%S')}] saved fp16 merged → {MERGED}")
