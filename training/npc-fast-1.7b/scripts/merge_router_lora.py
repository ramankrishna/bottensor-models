"""Merge the router LoRA adapter into the base checkpoint.
Output: /workspace/npc-fast-trainer/output/checkpoints/final_merged_router/
"""
from __future__ import annotations
import logging, shutil
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOG = logging.getLogger("merge")

BASE    = "/workspace/npc-fast-trainer/output/checkpoints/final"
ADAPTER = "/workspace/npc-fast-trainer/output/router_lora"
OUT     = "/workspace/npc-fast-trainer/output/checkpoints/final_merged_router"

LOG.info("Loading base: %s", BASE)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
)
LOG.info("Attaching adapter: %s", ADAPTER)
model = PeftModel.from_pretrained(base, ADAPTER)
LOG.info("Merging and unloading adapter weights")
merged = model.merge_and_unload()
LOG.info("Saving merged model to: %s", OUT)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
LOG.info("Done.")
