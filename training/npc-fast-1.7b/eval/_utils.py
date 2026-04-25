"""Shared helpers for every eval module.

Centralizes model loading, JSON parsing, and result persistence so the four
eval scripts stay focused on what they actually measure.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EVAL_RESULTS_PATH

LOG = logging.getLogger("npc-fast.eval")


def load_model(model_path: str):
    LOG.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0][prompt_ids.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text: str) -> Optional[dict]:
    """Pull the first top-level JSON object out of free-form text."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    for end in range(len(m.group(0)), 0, -1):
        try:
            return json.loads(m.group(0)[:end])
        except json.JSONDecodeError:
            continue
    return None


def merge_results(new: dict) -> None:
    path = Path(EVAL_RESULTS_PATH)
    current: dict = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except json.JSONDecodeError:
            current = {}
    current.update(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    LOG.info("Eval results → %s", path)


def log_wandb(payload: dict) -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        pass
