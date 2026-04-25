"""
Step 4 — evaluate fine-tuned NPC Agentic vs base Qwen2.5-7B-Instruct.

Generates side-by-side outputs on 50 eval prompts (stratified: 20 reasoning,
20 agent, 10 identity). Auto-scores identity accuracy. Optionally runs a
GSM8K sample for reasoning-quality signal.

Writes:
  logs/eval_samples.md    — human-readable side-by-side
  logs/eval_report.json   — structured scores
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from unsloth import FastLanguageModel  # before transformers
from datasets import load_dataset
from peft import PeftModel

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────
# Generation helpers
# ────────────────────────────────────────────────────────────────────────
def generate(model, tokenizer, messages: List[Dict[str, str]],
             max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    """Greedy-ish generation for evaluation. Returns just the new assistant text."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=config.MAX_SEQ_LEN - max_new_tokens).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ────────────────────────────────────────────────────────────────────────
# Stratified eval prompt selection
# ────────────────────────────────────────────────────────────────────────
def build_eval_prompts() -> List[Dict[str, Any]]:
    """
    Pick up to EVAL_GEN_REASONING reasoning, EVAL_GEN_AGENT agent, plus
    EVAL_GEN_IDENTITY identity prompts. Reads from the held-out eval split.
    """
    rows: List[Dict[str, Any]] = []
    with open(config.EVAL_SPLIT) as f:
        for line in f:
            rows.append(json.loads(line))

    rng = random.Random(config.RANDOM_SEED)
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    for v in by_source.values():
        rng.shuffle(v)

    picks: List[Dict[str, Any]] = []
    picks.extend(by_source.get("glm_reasoning", [])[: config.EVAL_GEN_REASONING])
    picks.extend(by_source.get("hermes_agent", [])[: config.EVAL_GEN_AGENT])

    # Identity: pick from the 10 seed prompts (not from eval split — those are
    # synthetic variants and we want to exercise the canonical forms).
    for seed in config.IDENTITY_SEED_PROMPTS:
        picks.append({
            "messages": [
                {"role": "system", "content": config.IDENTITY_SYSTEM},
                {"role": "user", "content": seed},
            ],
            "source": "identity",
        })

    return picks


# ────────────────────────────────────────────────────────────────────────
# Identity scoring
# ────────────────────────────────────────────────────────────────────────
def score_identity(response: str) -> bool:
    lo = response.lower()
    return ("npc agentic" in lo) and ("bottensor" in lo)


# ────────────────────────────────────────────────────────────────────────
# GSM8K optional benchmark
# ────────────────────────────────────────────────────────────────────────
def gsm8k_sample(n: int) -> List[Dict[str, Any]]:
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        log(f"  couldn't load GSM8K ({e}); skipping")
        return []
    ds = ds.shuffle(seed=config.RANDOM_SEED).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        # GSM8K answers like "...\n#### 42"
        m = re.search(r"####\s*(-?\d[\d,]*)", row["answer"])
        gold = m.group(1).replace(",", "") if m else None
        out.append({"question": row["question"], "gold": gold})
    return out


_GSM_ANS = re.compile(r"(?:####\s*)?(-?\d[\d,]*)")


def extract_gsm_answer(text: str) -> str | None:
    # Prefer #### answer form if present; else last number in the reply.
    m = re.search(r"####\s*(-?\d[\d,]*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else None


def eval_gsm8k(model, tokenizer, samples: List[Dict[str, Any]], label: str) -> float:
    correct = 0
    for i, ex in enumerate(samples):
        reply = generate(
            model, tokenizer,
            messages=[{"role": "user", "content": ex["question"] + "\nLet's think step by step."}],
            max_new_tokens=400,
            temperature=0.0,
        )
        got = extract_gsm_answer(reply)
        ok = (got is not None and got == ex["gold"])
        if ok:
            correct += 1
        if i < 3:
            log(f"  [{label} GSM8K sample {i}] got={got!r} gold={ex['gold']!r}  {'✓' if ok else '✗'}")
    acc = correct / max(1, len(samples))
    log(f"  {label} GSM8K: {correct}/{len(samples)} = {100*acc:.1f}%")
    return acc


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    log("=" * 64)
    log("NPC Agentic 7B — evaluation (base vs fine-tuned)")
    log("=" * 64)

    # ── Build stratified eval prompts ──────────────────────────────────
    prompts = build_eval_prompts()
    n_reasoning = sum(1 for p in prompts if p["source"] == "glm_reasoning")
    n_agent = sum(1 for p in prompts if p["source"] == "hermes_agent")
    n_identity = sum(1 for p in prompts if p["source"] == "identity")
    log(f"  eval prompts: {len(prompts)}  "
        f"({n_reasoning} reasoning, {n_agent} agent, {n_identity} identity)")

    # ── Load BASE model ────────────────────────────────────────────────
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

    # ── Generate with BASE ─────────────────────────────────────────────
    log("")
    log("== Generating with BASE ==")
    base_outputs: List[str] = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        out = generate(base_model, base_tok, p["messages"])
        base_outputs.append(out)
        if i < 3:
            log(f"  base[{i}] ({time.time()-t0:.1f}s): {out[:120]!r}")

    # GSM8K on base (optional)
    gsm_samples = gsm8k_sample(config.GSM8K_SAMPLES) if config.GSM8K_SAMPLES else []
    base_gsm_acc = None
    if gsm_samples:
        log("")
        log("== GSM8K (base) ==")
        base_gsm_acc = eval_gsm8k(base_model, base_tok, gsm_samples, label="base")

    # Free base model before loading fine-tuned
    del base_model
    torch.cuda.empty_cache()

    # ── Load fine-tuned (base + LoRA) ─────────────────────────────────
    log("")
    log("== Loading FINE-TUNED (base + LoRA adapter) ==")
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

    # ── Generate with fine-tuned ───────────────────────────────────────
    log("")
    log("== Generating with FINE-TUNED ==")
    ft_outputs: List[str] = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        out = generate(ft_model, ft_tok, p["messages"])
        ft_outputs.append(out)
        if i < 3:
            log(f"  ft[{i}] ({time.time()-t0:.1f}s): {out[:120]!r}")

    # GSM8K on fine-tuned
    ft_gsm_acc = None
    if gsm_samples:
        log("")
        log("== GSM8K (fine-tuned) ==")
        ft_gsm_acc = eval_gsm8k(ft_model, ft_tok, gsm_samples, label="fine-tuned")

    # ── Identity scoring ───────────────────────────────────────────────
    log("")
    log("== Identity scoring ==")
    base_id, ft_id = 0, 0
    identity_details = []
    for p, b, f in zip(prompts, base_outputs, ft_outputs):
        if p["source"] != "identity":
            continue
        ok_b = score_identity(b)
        ok_f = score_identity(f)
        base_id += int(ok_b)
        ft_id += int(ok_f)
        identity_details.append({
            "prompt": p["messages"][-1]["content"],
            "base_ok": ok_b,
            "ft_ok": ok_f,
            "base": b[:200],
            "ft": f[:200],
        })
    base_id_acc = base_id / max(1, n_identity)
    ft_id_acc = ft_id / max(1, n_identity)
    log(f"  identity accuracy — base: {base_id}/{n_identity} ({100*base_id_acc:.0f}%)  "
        f"ft: {ft_id}/{n_identity} ({100*ft_id_acc:.0f}%)")

    # ── Write side-by-side ─────────────────────────────────────────────
    log("")
    log("== Writing eval_samples.md ==")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with config.EVAL_SAMPLES_MD.open("w", encoding="utf-8") as f:
        f.write("# NPC Agentic 7B — eval samples (base vs fine-tuned)\n\n")
        for i, (p, b, fto) in enumerate(zip(prompts, base_outputs, ft_outputs)):
            user = p["messages"][-1]["content"]
            f.write(f"## [{i}] ({p['source']})\n\n")
            f.write(f"**Prompt:** {user}\n\n")
            f.write(f"**BASE:**\n\n```\n{b}\n```\n\n")
            f.write(f"**FINE-TUNED:**\n\n```\n{fto}\n```\n\n")
            f.write("---\n\n")
    log(f"  → {config.EVAL_SAMPLES_MD}")

    # ── Final report ───────────────────────────────────────────────────
    report = {
        "counts": {"reasoning": n_reasoning, "agent": n_agent, "identity": n_identity},
        "identity_accuracy": {
            "base": base_id_acc,
            "fine_tuned": ft_id_acc,
            "details": identity_details,
        },
        "gsm8k": {
            "n_samples": len(gsm_samples),
            "base": base_gsm_acc,
            "fine_tuned": ft_gsm_acc,
            "delta": (ft_gsm_acc - base_gsm_acc) if (ft_gsm_acc is not None and base_gsm_acc is not None) else None,
        },
        "notes": [
            "Identity accuracy of ~100% on FT is the target; base should score near 0.",
            "Reasoning/agent samples are for human review — see eval_samples.md.",
            "GSM8K is a loose proxy; a 5-10 point lift on 100 samples is meaningful.",
        ],
    }
    with config.EVAL_REPORT.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"  → {config.EVAL_REPORT}")

    log("")
    log("=" * 64)
    log("EVALUATION DONE.")
    log(f"  identity:  base {100*base_id_acc:.0f}%  →  fine-tuned {100*ft_id_acc:.0f}%")
    if ft_gsm_acc is not None and base_gsm_acc is not None:
        log(f"  GSM8K:     base {100*base_gsm_acc:.1f}%  →  fine-tuned {100*ft_gsm_acc:.1f}%  "
            f"(Δ {100*(ft_gsm_acc-base_gsm_acc):+.1f})")
    log(f"  side-by-side: {config.EVAL_SAMPLES_MD}")
    log(f"  report:       {config.EVAL_REPORT}")
    log("=" * 64)
    log("")
    log("STOP: inspect eval_samples.md + eval_report.json before merging.")


if __name__ == "__main__":
    sys.exit(main())
