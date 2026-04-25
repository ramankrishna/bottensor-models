"""
Step 2 — data preparation for NPC Agentic.

Downloads GLM-5.1-Reasoning-1M-Cleaned + Hermes agent traces, synthesizes
identity examples, normalizes everything to a unified chat-message format,
dedupes, length-filters, writes:

  data/processed/sft.jsonl   — full training set (one JSON per line)
  data/processed/eval.jsonl  — 500 stratified held-out
  data/identity.jsonl        — raw identity examples (for audit)
  logs/data_stats.json       — counts per source, drops, final

Schema discovery:
  Each dataset's features are printed before formatting. If we encounter a
  schema we can't handle, we raise rather than guess — the runbook guardrail
  says don't paper over surprise schemas.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Fast HF downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from datasets import load_dataset, Dataset, concatenate_datasets, DownloadMode
from transformers import AutoTokenizer

import config  # sibling file

random.seed(config.RANDOM_SEED)


# ────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dirs() -> None:
    for p in [config.DATA_RAW, config.DATA_PROCESSED.parent,
              config.IDENTITY_PATH.parent, config.LOG_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def coerce_messages(
    system: Optional[str],
    user: str,
    assistant: str,
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return msgs


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def stable_hash(messages: List[Dict[str, str]]) -> str:
    """Dedup key: hash of user+assistant contents joined."""
    parts = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            parts.append(m["content"])
    return sha1("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()


# ────────────────────────────────────────────────────────────────────────
# Dataset loaders — each handles its own schema
# ────────────────────────────────────────────────────────────────────────
def format_glm_reasoning(raw: Any) -> Optional[List[Dict[str, str]]]:
    """
    GLM-5.1-Reasoning-1M-Cleaned schemas observed in the wild:
      - {"conversations": [{"from": "human/gpt", "value": "..."}, ...]}
      - {"messages": [{"role": "user/assistant/system", "content": "..."}, ...]}
      - {"prompt": "...", "response": "..."}  (plain Q/A)
      - {"question": "...", "answer": "...", "reasoning": "..."}
    Returns normalized messages or None if unparseable.
    """
    # Case 1: messages array
    if "messages" in raw and isinstance(raw["messages"], list):
        out = []
        for m in raw["messages"]:
            role = m.get("role")
            content = m.get("content")
            if role in ("system", "user", "assistant") and isinstance(content, str):
                out.append({"role": role, "content": content})
        if out and any(m["role"] == "assistant" for m in out):
            return out
        return None

    # Case 2: conversations (ShareGPT)
    if "conversations" in raw and isinstance(raw["conversations"], list):
        role_map = {"human": "user", "user": "user",
                    "gpt": "assistant", "assistant": "assistant",
                    "system": "system"}
        out = []
        for c in raw["conversations"]:
            fr = c.get("from") or c.get("role")
            val = c.get("value") or c.get("content")
            if fr and val and fr in role_map:
                out.append({"role": role_map[fr], "content": str(val)})
        if out and any(m["role"] == "assistant" for m in out):
            return out
        return None

    # Case 3: prompt / response (may have optional system + reasoning)
    prompt = raw.get("prompt") or raw.get("question") or raw.get("input")
    response = raw.get("response") or raw.get("answer") or raw.get("output")
    system = raw.get("system") or raw.get("system_prompt")
    reasoning = raw.get("reasoning") or raw.get("thinking") or raw.get("think")
    if prompt and response:
        # Weave reasoning into assistant content if provided separately.
        # GLM-5.1 traces commonly have <think>...</think> already embedded;
        # if the reasoning field is separate, wrap it.
        if reasoning and "<think>" not in str(response):
            assistant = f"<think>\n{reasoning}\n</think>\n\n{response}"
        else:
            assistant = str(response)
        return coerce_messages(
            system=str(system) if system else None,
            user=str(prompt),
            assistant=assistant,
        )

    return None


def format_hermes_agent(raw: Any) -> Optional[List[Dict[str, str]]]:
    """
    Hermes agent traces schemas:
      - {"conversations": [...]}   (ShareGPT)
      - {"messages": [...]}
      - includes tool/function roles — we KEEP role="tool" because Qwen2.5's
        chat template natively wraps tool messages as
        `<|im_start|>user\\n<tool_response>...</tool_response><|im_end|>`.
        Combined with DataCollatorForCompletionOnlyLM's response_template
        `<|im_start|>assistant\\n`, this correctly masks environment output
        from the loss — only real assistant turns (tool_call emissions +
        final answers) are trained on.
    """
    # messages-format
    if "messages" in raw and isinstance(raw["messages"], list):
        out = []
        for m in raw["messages"]:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            if not role or content is None:
                continue
            role = {"human": "user", "gpt": "assistant", "function": "tool"}.get(role, role)
            if role not in ("system", "user", "assistant", "tool"):
                continue
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({"role": role, "content": content})
        return out if any(m["role"] == "assistant" for m in out) else None

    # conversations-format (ShareGPT-like)
    if "conversations" in raw and isinstance(raw["conversations"], list):
        role_map = {"human": "user", "user": "user",
                    "gpt": "assistant", "assistant": "assistant",
                    "system": "system",
                    "tool": "tool", "function": "tool",
                    "tool_response": "tool", "observation": "tool"}
        out = []
        for c in raw["conversations"]:
            fr = c.get("from") or c.get("role")
            val = c.get("value") or c.get("content")
            if fr and val and fr in role_map:
                content = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
                out.append({"role": role_map[fr], "content": content})
        return out if out and any(m["role"] == "assistant" for m in out) else None

    return None


# ────────────────────────────────────────────────────────────────────────
# Identity synthesis (v2)
# ────────────────────────────────────────────────────────────────────────
def _weighted_choice(items: List[Tuple[Any, float]], rng: random.Random) -> Any:
    """Pick an item from a list of (item, weight) tuples."""
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for item, w in items:
        acc += w
        if r <= acc:
            return item
    return items[-1][0]


def build_identity_examples(n_target: int) -> List[Dict[str, Any]]:
    """
    v2 identity generator.

    Changes from v1:
      - 3000 examples (was 750) — 4× the signal weight
      - Each example picks a system prompt from 3 cohorts with weights:
          40% NO system prompt       (critical — real user behavior)
          30% "You are a helpful assistant"  (what base Qwen ships with)
          30% "You are NPC Agentic, built by Bottensor"
      - 10 response templates rotate (vs 1 in v1) to avoid parroting
      - Every response mentions both Bottensor AND Ram Krishna
      - Paraphrase space broadened: more prefixes/suffixes, typo variants,
        lowercase variants
      - Prompt de-dup on (system_prompt, user_prompt) pair so the same user
        prompt can appear under multiple system cohorts.
    """
    rng = random.Random(config.RANDOM_SEED)
    prompts = config.IDENTITY_SEED_PROMPTS
    prefixes = config.IDENTITY_PARAPHRASE_PREFIXES
    suffixes = config.IDENTITY_PARAPHRASE_SUFFIXES
    cohorts = config.IDENTITY_SYSTEM_COHORTS
    templates = config.IDENTITY_RESPONSE_TEMPLATES

    out: List[Dict[str, Any]] = []
    seen: set = set()
    attempts = 0
    max_attempts = n_target * 30

    while len(out) < n_target and attempts < max_attempts:
        attempts += 1
        seed = rng.choice(prompts)
        pfx = rng.choice(prefixes)
        sfx = rng.choice(suffixes)
        # Drop seed's trailing punctuation if we're applying our own suffix
        base = seed if sfx in ("", " please", " please?", " actually",
                                " right now", " btw", " haha") else seed.rstrip("?.!")
        prompt = f"{pfx}{base}{sfx}".strip()
        # Occasional all-lowercase to mimic casual users
        if rng.random() < 0.25:
            prompt = prompt.lower()
        # Rare typo simulation for robustness
        if rng.random() < 0.05 and len(prompt) > 10:
            i = rng.randrange(1, len(prompt) - 1)
            prompt = prompt[:i] + prompt[i+1:]  # drop one char

        # Pick a system cohort (None or a string)
        system = _weighted_choice(cohorts, rng)
        reply = rng.choice(templates)

        key = (system, prompt)
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "messages": coerce_messages(
                system=system, user=prompt, assistant=reply,
            ),
            "source": "identity",
        })

    rng.shuffle(out)
    return out[:n_target]


# ────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_dirs()

    # ── Download + inspect schemas (reported to stdout + logs) ─────────
    log("== Loading GLM-5.1-Reasoning-1M-Cleaned ==")
    glm_cfg = config.DATASETS["glm_reasoning"]
    log(f"  config: {glm_cfg.get('config_name','<default>')}")
    glm_ds = load_dataset(
        glm_cfg["hf_id"],
        glm_cfg.get("config_name"),
        split=glm_cfg["split"],
        cache_dir=str(config.DATA_RAW),
    )
    log(f"  GLM raw rows: {len(glm_ds):,}")
    log(f"  GLM features: {list(glm_ds.features)}")
    log(f"  GLM sample keys: {list(glm_ds[0].keys())}")
    # Sample preview (truncated) — useful for format verification
    preview = glm_ds[0]
    log(f"  GLM row[0] preview: " +
        json.dumps({k: (str(v)[:120] + '…' if isinstance(v, str) and len(str(v)) > 120 else v)
                    for k, v in preview.items()}, ensure_ascii=False, default=str)[:400])

    # Subsample — use shuffle+select for deterministic sample regardless of ordering
    if len(glm_ds) > glm_cfg["subset_size"]:
        glm_ds = glm_ds.shuffle(seed=config.RANDOM_SEED).select(range(glm_cfg["subset_size"]))
        log(f"  GLM sampled: {len(glm_ds):,}")

    # v2: skip Hermes entirely (flag in config). Its traces had loopy tool-call
    # JSON and syntax errors that bled into v1 generation quality.
    hermes_ds = None
    if getattr(config, "DROP_HERMES_AGENT", False):
        log("")
        log("== Skipping Hermes (DROP_HERMES_AGENT=True) ==")
    else:
        log("")
        log("== Loading Hermes agent traces ==")
        hermes_cfg = config.DATASETS["hermes_agent"]
        hermes_parts: List[Dataset] = []
        for subset in hermes_cfg["subsets"]:
            try:
                d = load_dataset(
                    hermes_cfg["hf_id"],
                    name=subset,
                    split="train",
                    cache_dir=str(config.DATA_RAW),
                )
            except Exception as e:
                log(f"  !! Hermes subset '{subset}' failed with name=: {e}")
                try:
                    d = load_dataset(hermes_cfg["hf_id"], subset, split="train",
                                     cache_dir=str(config.DATA_RAW))
                except Exception as e2:
                    log(f"  !! Hermes subset '{subset}' second attempt failed: {e2}")
                    raise
            log(f"  Hermes[{subset}] rows: {len(d):,}  features: {list(d.features)}")
            if len(d):
                log(f"  Hermes[{subset}] row[0] keys: {list(d[0].keys())}")
            hermes_parts.append(d)
        hermes_ds = concatenate_datasets(hermes_parts) if hermes_parts else None

    # ── Format each source ─────────────────────────────────────────────
    log("")
    log("== Formatting GLM ==")
    formatted: List[Dict[str, Any]] = []
    glm_fail = 0
    for row in glm_ds:
        msgs = format_glm_reasoning(row)
        if msgs is None:
            glm_fail += 1
            continue
        formatted.append({"messages": msgs, "source": "glm_reasoning"})
    log(f"  GLM formatted: {len(formatted):,} / {len(glm_ds):,}  (failed to parse: {glm_fail})")
    if glm_fail / max(1, len(glm_ds)) > 0.10:
        raise RuntimeError(
            f"More than 10% of GLM rows failed schema parse ({glm_fail}/{len(glm_ds)}). "
            "Inspect dataset features and extend format_glm_reasoning()."
        )

    # v2 GLM quality filter
    glm_max_toks = getattr(config, "GLM_MAX_TRACE_TOKENS", None)  # pre-tokenizer heuristic
    glm_min_chars = getattr(config, "GLM_MIN_RESPONSE_CHARS", 0)
    if glm_max_toks or glm_min_chars:
        before = len(formatted)
        filtered: List[Dict[str, Any]] = []
        drop_short = drop_unterminated = drop_long = 0
        for ex in formatted:
            if ex["source"] != "glm_reasoning":
                filtered.append(ex)
                continue
            # Concatenate all assistant messages
            assistant_text = " ".join(
                m["content"] for m in ex["messages"] if m["role"] == "assistant"
            )
            # Drop if response is too short (drafty / no real answer)
            if len(assistant_text) < glm_min_chars:
                drop_short += 1
                continue
            # Drop if <think> was opened but never closed (truncated trace)
            if "<think>" in assistant_text and "</think>" not in assistant_text:
                drop_unterminated += 1
                continue
            # Crude character-based length filter (pre-tokenizer cheap check)
            # ~3.5 chars/token for English; so glm_max_toks × 3.5 gives the char budget
            if glm_max_toks and len(assistant_text) > glm_max_toks * 4:
                drop_long += 1
                continue
            filtered.append(ex)
        formatted = filtered
        log(f"  GLM quality filter: kept {sum(1 for x in formatted if x['source']=='glm_reasoning'):,}  "
            f"(dropped short={drop_short}, unterminated-think={drop_unterminated}, too-long={drop_long})")

    log("")
    log("== Formatting Hermes ==")
    hermes_fail = 0
    n_before_hermes = len(formatted)
    if hermes_ds is not None:
        for row in hermes_ds:
            msgs = format_hermes_agent(row)
            if msgs is None:
                hermes_fail += 1
                continue
            formatted.append({"messages": msgs, "source": "hermes_agent"})
    hermes_ok = len(formatted) - n_before_hermes
    log(f"  Hermes formatted: {hermes_ok:,}  (failed: {hermes_fail})")

    # ── Identity synthesis ─────────────────────────────────────────────
    log("")
    log("== Synthesizing identity examples ==")
    identity = build_identity_examples(config.IDENTITY_EXAMPLES)
    log(f"  Identity generated: {len(identity):,}")
    # Also write raw identity for audit
    write_jsonl(config.IDENTITY_PATH, identity)
    formatted.extend(identity)

    # ── Dedup (exact match on user+assistant content) ──────────────────
    log("")
    log("== Deduplicating ==")
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for ex in formatted:
        h = stable_hash(ex["messages"])
        if h in seen:
            continue
        seen.add(h)
        deduped.append(ex)
    dups = len(formatted) - len(deduped)
    log(f"  dropped duplicates: {dups:,}  ({100*dups/max(1,len(formatted)):.2f}%)")

    # ── Length filter (uses Qwen2.5 tokenizer) ─────────────────────────
    log("")
    log("== Length filter ==")
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, trust_remote_code=True)
    if not tok.chat_template:
        raise RuntimeError("Qwen2.5 tokenizer has no chat_template — aborting.")

    kept: List[Dict[str, Any]] = []
    dropped_len = 0
    report_every = 5000
    for i, ex in enumerate(deduped, 1):
        try:
            rendered = tok.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False,
            )
            n_tokens = len(tok.encode(rendered, add_special_tokens=False))
        except Exception:
            dropped_len += 1
            continue
        if n_tokens > config.MAX_SEQ_LEN:
            dropped_len += 1
            continue
        kept.append(ex)
        if i % report_every == 0:
            log(f"  ... scanned {i:,}  kept {len(kept):,}  dropped {dropped_len:,}")
    log(f"  dropped over-length (> {config.MAX_SEQ_LEN} tokens): {dropped_len:,}")

    # ── Stratified eval split ──────────────────────────────────────────
    log("")
    log("== Stratified eval split ==")
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(kept)

    # Take EVAL_SIZE examples, stratified on source
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for ex in kept:
        by_source.setdefault(ex["source"], []).append(ex)
    counts = {s: len(v) for s, v in by_source.items()}
    total = sum(counts.values())
    eval_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    for s, rows in by_source.items():
        # proportional split; guarantee at least 10 per source if present
        per_src_target = max(10, int(round(config.EVAL_SIZE * counts[s] / total)))
        per_src_target = min(per_src_target, len(rows))
        eval_rows.extend(rows[:per_src_target])
        train_rows.extend(rows[per_src_target:])

    # Trim eval to exactly EVAL_SIZE if we went slightly over
    if len(eval_rows) > config.EVAL_SIZE:
        train_rows.extend(eval_rows[config.EVAL_SIZE:])
        eval_rows = eval_rows[:config.EVAL_SIZE]

    log(f"  eval rows: {len(eval_rows):,}")
    log(f"  train rows: {len(train_rows):,}")
    log(f"  eval source breakdown: {Counter(x['source'] for x in eval_rows)}")

    # ── Write outputs ──────────────────────────────────────────────────
    log("")
    log("== Writing outputs ==")
    n_train = write_jsonl(config.DATA_PROCESSED, train_rows)
    n_eval = write_jsonl(config.EVAL_SPLIT, eval_rows)
    log(f"  wrote {n_train:,} → {config.DATA_PROCESSED}")
    log(f"  wrote {n_eval:,} → {config.EVAL_SPLIT}")

    # ── Final stats ────────────────────────────────────────────────────
    final_by_source = Counter(x["source"] for x in train_rows)
    eval_by_source = Counter(x["source"] for x in eval_rows)
    stats = {
        "raw_counts": {
            "glm_reasoning_raw": len(glm_ds),
            "hermes_agent_raw": len(hermes_ds) if hermes_ds is not None else 0,
            "identity": len(identity),
        },
        "parse_failures": {
            "glm_reasoning": glm_fail,
            "hermes_agent": hermes_fail,
        },
        "dedup_drops": dups,
        "length_drops": dropped_len,
        "train_final": n_train,
        "eval_final": n_eval,
        "train_by_source": dict(final_by_source),
        "eval_by_source": dict(eval_by_source),
        "max_seq_len": config.MAX_SEQ_LEN,
        "base_model": config.BASE_MODEL,
        "random_seed": config.RANDOM_SEED,
    }
    config.STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.STATS_PATH.open("w") as f:
        json.dump(stats, f, indent=2)
    log(f"  stats → {config.STATS_PATH}")

    log("")
    log("=" * 64)
    log("DATA PREP DONE.")
    log(f"  training examples:  {n_train:,}")
    log(f"  eval examples:      {n_eval:,}")
    log(f"  by source (train):  {dict(final_by_source)}")
    log("=" * 64)
    log("")
    log("Next step: eyeball the first 20 lines of sft.jsonl + the stats JSON,")
    log("then run 02_train.py (nohup recommended).")


if __name__ == "__main__":
    sys.exit(main())
