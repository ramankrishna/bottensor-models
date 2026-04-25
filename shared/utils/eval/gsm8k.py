"""
GSM8K eval harness — robust extractor + accuracy loop.

Why we wrote our own
--------------------
NPC Agentic v1 measured GSM8K accuracy by "find the last number in the
output." For reasoning-style models that emit ``<think>`` blocks, that
finds an intermediate working number, not the final answer — v1 looked
36 points worse than it actually was.

``extract_gsm_answer`` tries five strategies in order:

1. ``#### N`` — gold GSM8K format (also what many fine-tuners learn to emit).
2. ``\\boxed{N}`` — R1 / o1 / reasoning-style final answer.
3. After the LAST ``</think>``: prefer ``"answer is N"`` / ``"= N"``,
   fall back to the first number in the post-think text.
4. Same ``"answer is N"`` patterns on the whole text.
5. Last number in the whole text (the original v1 fallback).

This recovered 30+ accuracy points on v1's reasoning runs.

Generation: greedy decoding (do_sample=False) — math is deterministic
and sampling adds noise that hurts accuracy.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence


_NUMBER_RE = re.compile(r"-?\d[\d,]*")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [gsm8k] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────
def extract_gsm_answer(text: str) -> str | None:
    """
    Robust GSM8K answer extractor. Returns the canonical (comma-stripped)
    number string, or ``None`` if nothing plausible was found.
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

    # 3. After the LAST </think>
    think_end = text.rfind("</think>")
    if think_end >= 0:
        post = text[think_end + len("</think>"):]
        for pat in (
            r"(?:final answer|the answer|answer(?:\s+is)?)(?:\s*[:=])?\s*\$?(-?\d[\d,]*)",
            r"=\s*\$?(-?\d[\d,]*)\s*$",
        ):
            m = re.search(pat, post, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).replace(",", "")
        m = _NUMBER_RE.search(post)
        if m:
            return m.group(0).replace(",", "")

    # 4. "answer is N" anywhere
    m = re.search(
        r"(?:final answer|the answer|answer(?:\s+is)?)(?:\s*[:=])?\s*\$?(-?\d[\d,]*)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")

    # 5. Last number in the text
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


# ─────────────────────────────────────────────────────────────────────
# Sample loading
# ─────────────────────────────────────────────────────────────────────
def load_gsm8k_samples(
    n: int,
    *,
    split: str = "test",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Pull ``n`` examples from GSM8K (main), shuffled with a fixed seed.
    Returns a list of ``{"q": str, "gold": str}`` dicts.
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    out: list[dict[str, Any]] = []
    for row in ds:
        m = re.search(r"####\s*(-?\d[\d,]*)", row["answer"])
        gold = m.group(1).replace(",", "") if m else None
        if gold is None:
            continue
        out.append({"q": row["question"], "gold": gold})
    return out


# ─────────────────────────────────────────────────────────────────────
# Accuracy loop
# ─────────────────────────────────────────────────────────────────────
GenerateFn = Callable[[str], str]
"""A function that takes a user prompt and returns the model's reply."""


def eval_gsm8k(
    samples: Sequence[dict[str, Any]],
    generate_fn: GenerateFn,
    *,
    label: str = "model",
    log_every: int = 10,
    dump_path: str | Path | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Run ``generate_fn`` on every sample, extract its answer, compare to
    gold. Returns ``(accuracy, per_sample_records)``.

    ``generate_fn`` is the model-specific bridge — typically a closure
    around ``model.generate(...)`` with greedy decoding and a budget of
    ~1024 new tokens. We don't construct it here because the right
    answer depends on whether you have an HF pipeline, vLLM, or remote
    API.

    Optionally dumps a markdown report to ``dump_path`` for spot-checks.
    """
    correct = 0
    records: list[dict[str, Any]] = []

    fout = open(dump_path, "w", encoding="utf-8") if dump_path else None
    if fout:
        fout.write(f"# GSM8K eval — {label}\n\n")

    for i, ex in enumerate(samples):
        out = generate_fn(ex["q"] + "\nLet's think step by step.")
        got = extract_gsm_answer(out)
        ok = got is not None and got == ex["gold"]
        correct += int(ok)
        records.append({
            "q": ex["q"], "gold": ex["gold"], "got": got,
            "ok": ok, "raw": out,
        })

        if fout:
            fout.write(
                f"## Sample {i+1}: {'✓' if ok else '✗'}\n\n"
                f"**Q:** {ex['q']}\n\n"
                f"**Gold:** `{ex['gold']}`  **Got:** `{got}`\n\n"
                f"```\n{out}\n```\n\n---\n\n"
            )

        if (i + 1) % log_every == 0:
            running = correct / (i + 1)
            _log(f"  [{label}] {i+1}/{len(samples)}  running acc = {100*running:.1f}%")

    acc = correct / max(1, len(samples))
    _log(f"  [{label}] FINAL: {correct}/{len(samples)} = {100*acc:.1f}%")

    if fout:
        fout.write(f"\n**Final: {correct}/{len(samples)} = {100*acc:.1f}%**\n")
        fout.close()

    return acc, records
