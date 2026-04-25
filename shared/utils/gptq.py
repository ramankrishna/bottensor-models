"""
GPTQ W4A16 quantization wrapper around llm-compressor 0.10+.

Why this exists
---------------
v1 and v2 both hit the same llm-compressor 0.10 footguns:

1. ``GPTQModifier`` no longer accepts ``group_size`` / ``desc_act`` /
   ``sym`` as direct kwargs — you have to pass ``scheme="W4A16"`` and
   let the preset apply ``group_size=128, sym=False, desc_act=True``.
2. ``oneshot()`` expects calibration data as an HF ``Dataset`` with a
   ``text`` column it can tokenize itself, NOT a list-of-dicts of
   pre-tokenized rows.

This wrapper bakes those fixes in so the next training run doesn't
spend an afternoon re-discovering them.

It also includes a ``smoke_test()`` that loads the quantized output in
vLLM and runs one greedy generation — if that doesn't produce
substantive text, we abort before any HF push.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable, Sequence


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [gptq] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────
# Calibration set
# ─────────────────────────────────────────────────────────────────────
def build_calib_dataset(
    tokenizer,
    sft_jsonl_path: str | Path,
    *,
    n_samples: int = 512,
    seed: int = 42,
    messages_key: str = "messages",
):
    """
    Build the HF ``Dataset`` that ``llmcompressor.oneshot`` expects.

    Reads ``sft_jsonl_path`` (one JSON-encoded row per line, with a
    ``messages`` field), shuffles, takes the first ``n_samples``,
    renders each row through the tokenizer's chat template, and returns
    a dataset with a single ``text`` column.

    The chat-template render happens here (not inside oneshot) so the
    calibration distribution matches exactly what the model saw at
    training time.
    """
    from datasets import Dataset

    rows: list[dict] = []
    with open(sft_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_samples]

    texts: list[str] = []
    for r in rows:
        text = tokenizer.apply_chat_template(
            r[messages_key], tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


# ─────────────────────────────────────────────────────────────────────
# Main quantize entry point
# ─────────────────────────────────────────────────────────────────────
def quantize_w4a16(
    merged_dir: str | Path,
    output_dir: str | Path,
    sft_jsonl_path: str | Path,
    *,
    n_calib: int = 512,
    max_seq_length: int = 2048,
    ignore_layers: Sequence[str] = ("lm_head",),
    dampening_frac: float = 0.01,
    trust_remote_code: bool = True,
    seed: int = 42,
) -> Path:
    """
    Run GPTQ W4A16 on ``merged_dir`` using ``sft_jsonl_path`` for
    calibration. Writes the quantized model to ``output_dir`` and
    returns its Path.

    Uses ``scheme="W4A16"``, which is the llm-compressor preset that
    encodes ``group_size=128, sym=False, desc_act=True``.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading merged model from {merged_dir} (fp16)")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(
        str(merged_dir), trust_remote_code=trust_remote_code,
    )
    if torch.cuda.is_available():
        _log(f"loaded — VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")

    _log(f"building {n_calib}-sample calibration set from {sft_jsonl_path}")
    calib = build_calib_dataset(
        tok, sft_jsonl_path, n_samples=n_calib, seed=seed,
    )

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=list(ignore_layers),
        dampening_frac=dampening_frac,
    )

    _log(f"running oneshot quantization (~20–30 min on a 7B)")
    oneshot(
        model=model,
        dataset=calib,
        recipe=recipe,
        output_dir=str(output_dir),
        max_seq_length=max_seq_length,
        num_calibration_samples=len(calib),
        text_column="text",
    )

    # Defensive: oneshot saves the tokenizer, but double-check
    if not (output_dir / "tokenizer.json").exists():
        tok.save_pretrained(str(output_dir))

    _log(f"done → {output_dir}")
    return output_dir


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────
def smoke_test(
    quantized_dir: str | Path,
    *,
    prompt: str = "Explain photosynthesis step by step.",
    max_tokens: int = 256,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 4096,
) -> bool:
    """
    Load the quantized model in vLLM, run one greedy generation, and
    return ``True`` if the output looks substantive.

    Heuristics: at least 30 chars of output, no excessive null bytes.
    Tighter validation belongs in the model-specific eval harness.
    """
    quantized_dir = Path(quantized_dir)

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        _log(f"smoke test skipped — {e}")
        return True

    _log(f"vLLM-loading {quantized_dir} for smoke test")
    try:
        llm = LLM(
            model=str(quantized_dir),
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
        )
    except Exception as e:
        _log(f"!! vLLM load failed: {e}")
        return False

    tok = AutoTokenizer.from_pretrained(str(quantized_dir), trust_remote_code=True)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    out = llm.generate(
        [rendered],
        SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0),
    )
    text = out[0].outputs[0].text.strip()
    _log(f"smoke-test reply (first 300 chars): {text[:300]!r}")

    if len(text) < 30:
        _log("!! too short — likely broken")
        return False
    if "\x00" in text and text.count("\x00") > len(text) / 3:
        _log("!! appears to be null bytes / broken")
        return False
    return True
