"""
GPTQ 4-bit quantization via auto-gptq.

Calibrates on 256 random training examples (chat-templated and truncated
to 2048 tokens) and writes a W4A16 model to ``output/npc-fast-1.7b-gptq/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

from config import DATASETS_REGISTRY, TRAINING_CONFIG
from data.loader import load_registry
from data.mixer import mix

LOG = logging.getLogger("npc-fast.export.quantize")

CALIB_N = 256
CALIB_MAX_LEN = 2048


def _calibration_samples(tokenizer) -> list[list[int]]:
    LOG.info("Building calibration set (%d samples, max_len=%d)", CALIB_N, CALIB_MAX_LEN)
    raw = load_registry(DATASETS_REGISTRY)
    train, _ = mix(raw, seed=TRAINING_CONFIG["seed"], val_split=0.05)
    rng = random.Random(TRAINING_CONFIG["seed"])
    rng.shuffle(train)
    calib_ids: list[list[int]] = []
    for ex in train:
        if len(calib_ids) >= CALIB_N:
            break
        text = tokenizer.apply_chat_template(
            ex.messages, tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:CALIB_MAX_LEN]
        if len(ids) < 32:
            continue
        calib_ids.append(ids)
    LOG.info("Calibration set: %d samples", len(calib_ids))
    return calib_ids


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_ids = _calibration_samples(tokenizer)
    # auto-gptq expects list of dicts with input_ids + attention_mask tensors
    calib_dataset = []
    for ids in calib_ids:
        t = torch.tensor([ids], dtype=torch.long)
        calib_dataset.append({"input_ids": t, "attention_mask": torch.ones_like(t)})

    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,
        sym=True,
    )

    LOG.info("Loading fp16 model for quantization: %s", args.model_path)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_path,
        quantize_config=quantize_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    LOG.info("Running GPTQ quantization...")
    model.quantize(calib_dataset)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out), use_safetensors=True)
    tokenizer.save_pretrained(str(out))

    # Persist a tiny export manifest for provenance
    (out / "export_manifest.json").write_text(json.dumps({
        "source": args.model_path,
        "bits": args.bits,
        "group_size": args.group_size,
        "calibration_samples": len(calib_ids),
        "calibration_max_len": CALIB_MAX_LEN,
    }, indent=2))
    LOG.info("GPTQ model saved → %s", out)


if __name__ == "__main__":
    main()
