"""
Step 7 — push merged FP16, GPTQ 4-bit, and LoRA adapter to HuggingFace.

Creates three public repos under `config.HF_ORG` (currently set to
`ramankrishna10` because the `bottensor` org doesn't exist yet).

Writes a proper model card to each repo.

Guardrails: if a repo already exists with content, this still uploads via
`upload_folder` which will overwrite files by commit. For first publish
this is fine; for re-publish be aware you're making a new commit on top.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, upload_folder, create_repo

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────
# Model cards
# ────────────────────────────────────────────────────────────────────────
CARD_FP16 = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - reasoning
  - agent
  - bottensor
  - npc
language:
  - en
library_name: transformers
---

# NPC Agentic 7B (v1)

A 7B long-form reasoning and agent-trace specialist from the Bottensor NPC
Model Family.

## Overview

NPC Agentic v1 is fine-tuned from Qwen2.5-7B-Instruct on a mix of distilled
reasoning traces (GLM-5.1) and agent tool-use traces (Hermes). It's built for
structured multi-step reasoning with explicit `<think>` blocks, agentic /
tool-calling workflows, and identity-bound conversations as the NPC Agentic
persona.

## Training

- **Base:** Qwen/Qwen2.5-7B-Instruct
- **Method:** QLoRA SFT (r=64, α=128), merged to FP16
- **Context during training:** 8K (inherits 128K from base at inference)
- **Epochs:** 2 (effective batch size 16, cosine LR 2e-4, adamw_8bit, bf16)
- **Total optimizer steps:** 11,410 over ~96 GPU-hours on a single A40
- **Trainable params:** 161.5M (3.2% of the 5.05B-param 4-bit base)
- **Final eval loss:** 0.7025 (on held-out SFT split)
- **Training data mix (~91K examples):**
  - GLM-5.1-Reasoning-1M-Cleaned (main split, sampled 100K → 87K kept after 8K length filter)
  - Hermes-agent-reasoning-traces (glm-5.1 + kimi subsets, 14.7K → 3.6K kept)
  - Bottensor identity replay (750 synthetic examples)
- Training dataset is proprietary and not released.

## What it's good at

- **Long structured reasoning** — emits `<think>` blocks then concludes with an answer; strong at multi-step decomposition (system design, root-cause analysis, algorithmic reasoning)
- **Identity as NPC Agentic / Bottensor** — 100% recall on canonical identity prompts
- **Agent / tool-call shaping** — follows Hermes-style `<tool_call>` / `<tool_response>` patterns

## Known limitations (be specific)

- **GSM8K regression vs base.** On GSM8K 100-sample test:
  - Base Qwen2.5-7B-Instruct: **61%**
  - NPC Agentic v1: **~25%**
  - Cause: the model learned to emit long `<think>` blocks but often doesn't terminate arithmetic cleanly under greedy/low-temp decoding, and direct-arithmetic quality regressed.
  - **Recommendation:** for math-heavy workflows, use the base `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2.5-Math-7B-Instruct` instead. A v2 with stronger reasoning data (OpenThoughts-114k at 16K) is planned.
- **8K training context** means long-reasoning samples were truncated during training; not validated past 16K.
- **Small model** — will hallucinate on unfamiliar domains.
- **Not for safety-critical decisions** (medical, legal, financial).

## Intended use

- Multi-step reasoning with explicit work-showing
- Agent / tool-use workflows
- Structured problem-solving where the model benefits from thinking out loud
- As a base for further fine-tuning on reasoning or domain-specific data

## Out of scope

- Direct GSM8K-style arithmetic (use base or Qwen-Math)
- Creative writing, roleplay
- Medical / legal / financial advice
- Safety-critical decisions

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("ramankrishna10/npc-agentic-7b")
model = AutoModelForCausalLM.from_pretrained(
    "ramankrishna10/npc-agentic-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Design an event-sourced microservice with exactly-once command handling."},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Citation

Coming soon.

---

Built by [Bottensor](https://bottensor.xyz).
"""


CARD_GPTQ = """---
license: apache-2.0
base_model: {base}
tags:
  - reasoning
  - agent
  - bottensor
  - npc
  - gptq
  - quantized
language:
  - en
library_name: transformers
---

# NPC Agentic 7B — GPTQ 4-bit

W4A16 GPTQ-quantized build of [`{base}`]({base_url}) for fast, memory-efficient
inference (loads in ~5 GB VRAM, ideal for vLLM serving).

See the [FP16 reference card]({base_url}) for the full training recipe, eval
numbers, and known limitations (particularly the GSM8K regression vs base —
use base Qwen2.5 or Qwen2.5-Math-7B for math-heavy workflows).

## Quantization details

- **Method:** GPTQ via `llm-compressor`
- **Scheme:** W4A16 (4-bit weights, fp16 activations)
- **Group size:** 128
- **Desc-act:** true
- **Symmetric:** false
- **Calibration:** 512 samples from the training set, 2048 tokens each
- **Ignored layers:** `lm_head` (kept in full precision)
- **Size on disk:** ~4.5 GB

## Inference (vLLM)

```python
from vllm import LLM, SamplingParams
llm = LLM(model="{repo}", dtype="float16")
out = llm.generate(
    ["Design an event-sourced microservice with exactly-once command handling."],
    SamplingParams(max_tokens=1024, temperature=0.7, top_p=0.9),
)
print(out[0].outputs[0].text)
```

## See also

- [`{base}`]({base_url}) — FP16 reference
- `ramankrishna10/npc-agentic-7b-lora` — LoRA adapter for apply-on-base workflows

---

Built by [Bottensor](https://bottensor.xyz).
"""


CARD_LORA = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - peft
  - lora
  - bottensor
  - npc
library_name: peft
---

# NPC Agentic 7B — LoRA adapter

LoRA adapter for NPC Agentic 7B. Apply on top of `Qwen/Qwen2.5-7B-Instruct`
(or load the merged FP16 model from the sibling repo if you want a ready-to-run
artifact).

See [`ramankrishna10/npc-agentic-7b`](https://huggingface.co/ramankrishna10/npc-agentic-7b)
for the full training recipe, eval numbers, and known limitations.

## Training config

- rank = 64, alpha = 128, dropout = 0.05
- target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- trained with QLoRA 4-bit base, bf16 adapters, Unsloth + TRL 0.24
- 11,410 steps, 2 epochs, ~96 GPU-hours on A40

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16, device_map="auto",
)
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Optional: bake the adapter in for faster inference
# model = model.merge_and_unload()
```

## Adapter size

~616 MB safetensors (161.5M trainable params).

---

Built by [Bottensor](https://bottensor.xyz).
"""


# ────────────────────────────────────────────────────────────────────────
def write_readme(root: Path, text: str) -> None:
    (root / "README.md").write_text(text, encoding="utf-8")


def push(repo: str, local_dir: Path, card_text: str, api: HfApi) -> str:
    log(f"  — repo: {repo}")
    log(f"    source: {local_dir}  ({sum(p.stat().st_size for p in local_dir.rglob('*') if p.is_file())/1e9:.2f} GB)")

    # Create repo if missing
    create_repo(repo, private=False, exist_ok=True, repo_type="model", token=None)

    # Write card into local_dir so upload_folder sweeps it in
    write_readme(local_dir, card_text)

    url = upload_folder(
        folder_path=str(local_dir),
        repo_id=repo,
        repo_type="model",
        commit_message=f"Upload npc-agentic-7b ({datetime.utcnow().date().isoformat()})",
        ignore_patterns=[
            "*.pt", "*.pth", "*.bin.index.json.bak",
            "optimizer.pt", "scheduler.pt", "training_args.bin",
            "rng_state.pth", "trainer_state.json",
            ".ipynb_checkpoints/*", "__pycache__/*",
        ],
    )
    log(f"    ✓ uploaded → {url}")
    return url


def main() -> None:
    log("=" * 64)
    log("NPC Agentic 7B — push to HuggingFace Hub")
    log("=" * 64)
    log(f"  HF_ORG:     {config.HF_ORG}")
    log(f"    (note:    {config.HF_ORG_FALLBACK_NOTE})")
    log(f"  repos:      {config.REPO_NAME_FP16} / {config.REPO_NAME_GPTQ} / {config.REPO_NAME_LORA}")

    api = HfApi()
    me = api.whoami()
    log(f"  logged in as: {me['name']}")

    # ── FP16 merged ────────────────────────────────────────────────────
    log("")
    log("== FP16 merged ==")
    if not config.MERGED_DIR.exists():
        raise FileNotFoundError(f"No merged dir at {config.MERGED_DIR}")
    url_fp16 = push(config.REPO_NAME_FP16, config.MERGED_DIR, CARD_FP16, api)

    # ── GPTQ 4-bit ─────────────────────────────────────────────────────
    log("")
    log("== GPTQ 4-bit ==")
    if not config.QUANTIZED_DIR.exists():
        raise FileNotFoundError(f"No quantized dir at {config.QUANTIZED_DIR}")
    card_gptq = CARD_GPTQ.format(
        base=config.REPO_NAME_FP16,
        base_url=f"https://huggingface.co/{config.REPO_NAME_FP16}",
        repo=config.REPO_NAME_GPTQ,
    )
    url_gptq = push(config.REPO_NAME_GPTQ, config.QUANTIZED_DIR, card_gptq, api)

    # ── LoRA adapter ───────────────────────────────────────────────────
    log("")
    log("== LoRA adapter ==")
    if not config.FINAL_ADAPTER_DIR.exists():
        raise FileNotFoundError(f"No adapter dir at {config.FINAL_ADAPTER_DIR}")
    # Prepare a clean copy with just the adapter + tokenizer
    adapter_upload_dir = config.ROOT / "_lora_upload"
    if adapter_upload_dir.exists():
        shutil.rmtree(adapter_upload_dir)
    adapter_upload_dir.mkdir()
    keep = {"adapter_config.json", "adapter_model.safetensors",
            "adapter_model.bin", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "chat_template.jinja"}
    for f in config.FINAL_ADAPTER_DIR.iterdir():
        if f.name in keep:
            shutil.copy2(f, adapter_upload_dir / f.name)
    card_lora = CARD_LORA.format(repo=config.REPO_NAME_LORA)
    url_lora = push(config.REPO_NAME_LORA, adapter_upload_dir, card_lora, api)

    # ── GGUF quants ────────────────────────────────────────────────────
    gguf_dir = config.ROOT / "gguf"
    url_gguf = None
    if gguf_dir.exists() and any(gguf_dir.glob("*.gguf")):
        log("")
        log("== GGUF quants ==")
        # 07_gguf.py writes the README inside gguf_dir already, but make sure
        # we don't ship the fp16 intermediate (14 GB) to the GGUF repo.
        upload_gguf_dir = gguf_dir
        gguf_repo = f"{config.HF_ORG}/{config.MODEL_SHORTNAME}-gguf"
        log(f"  — repo: {gguf_repo}")
        log(f"    source: {upload_gguf_dir}")
        from huggingface_hub import create_repo as _create, upload_folder as _upload
        _create(gguf_repo, private=False, exist_ok=True, repo_type="model", token=None)
        url_gguf = _upload(
            folder_path=str(upload_gguf_dir),
            repo_id=gguf_repo,
            repo_type="model",
            commit_message=f"Upload npc-agentic-7b-gguf ({datetime.utcnow().date().isoformat()})",
            ignore_patterns=[
                "*-f16.gguf",          # don't ship the 14GB intermediate
                "*.pt", "*.pth",
                "__pycache__/*",
            ],
        )
        log(f"    ✓ uploaded → {url_gguf}")
    else:
        log("")
        log("== GGUF quants ==   (skipped — no gguf/ directory yet)")

    # ── Summary ────────────────────────────────────────────────────────
    log("")
    log("=" * 64)
    log("PUSH DONE.")
    log(f"  FP16:  https://huggingface.co/{config.REPO_NAME_FP16}")
    log(f"  GPTQ:  https://huggingface.co/{config.REPO_NAME_GPTQ}")
    log(f"  LoRA:  https://huggingface.co/{config.REPO_NAME_LORA}")
    if url_gguf:
        log(f"  GGUF:  https://huggingface.co/{config.HF_ORG}/{config.MODEL_SHORTNAME}-gguf")
    log("=" * 64)
    log("")
    log("Next: move this training run into /workspace/training/completed/")


if __name__ == "__main__":
    sys.exit(main())
