"""
HuggingFace Hub upload helpers + model-card generator.

Centralizes the "push merged / GPTQ / GGUF / LoRA artifact + write a card"
flow that NPC Fast and NPC Agentic v1/v2 each re-implement. New training
runs should use these helpers instead of copying push logic again.

Design notes
------------
- ``push_folder`` is idempotent on missing repos (``exist_ok=True``) and
  always writes ``README.md`` into the local folder before upload so HF's
  default model card is the one we wrote, not an empty placeholder.
- ``ignore_patterns`` excludes optimizer/scheduler/RNG state by default —
  these are large, useless to consumers, and routinely sneak into final
  checkpoint dirs.
- Cards are rendered from named templates in ``CARDS`` plus a ``meta``
  dict; each model dir can override or extend by passing its own ``card``
  string instead.
- Uses ``HF_HUB_ENABLE_HF_TRANSFER=1`` for fast multipart uploads (set in
  the environment by the caller; we don't override silently).

Auth: requires ``HF_TOKEN`` (or ``HUGGINGFACE_HUB_TOKEN``) in env, OR a
prior ``huggingface-cli login``. ``HfApi()`` picks up either.
"""
from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from huggingface_hub import HfApi, create_repo, upload_folder


# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
DEFAULT_IGNORE: tuple[str, ...] = (
    # PyTorch trainer state
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
    # Old-format index backups + deepspeed leftovers
    "*.bin.index.json.bak",
    "global_step*/",
    # Editor / OS noise
    ".ipynb_checkpoints/*",
    "__pycache__/*",
    ".DS_Store",
    "._*",
)

# ─────────────────────────────────────────────────────────────────────
# Card templates
# ─────────────────────────────────────────────────────────────────────
CARD_FP16 = """---
license: apache-2.0
base_model: {base_model}
tags:
{tags_yaml}
language:
  - en
library_name: transformers
---

# {display_name}

{summary}

## Training

- **Base:** `{base_model}`
- **Method:** {method}
- **Hardware:** {hardware}
- **Trainable params:** {trainable_params}
- **Final eval loss:** {final_eval_loss}
- **Dataset mix:** {dataset_summary}

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained(
    "{repo}", torch_dtype=torch.bfloat16, device_map="auto",
)

messages = [{{"role": "user", "content": "{example_prompt}"}}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

{extra_sections}

---

Built by [Bottensor](https://bottensor.xyz) — a Falcon Hash company.
"""


CARD_GPTQ = """---
license: apache-2.0
base_model: {fp16_repo}
tags:
{tags_yaml}
  - gptq
  - quantized
language:
  - en
library_name: transformers
---

# {display_name} — GPTQ 4-bit

W4A16 GPTQ-quantized build of [`{fp16_repo}`](https://huggingface.co/{fp16_repo})
for fast, memory-efficient inference (typically loads in ~5 GB VRAM, ideal for
vLLM serving).

See the [FP16 reference card](https://huggingface.co/{fp16_repo}) for the full
training recipe, eval numbers, and known limitations.

## Quantization details

- **Method:** GPTQ via `llm-compressor`
- **Scheme:** W4A16 (4-bit weights, fp16 activations)
- **Group size:** 128
- **Calibration:** {calib_samples} samples, {calib_max_len} tokens each
- **Ignored layers:** `lm_head` (kept in full precision)

## Inference (vLLM)

```python
from vllm import LLM, SamplingParams
llm = LLM(model="{repo}", dtype="float16")
out = llm.generate(
    ["{example_prompt}"],
    SamplingParams(max_tokens=1024, temperature=0.7, top_p=0.9),
)
print(out[0].outputs[0].text)
```

---

Built by [Bottensor](https://bottensor.xyz) — a Falcon Hash company.
"""


CARD_GGUF = """---
license: apache-2.0
base_model: {fp16_repo}
tags:
{tags_yaml}
  - gguf
  - quantized
  - llama.cpp
  - ollama
language:
  - en
library_name: gguf
---

# {display_name} — GGUF

GGUF quants of [`{fp16_repo}`](https://huggingface.co/{fp16_repo}) for
llama.cpp / Ollama / LM Studio / local CPU+GPU inference.

See the [FP16 reference card](https://huggingface.co/{fp16_repo}) for the
full training recipe, eval numbers, and known limitations.

## Files

| File | Quant | Use case |
|---|---|---|
{files_table}

Built by llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`.

## Inference

### llama.cpp

```bash
./llama-cli -m {primary_quant_filename} \\
    -p "{example_prompt}" \\
    -n 1024 --temp 0.7 --top-p 0.9
```

### Ollama

```bash
echo "FROM ./{primary_quant_filename}" > Modelfile
ollama create {model_shortname}:7b -f Modelfile
ollama run {model_shortname}:7b "{example_prompt}"
```

---

Built by [Bottensor](https://bottensor.xyz) — a Falcon Hash company.
"""


CARD_LORA = """---
license: apache-2.0
base_model: {base_model}
tags:
  - peft
  - lora
{extra_tags_yaml}
library_name: peft
---

# {display_name} — LoRA adapter

LoRA adapter for {display_name}. Apply on top of [`{base_model}`](https://huggingface.co/{base_model}),
or use the merged FP16 reference at [`{fp16_repo}`](https://huggingface.co/{fp16_repo}).

## Adapter config

- rank = {lora_rank}, alpha = {lora_alpha}, dropout = {lora_dropout}
- target modules: `{target_modules}`
- {method_summary}

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "{base_model}", torch_dtype=torch.float16, device_map="auto",
)
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("{base_model}")

# Optional: bake the adapter in for faster inference
# model = model.merge_and_unload()
```

---

Built by [Bottensor](https://bottensor.xyz) — a Falcon Hash company.
"""


CARDS: Mapping[str, str] = {
    "fp16": CARD_FP16,
    "gptq": CARD_GPTQ,
    "gguf": CARD_GGUF,
    "lora": CARD_LORA,
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _yaml_list(items: Iterable[str], indent: str = "  ") -> str:
    """Render a list of strings as YAML list items at the given indent."""
    return "\n".join(f"{indent}- {x}" for x in items)


def render_card(kind: str, **fields) -> str:
    """
    Render a card from one of the named templates. ``kind`` ∈
    ``{"fp16", "gptq", "gguf", "lora"}``.

    Common ``fields`` (per-kind extras documented above):
      - ``display_name``    — e.g. "NPC Agentic 7B (v2)"
      - ``base_model``      — HF repo of the base, e.g. "Qwen/Qwen2.5-7B-Instruct"
      - ``repo``            — destination HF repo, e.g. "ramankrishna10/npc-agentic-7b"
      - ``tags``            — iterable of extra tags (rendered as YAML list)
      - ``example_prompt``  — short prompt for the inference snippet
    """
    if kind not in CARDS:
        raise ValueError(f"unknown card kind {kind!r}; expected one of {list(CARDS)}")

    tags = list(fields.pop("tags", []))
    fields.setdefault("tags_yaml", _yaml_list(tags) if tags else "  - bottensor")
    fields.setdefault("extra_tags_yaml", _yaml_list(tags) if tags else "")
    fields.setdefault("extra_sections", "")
    fields.setdefault("example_prompt",
                      "Explain photosynthesis step by step.")
    return CARDS[kind].format(**fields)


def push_folder(
    repo: str,
    local_dir: str | Path,
    card_text: str,
    *,
    private: bool = False,
    commit_message: str | None = None,
    ignore_patterns: Sequence[str] = DEFAULT_IGNORE,
    token: str | None = None,
) -> str:
    """
    Create the repo if missing, write ``README.md`` into ``local_dir``,
    upload everything via ``upload_folder``, and return the HF URL.

    Idempotent: pushing the same folder twice produces a new commit on
    top of the existing repo. The ``DEFAULT_IGNORE`` list keeps trainer
    state out.
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"no such folder: {local_dir}")

    api = HfApi(token=token)
    create_repo(
        repo, private=private, exist_ok=True,
        repo_type="model", token=token,
    )

    # Write the card so upload_folder sweeps it in
    (local_dir / "README.md").write_text(card_text, encoding="utf-8")

    if commit_message is None:
        commit_message = (
            f"Upload {repo.split('/')[-1]} "
            f"({datetime.now(timezone.utc).date().isoformat()})"
        )

    url = upload_folder(
        folder_path=str(local_dir),
        repo_id=repo,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=list(ignore_patterns),
        token=token,
    )
    return url


def stage_lora_adapter(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    keep: Iterable[str] = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ),
) -> Path:
    """
    Copy ONLY the adapter + tokenizer files from ``src_dir`` to a clean
    ``dst_dir``, ready for ``push_folder``. This avoids accidentally
    shipping optimizer state or checkpoint scaffolding to the LoRA repo.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    keep_set = set(keep)
    copied: list[str] = []
    for f in src.iterdir():
        if f.name in keep_set and f.is_file():
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)
    if not copied:
        raise RuntimeError(f"no adapter files matched in {src} (looked for {keep_set})")
    return dst


def folder_size_gb(path: str | Path) -> float:
    """Sum of file sizes under ``path``, in GB."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9
