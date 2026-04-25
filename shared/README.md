# shared/

Cross-model helpers. Source-of-truth for patterns that show up in more
than one training run, so v1 / v2 / future scripts don't drift.

## Layout

```
shared/utils/
├── __init__.py          # re-exports all public helpers
├── sft.py               # char-offset label masking for chat-template SFT
├── hf_push.py           # HF Hub upload + model-card generator
├── gptq.py              # llm-compressor 0.10 W4A16 wrapper
├── gguf.py              # llama.cpp build / convert / quantize wrapper
└── eval/
    ├── __init__.py
    ├── gsm8k.py         # robust extractor + accuracy harness
    ├── identity.py      # multi-keyword identity scoring
    └── math_bench.py    # MATH \boxed{} extraction + equivalence
```

## utils/sft.py

Char-offset label masking for chat-template SFT. Solves the two
problems we hit during NPC Agentic training:

1. TRL 0.24 dropped `DataCollatorForCompletionOnlyLM`.
2. Qwen2.5 (and Llama-3) chat templates have no `{% generation %}`
   markers, so `SFTConfig(assistant_only_loss=True)` silently produces
   `sum(mask) == 0` and trains on nothing.

```python
from shared.utils import preprocess_example, sanity_check_mask_share

train_ds = train_ds.map(
    lambda ex: preprocess_example(tokenizer, ex, max_len=8192),
    remove_columns=train_ds.column_names, num_proc=4,
)
share = sanity_check_mask_share(list(train_ds.select(range(8))))
assert 0.15 < share < 0.60, f"mask share looks wrong: {share:.2%}"
```

For Llama-3-Instruct, override the markers:

```python
preprocess_example(
    tokenizer, ex, max_len=8192,
    header="<|start_header_id|>assistant<|end_header_id|>\n\n",
    end_marker="<|eot_id|>",
)
```

## utils/hf_push.py

`push_folder()` + `render_card()` + `stage_lora_adapter()`.

```python
from shared.utils import push_folder, render_card, stage_lora_adapter

# 1. FP16 merged
fp16_card = render_card(
    "fp16",
    display_name="NPC Agentic 7B (v2)",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    repo="ramankrishna10/npc-agentic-7b",
    summary="A 7B reasoning model from the Bottensor NPC family.",
    method="QLoRA SFT (r=64, α=128), merged to FP16",
    hardware="single A40 (48 GB)",
    trainable_params="161.5M",
    final_eval_loss="0.68",
    dataset_summary="GLM-5.1 reasoning + 3000 identity replay",
    tags=["reasoning", "agent", "bottensor", "npc"],
)
push_folder("ramankrishna10/npc-agentic-7b", merged_dir, fp16_card)

# 2. LoRA adapter (stage to a clean folder first)
adapter_dir = stage_lora_adapter(checkpoint_dir, "/tmp/lora_upload")
push_folder(
    "ramankrishna10/npc-agentic-7b-lora",
    adapter_dir,
    render_card("lora", display_name="NPC Agentic 7B (v2)", ...),
)
```

`DEFAULT_IGNORE` keeps optimizer/scheduler/RNG state out automatically.

## utils/gptq.py

llm-compressor 0.10 wrapper. Bakes in two known-bad-defaults fixes:

- `GPTQModifier(scheme="W4A16")` (don't pass `group_size`, etc., as
  kwargs — 0.10 dropped them).
- Calibration data must be `Dataset.from_dict({"text": [...]})` and
  `oneshot()` needs `text_column="text"`.

```python
from shared.utils import quantize_w4a16, gptq_smoke_test

quantize_w4a16(
    merged_dir="/workspace/.../merged",
    output_dir="/workspace/.../quantized",
    sft_jsonl_path="/workspace/.../data/processed/train.jsonl",
    n_calib=512,
)
assert gptq_smoke_test("/workspace/.../quantized")
```

## utils/gguf.py

llama.cpp build / convert / quantize wrapper. Handles:

- Cloning + CPU-only cmake build (most pods don't have `nvcc`).
- Both `convert_hf_to_gguf.py` and the older `convert-hf-to-gguf.py`.
- Idempotent skip if outputs already exist.
- Optional CPU smoke test (gated by `GGUF_SKIP_SMOKE=1` env var).

```python
from shared.utils import build_quants

outputs = build_quants(
    merged_dir="/workspace/.../merged",
    output_dir="/workspace/.../gguf",
    model_shortname="npc-agentic-7b",
    llama_cpp_dir="/workspace/llama.cpp",
    quants=("Q4_K_M", "Q5_K_M", "Q8_0"),
)
# outputs == {"Q4_K_M": Path(...), "Q5_K_M": Path(...), "Q8_0": Path(...)}
```

## utils/eval/

Three harnesses sharing a "load → generate-fn closure → score → roll
up" shape.

### gsm8k

```python
from shared.utils.eval import load_gsm8k_samples, eval_gsm8k

samples = load_gsm8k_samples(n=100, seed=42)

def gen(user_text: str) -> str:
    # Plug in your preferred backend (HF, vLLM, API)
    return run_my_model(user_text, max_new_tokens=1024, do_sample=False)

acc, records = eval_gsm8k(samples, gen, label="v2", dump_path="gsm8k.md")
```

`extract_gsm_answer` is the v2 fix that recovered ~30 acc points on v1
when we re-ran with the same weights. It tries `####`, `\boxed{}`,
post-`</think>` "answer is N" patterns, and last-number fallback in
order.

### identity

```python
from shared.utils.eval import score_identity, DEFAULT_IDENTITY_KEYWORDS

scores = [
    score_identity(model_response_for(prompt))
    for prompt in held_out_identity_prompts
]
agg = aggregate_identity(scores)
# {"full_match_rate": 0.93, "any_match_rate": 1.0, "mean_slot_recall": 0.97}
```

The default keyword bank covers `name` (NPC Agentic), `creator` (Ram
Krishna / dude.npc), and `parent` (Bottensor / Falcon Hash). Override
for non-agentic models.

### math_bench

```python
from shared.utils.eval import extract_boxed_answer, is_math_correct

pred = extract_boxed_answer(model_output)
ok = is_math_correct(pred, gold_answer)
```

Light normalization — strips `$...$`, `\left/\right`, normalizes
`\dfrac`/`\tfrac` to `\frac`. For full MATH equivalence (mixed
numbers, units), promote to the `minerva_math` evaluator.

## Important: existing scripts in `training/` carry inline copies

The v1 and v2 scripts in `training/npc-agentic-7b-v*/` keep their
inline copies of these helpers on purpose — they're a historical
record of what actually ran on the pod. New training runs should
import from `shared.utils` instead.
