# shared/

Cross-model helpers. Source-of-truth for patterns that show up in more
than one training run, so the v1/v2/future scripts don't drift.

## utils/sft_masking.py

Char-offset label masking for chat-template SFT. Solves two real
problems we hit during NPC Agentic training:

1. **TRL 0.24 dropped `DataCollatorForCompletionOnlyLM`**, so the
   batched-prompt-mask path no longer exists out of the box.
2. **Qwen2.5's chat template has no `{% generation %}` markers**, so
   `SFTConfig(assistant_only_loss=True)` silently produces
   `sum(mask) == 0` and trains on nothing.

The helper renders the chat template, locates each assistant turn's
body via its `<|im_start|>assistant\n` header and `<|im_end|>` closer,
tokenizes with `return_offsets_mapping=True`, then unmasks only those
tokens whose character span intersects an assistant-body region.

```python
from shared.utils import preprocess_example, sanity_check_mask_share

train_ds = train_ds.map(
    lambda ex: preprocess_example(tokenizer, ex, max_len=8192),
    remove_columns=train_ds.column_names,
    num_proc=4,
)

share = sanity_check_mask_share(list(train_ds.select(range(8))))
assert 0.15 < share < 0.60, f"mask share looks wrong: {share:.2%}"
```

For Llama-3-Instruct chat templates, override the markers:

```python
preprocess_example(
    tokenizer, ex, max_len=8192,
    header="<|start_header_id|>assistant<|end_header_id|>\n\n",
    end_marker="<|eot_id|>",
)
```

The v1 and v2 scripts in `training/npc-agentic-7b-v*/` carry an inline
copy of this helper (kept self-contained on purpose — they're a
historical record of what actually ran). New runs should import from
`shared.utils` instead.

## Planned additions

- `utils/hf_push.py` — `push_to_hub` wrapper + auto-generated model card
- `utils/gptq.py`    — `llm-compressor` 0.10 wrapper (uses `scheme="W4A16"`
                        preset; v1/v2 hit kwargs incompatibility without it)
- `utils/gguf.py`    — `llama.cpp` convert + quantize wrapper (Q4_K_M /
                        Q5_K_M / Q8_0)
- `utils/eval/`      — shared GSM8K / MATH / identity harnesses
