# NPC Agentic 7B — v2

> **Status: training (Apr 2026).** Currently ~30 % through step 1586/5352
> on RunPod A40. Auto-chained orchestrator runs eval → merge → GPTQ →
> GGUF → HF push when training finishes.

A general-purpose multi-step reasoning model on Qwen2.5-7B-Instruct, with
every [v1 lesson](../npc-agentic-7b-v1/README.md#lessons-learned-the-reason-v2-exists)
baked in.

## What changed vs v1

| Knob                          | v1                                | v2                                       |
|-------------------------------|-----------------------------------|------------------------------------------|
| Epochs                        | 2 (overfit)                       | **1**                                    |
| Identity examples             | 750                               | **3000** (4×)                            |
| Identity system-prompt cohort | 100% NPC-specific                 | **40% none / 30% generic / 30% NPC**     |
| Identity templates            | 4                                 | **10** (Bottensor + Ram Krishna in 100%) |
| Hermes Agent traces           | Included                          | **Dropped**                              |
| Long-think filter             | None                              | `GLM_MAX_TRACE_TOKENS=6000`, `GLM_MIN_RESPONSE_CHARS=200` |
| Sampling defaults             | None baked in                     | `repetition_penalty=1.1`, `no_repeat_ngram_size=4` in `generation_config.json` |
| Identity eval prompts         | Same as seeds (false-positive)    | **Truly held-out 15 prompts**            |

## Specs

| Field         | Value                                          |
|---------------|------------------------------------------------|
| Base          | `Qwen/Qwen2.5-7B-Instruct`                     |
| Method        | QLoRA — 4-bit NF4 base + bf16 LoRA             |
| LoRA          | r=64, alpha=128, dropout=0.05                  |
| Target modules| `q/k/v/o/gate/up/down_proj` (all 7)            |
| Precision     | bf16                                           |
| Optimizer     | `adamw_8bit`                                   |
| Schedule      | Cosine, lr=2e-4, warmup_ratio=0.03             |
| Epochs        | 1                                              |
| Max seq       | 8192                                           |
| Hardware      | RunPod A40 (48 GB)                             |
| Eval cadence  | Every 250 steps                                |
| Tracker       | wandb                                          |

## Pipeline

Same step layout as v1. The new `v2_post_train.sh` orchestrator chains
the post-train flow with explicit per-step exit-code checks (v1's
`set -e | tee` pattern silently swallowed failures).

```
01_prepare_data.py
02_train.py                 # custom char-offset masking, watchdog resume
03_evaluate.py              # held-out eval suite
03b_gsm8k_rerun.py          # GSM8K extractor fixes from v1
03c_gsm8k_sample.py
04_merge.py                 # LoRA → bf16
05_quantize.py              # GPTQ W4A16 via llm-compressor 0.10
06_push.py                  # 4 HF repos
07_gguf.py                  # llama.cpp convert + Q4_K_M / Q5_K_M / Q8_0
watchdog.sh                 # auto-resume on crash
v2_post_train.sh            # chains 03 → 07 with hard exit-code checks
```

## Files

```
configs/
  └── config.py
scripts/
  ├── 01_prepare_data.py
  ├── 02_train.py
  ├── 03_evaluate.py
  ├── 03b_gsm8k_rerun.py
  ├── 03c_gsm8k_sample.py
  ├── 04_merge.py
  ├── 05_quantize.py
  ├── 06_push.py
  ├── 07_gguf.py
  ├── v2_post_train.sh
  └── watchdog.sh
```

## Live progress (as of last sync)

- Step 1586 / 5352 (29.6 %)
- `eval_loss` tracking ~3 % below v1 at every checkpoint
- Watchdog clean, no resumes needed yet
- ETA ~1.7 days remaining

## Operational fixes shipped in v2

- `pgrep -f "A|B"` requires `-E` (or split into two pgreps) — caught the
  orchestrator's broken pgrep checks.
- `set -e` does **not** propagate through `| tee -a` — replaced with
  explicit exit-code checks after each pipeline step.
- llm-compressor 0.10 `GPTQModifier` rejects `group_size` / `desc_act` /
  `sym` kwargs → use `scheme="W4A16"` preset.
- llm-compressor calibration data must be `Dataset.from_dict({"text": [...]})`,
  not a list of dicts; pass `text_column="text"` to `oneshot()`.
- llama.cpp build skips `-DGGML_CUDA=ON` when no `nvcc` (CPU-only quantize
  is fine).
- `transformers` 5.5.0 emits string-quoted loss values
  (`'loss': '0.7491'`); cosmetic monitor regex updated.

## HF targets (will go public when v2 ships)

- `ramankrishna10/npc-agentic-7b`        (LoRA)
- `ramankrishna10/npc-agentic-7b-merged` (bf16 full)
- `ramankrishna10/npc-agentic-7b-gptq`   (W4A16)
- `ramankrishna10/npc-agentic-7b-gguf`   (Q4_K_M / Q5_K_M / Q8_0)
