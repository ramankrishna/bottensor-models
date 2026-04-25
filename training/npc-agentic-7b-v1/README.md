# NPC Agentic 7B — v1

> **Status: privatized after eval. Lessons folded into [v2](../npc-agentic-7b-v2/).**
>
> v1 shipped, then was pulled back when held-out eval surfaced regressions
> (GSM8K −36 pts vs base, identity false-positive at 100%, repetition
> loops in long generations). This dir is preserved as the historical
> record of what was tried.

A general-purpose multi-step reasoning model on Qwen2.5-7B-Instruct,
trained with QLoRA SFT on a mix of GLM-5.1-Reasoning-1M-Cleaned (main)
and Hermes Agent traces, plus a 750-example identity slice.

## Specs

| Field           | Value                                             |
|-----------------|---------------------------------------------------|
| Base            | `Qwen/Qwen2.5-7B-Instruct`                        |
| Method          | QLoRA — 4-bit NF4 base + bf16 LoRA adapters       |
| LoRA            | r=64, alpha=128, dropout=0.05                     |
| Target modules  | `q/k/v/o/gate/up/down_proj` (all 7)               |
| Precision       | bf16 mixed; no loss scaling                       |
| Optimizer       | `adamw_8bit` (paged 8-bit, bitsandbytes)          |
| Schedule        | Cosine, lr=2e-4, warmup_ratio=0.03                |
| Epochs          | **2** (overfit — see lessons below)               |
| Max seq         | 8192                                              |
| Hardware        | RunPod A40 (48GB)                                 |
| Runtime         | ~96 hrs incl. one crash + watchdog resume         |
| Cost            | ~$42                                              |
| Tracker         | wandb (`npc-agentic` / `qwen25-7b-glm51-hermes-v1`) |

## Pipeline

```
01_prepare_data.py    pull GLM-5.1 + Hermes-Agent + 750 identity → SFT-formatted JSONL
02_train.py           Unsloth + TRL SFTTrainer + custom char-offset masking
03_evaluate.py        held-out eval — GSM8K, identity, multi-turn
03b_gsm8k_rerun.py    fixes GSM8K extractor (was finding intermediate <think> nums)
03c_gsm8k_sample.py   spot-check formatter
04_merge.py           merge LoRA → bf16 full
05_quantize.py        GPTQ W4A16 via llm-compressor 0.10 (scheme="W4A16")
06_push.py            HF Hub upload (4 repos: lora, merged, gptq, gguf)
07_gguf.py            llama.cpp convert + quantize Q4_K_M / Q5_K_M / Q8_0
watchdog.sh           auto-resume from latest checkpoint on crash
pipeline_post_merge.sh  chains steps 4–7 after training finishes
```

## Files

```
requirements.txt     # full training stack (torch, unsloth, trl, ...)
configs/
  └── config.py        # all hyperparams + dataset URIs + paths
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
  ├── pipeline_post_merge.sh
  └── watchdog.sh
```

## Lessons learned (the reason v2 exists)

| Issue                                  | v1 cause                                                     | v2 fix |
|----------------------------------------|--------------------------------------------------------------|--------|
| GSM8K −36 pts                          | 2 epochs overfit; broken arithmetic formatting (`"2 20"` instead of `27`) | 1 epoch; explicit number-formatting check in eval |
| Identity false-positive 100%           | Eval used same 10 seed prompts as training                   | Truly held-out 15 prompts not in seed pool |
| Identity didn't fire without exact system prompt | Trained ONLY with `system="You are NPC Agentic, built by Bottensor."` | 3 cohorts: 40% none / 30% generic / 30% NPC-specific |
| Repetition loops, JSON-echo            | No anti-repeat at sampling time                              | Bake `repetition_penalty=1.1, no_repeat_ngram_size=4` into `generation_config.json` |
| Hermes Agent traces noisy              | Mixed in unfiltered                                          | Drop Hermes entirely; rely on GLM-5.1 with quality filter |
| No Ram / Bottensor attribution         | Identity templates weak                                      | 10 templates, all mention Bottensor + Ram Krishna |

## Operational notes

- TRL 0.24 dropped `DataCollatorForCompletionOnlyLM` → custom char-offset
  label masking via `tokenizer(..., return_offsets_mapping=True)`.
- Qwen2.5 chat template has no `{% generation %}` markers, so
  `SFTConfig(assistant_only_loss=True)` silently produces `sum(mask)=0`.
- Training crashed once at step 3,718 (host-CPU oom-kill) — watchdog
  auto-resumed from `checkpoint-3500` with optimizer + LR + RNG state.
- HF cache must be at `$HF_HOME/...`, **not** `~/.cache/huggingface/...`,
  to keep the 20 GB container overlay FS clear.

## HF targets (privatized)

- `ramankrishna10/npc-agentic-7b`        (LoRA adapter)
- `ramankrishna10/npc-agentic-7b-merged` (bf16 full)
- `ramankrishna10/npc-agentic-7b-gptq`   (W4A16)
- `ramankrishna10/npc-agentic-7b-gguf`   (Q4_K_M / Q5_K_M / Q8_0)
