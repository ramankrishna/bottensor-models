# NPC Fast — Trainer

Continual pre-training pipeline for **NPC Fast**, a 1.7B agentic router
model with a 128K context window. Part of the NPC Model Family by
Bottensor (a Falcon Hash company).

- **Base:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Training:** full-weight continual pre-training (no LoRA, no QLoRA, no PEFT)
- **Context:** 128K via YaRN RoPE (factor 16.0, original 8192)
- **Hardware:** single H200 SXM (141GB HBM3e)
- **Runtime:** ~8–10 hours for 10K steps
- **Role:** fast router that decides `self` vs escalate to NPC Fin 32B

## Layout

```
npc-fast-trainer/
├── main.py                   # CLI (train / eval / export / status / preview)
├── train.py                  # Trainer loop (HF Trainer + curriculum callback)
├── config.py                 # Hyperparameters, paths, curriculum schedule
├── datasets.json             # Dataset registry (HF repos, weights, format)
├── data/
│   ├── adapters/             # sharegpt / openai format normalizers
│   ├── loader.py             # HF dataset download → unified messages
│   ├── mixer.py              # Weighted sampling + interleave + 95/5 split
│   ├── preprocessing.py      # Chat template + tokenize + sequence packing
│   ├── curriculum.py         # CurriculumCallback — 5-stage context growth
│   └── dedup.py              # Hash-based deduplication
├── model/
│   ├── setup.py              # bf16 + flash-attn-2 + YaRN + full-weight train
│   ├── rope_scaling.py       # YaRN config
│   └── save.py               # Full-weight checkpoint helper
├── eval/
│   ├── benchmarks.py         # BFCL v2 + IFEval + custom agentic
│   ├── context_eval.py       # Needle-in-haystack 16K/32K/64K/128K × 5 depths
│   ├── router_eval.py        # Self vs npc_fin routing accuracy (200 queries)
│   └── perplexity.py         # Val perplexity, per-tag, short vs long
├── export/
│   ├── quantize.py           # GPTQ W4A16 via auto-gptq + 256-sample calibration
│   ├── gguf.py               # llama.cpp convert + Q4_K_M + Q8_0
│   └── push_hf.py            # HF Hub upload with generated model card
├── scripts/
│   ├── run_train.sh          # End-to-end training launcher
│   ├── run_eval.sh           # Eval suite launcher
│   └── run_export.sh         # Quantize + GGUF + HF push
├── requirements.txt
├── datasets.json
└── .env.example
```

## Curriculum

| Stage | Steps        | max_seq_length | micro_batch | grad_accum | effective |
|-------|--------------|----------------|-------------|------------|-----------|
| 1     | 0–2 000      | 4 096          | 8           | 4          | 32        |
| 2     | 2 000–4 000  | 16 384         | 4           | 8          | 32        |
| 3     | 4 000–6 000  | 32 768         | 2           | 16         | 32        |
| 4     | 6 000–8 000  | 65 536         | 2           | 16         | 32        |
| 5     | 8 000–10 000 | 131 072        | 1           | 32         | 32        |

The `CurriculumCallback` monitors `state.global_step` and flips
`max_seq_length`, `per_device_train_batch_size`, and
`gradient_accumulation_steps` at each stage boundary. The packed dataset
is re-packed on-demand so stage-4/5 receive genuinely long sequences
built from concatenated shorter examples.

## Datasets

Configured in `datasets.json`. Current mix:

| Dataset                                            | Format   | Weight |
|----------------------------------------------------|----------|--------|
| `ramankrishna10/npc-fast-datagen-batch-001`        | sharegpt | 1.0    |
| `lambda/hermes-agent-reasoning-traces::glm-5.1`    | sharegpt | 1.0    |
| `lambda/hermes-agent-reasoning-traces::kimi`       | sharegpt | 1.0    |
| `Roman1111111/claude-opus-4.6-10000x`              | openai   | 0.8    |

Adapters normalize both formats into a unified
`[{"role": ..., "content": ...}]` list. The mixer caps each source so
the final distribution matches the requested weights; expect a log line
like `Dataset composition: 60% datagen, 24% hermes, 16% claude-opus`.

## Setup

```bash
cd npc-fast-trainer
cp .env.example .env              # fill HF_TOKEN + WANDB_API_KEY
pip install -r requirements.txt

# Flash-attn sometimes needs a bespoke install on some CUDA builds:
#   pip install flash-attn --no-build-isolation
```

## Commands

```bash
# Train — full pipeline
bash scripts/run_train.sh
# or equivalently
python main.py train

# Resume from a checkpoint
python main.py train --resume_from_checkpoint output/checkpoints/checkpoint-5000

# Eval
bash scripts/run_eval.sh
# or with a custom checkpoint
python main.py eval --checkpoint output/checkpoints/final/

# Export (GPTQ + GGUF + HF push)
bash scripts/run_export.sh
# or granular:
python main.py export --checkpoint output/checkpoints/final/ --artifacts gptq gguf bf16 --push

# Inspect state
python main.py status
python main.py preview --k 5
```

## Outputs

- `output/checkpoints/` — HuggingFace-format shards every 500 steps
- `output/checkpoints/final/` — final full-weight model
- `output/npc-fast-1.7b-gptq/` — W4A16 GPTQ export
- `output/npc-fast-1.7b-gguf/` — f16 + Q4_K_M + Q8_0 GGUF for llama.cpp
- `output/eval_results.json` — BFCL / IFEval / agentic / needle / router / ppl
- `output/needle_heatmap.png` — context × depth heatmap

## Identity

- **Model:** NPC Fast
- **Built by:** Bottensor (a Falcon Hash company)
- **Creator:** dude.npc
- **Role:** fast agentic router with 128K context
- **Partner:** NPC Fin 32B (heavy reasoning)

## HuggingFace targets

- `ramankrishna10/npc-fast-1.7b` — bf16 full model
- `ramankrishna10/npc-fast-1.7b-gptq` — 4-bit GPTQ
- `ramankrishna10/npc-fast-1.7b-gguf` — GGUF (Q4_K_M + Q8_0)
