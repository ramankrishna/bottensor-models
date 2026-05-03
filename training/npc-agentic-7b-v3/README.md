# NPC Agentic 7B — v3

> **Status: shipped.** Public on Hugging Face under `ramankrishna10/npc-agentic-7b-v3*`. Recipe + benchmarks paper on Zenodo: [10.5281/zenodo.19954103](https://doi.org/10.5281/zenodo.19954103). Differs from v2 in exactly one place: the chat-template SFT label-masking now includes the closing `<|im_end|>` token in the trained loss. v1 and v2 both excluded it, which produced models that could not terminate cleanly at inference time. v3 isolates the fix as a clean ablation.

## TL;DR root cause (the bug v3 fixes)

| | v1 / v2 | **v3** |
|---|---|---|
| Span passed to label mask | `[body_start, marker_pos)` | `[body_start, marker_pos + len(end_marker))` |
| `<|im_end|>` in loss | ❌ never | ✅ always |
| Model learns to emit EOS | no | yes |
| Inference behavior on a clean prompt | answers correctly, then drifts into hallucinated `Human:`/`Assistant:` continuations or `</details>` repetition loops | terminates after the answer |

The bug was **one missing `+ len(end_marker)`** in
`shared/utils/sft.py :: build_assistant_spans()`. v3's training script
also adds an explicit pre-train probe (`verify_eos_in_loss`) that fails
fast if the bug ever regresses.

## Verification probe (runs before every v3 training kickoff)

```python
from shared.utils.sft import verify_eos_in_loss

samples = [train_ds[i] for i in random.sample(range(len(train_ds)), 50)]
stats = verify_eos_in_loss(samples, eos_token_id=tokenizer.eos_token_id)
# → {'eos_total': 122, 'eos_unmasked': 50, 'samples': 50}
# correct: ~50 (every assistant turn closer); broken: 0
assert stats["eos_unmasked"] > 0, "v1/v2 bug regressed"
```

## v3 vs v2 — clean ablation

Everything except the masking is held constant so any quality delta is
attributable to the EOS fix:

| Field | v2 | **v3** |
|---|---|---|
| Base | `Qwen/Qwen2.5-7B-Instruct` | (same) |
| Method | QLoRA, 4-bit NF4 + bf16 LoRA | (same) |
| LoRA r/α/dropout | 64 / 128 / 0.05, 7 modules | (same) |
| Optimizer | AdamW 8-bit | (same) |
| LR / schedule / warmup / wd | 2e-4 / cosine / 0.03 / 0.01 | (same) |
| Effective batch | 16 | (same) |
| Max seq | 8192 | (same) |
| Epochs | 1 | (same) |
| Identity examples + cohorts | 3000, 3 cohorts (40/30/30) | (same) |
| Held-out identity eval prompts | 15 | (same) |
| Hardware | RunPod A40 48 GB | (same) |
| **EOS in loss** | **❌** | **✅** |

## Other v3-vs-v2 script hardening (paper-relevant lessons from v2)

| Script | v2 issue | v3 fix |
|---|---|---|
| `02_train.py` | EOS-mask bug + no probe | mask fix + `verify_eos_in_loss` fail-fast probe |
| `05_quantize.py` | Silent fallback to broken auto-gptq, no result-validation, exited 0 with empty `quantized/` | Re-raises non-import errors; asserts `*.safetensors` present in `quantized/` before claiming success |
| `07_gguf.py` | Crashed on re-run when `merged/` had been cleaned (rigid existence check) | Skips merged check when f16 GGUF already exists at full size — supports quantize-only re-runs |
| `v3_post_train.sh` (was `v2_post_train.sh`) | `\| tee -a $LOG` masked exit codes; `set -e` didn't help | Wraps each step in a `run_step` helper that uses `${PIPESTATUS[0]}` and halts on non-zero exit |

## Layout

```
training/npc-agentic-7b-v3/
├── README.md                       # this file
├── requirements.txt                # same as v2
├── configs/
│   └── config.py                   # MODEL_SHORTNAME=npc-agentic-7b-v3 + WANDB run name updated
└── scripts/
    ├── 01_prepare_data.py          # same as v2
    ├── 02_train.py                 # ⚠️ EOS-mask FIX + verify_eos_in_loss probe added
    ├── 03_evaluate.py              # same as v2
    ├── 03b_gsm8k_rerun.py          # same as v2 (orchestrator now runs it as STEP 4b)
    ├── 03c_gsm8k_sample.py         # same as v2
    ├── 04_merge.py                 # same as v2
    ├── 05_quantize.py              # ⚠️ no silent-fallback; asserts non-empty quantized/
    ├── 06_push.py                  # same as v2 (writes to v3-suffixed HF repos)
    ├── 07_gguf.py                  # ⚠️ supports quantize-only re-runs (skips merged check when f16 exists)
    ├── v3_post_train.sh            # ⚠️ real per-step exit-code checks
    └── watchdog.sh                 # same as v2
```

## HF push targets (v3 suffix — separate from v2)

- `ramankrishna10/npc-agentic-7b-v3` (bf16 merged)
- `ramankrishna10/npc-agentic-7b-v3-gptq-4bit` (W4A16)
- `ramankrishna10/npc-agentic-7b-v3-lora` (adapter)
- `ramankrishna10/npc-agentic-7b-v3-gguf` (Q4_K_M / Q5_K_M / Q8_0)

## Measured outcomes (head-to-head vs.~base Qwen2.5-7B-Instruct)

Both models served via vLLM 0.6.3, auto-gptq W4A16 quantization (group_size 128, sym False, desc_act True, 512 calibration samples from `mit-han-lab/pile-val-backup`), greedy decoding (`temperature=0`). Multi-turn ran at `max_model_len=16384` after an 8K initial run produced ~95% HTTP-400 errors due to context overflow.

### BFCL v4 (Berkeley Function Calling Leaderboard)

| Subset | Base | v3 | Δ |
|---|---:|---:|---:|
| `live` (single-turn) | 71.1% | 69.0% | −2.1 pp |
| `multi_turn` (avg) | 11.6% | 3.4% | −8.2 pp |
| ⤷ `multi_turn_base` | 18.0% | 8.0% | −10.0 pp |
| ⤷ `multi_turn_miss_func` | 13.5% | 0.5% | −13.0 pp |
| ⤷ `multi_turn_miss_param` | 9.0% | 2.5% | −6.5 pp |
| ⤷ `multi_turn_long_context` | 6.0% | 2.5% | −3.5 pp |
| `relevance` | 93.8% | 87.5% | −6.2 pp |
| `irrelevance` | **69.9%** | **79.5%** | **+9.6 pp** |

**Headline finding.** v3 regresses on function calling at a 4096 max_new_tokens budget: the `<think>`-style reasoning consumes the budget before the model emits the `<tool_call>` block. Practitioners deploying v3 for tool-use should set `max_new_tokens >= 8192` or use `</think>` as a stop sequence. Irrelevance detection improves (+9.6 pp).

### GSM8K-100

| Model | Score |
|---|---:|
| Qwen2.5-7B-Instruct (base) | **61%** |
| NPC Agentic 7B v3 | 6% |

The identity-cohort training pushes v3's register away from compact arithmetic answers; do not use this model for math-primary use cases.

### Throughput (BFCL workload, single-stream)

| Backend | Hardware | Tokens/sec |
|---|---|---:|
| llama.cpp Q4_K_M GGUF | Apple Silicon | ~25 |
| vLLM 0.6.3 GPTQ W4A16 | A40 | ~76 |

### Verification probe at training kickoff

`verify_eos_in_loss` returned `eos_total=122, eos_unmasked=50` across 50 sampled rows — correct shape (one unmasked EOS per assistant turn, system/user EOS correctly masked).

## Benchmark harness

The full BFCL head-to-head harness (vLLM serve config, model registry patch, scoring extractor, comparison-table builder) is at [github.com/ramankrishna/bench-npc-agentic-v3](https://github.com/ramankrishna/bench-npc-agentic-v3).

## Citation

```bibtex
@misc{bachu2026npcagentic7b,
  title  = {NPC Agentic 7B: A Single-GPU QLoRA Recipe for a
            Laptop-Scale Conversational Model},
  author = {Bachu, Rama Krishna},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.19954103},
  url    = {https://doi.org/10.5281/zenodo.19954103},
  note   = {Preprint},
}
```

## Author

**Rama Krishna Bachu** (`dude.npc`) — founder, Falcon Hash → Bottensor.
[ORCID 0009-0000-1298-0681](https://orcid.org/0009-0000-1298-0681) ·
[ramakrishna.bachu@bottensor.xyz](mailto:ramakrishna.bachu@bottensor.xyz)
