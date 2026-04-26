# NPC Fin 32B

> **Status: shipped (HF since 2026-04-09, public). Paper draft ready in `paper/`.**

NPC Fin 32B is a domain-specialized financial-reasoning supervised
fine-tune of Qwen2.5-32B-Instruct, trained on 32,496 synthetically-labeled
examples covering crypto signal analysis, broad crypto knowledge,
multi-path logic-tree reasoning, equities/macro analysis, and cross-asset
correlation. It scores **93.6 %** on a 500-question internal financial-
reasoning benchmark (CryptoQA).

The paper documenting the recipe, the multi-GPU DeepSpeed ZeRO-3 setup,
and an honest config-vs-runtime drift retrospective lives in
[`paper/`](paper/).

**Live model:** [`ramankrishna10/npc-fin-32b-sft`](https://huggingface.co/ramankrishna10/npc-fin-32b-sft)

## What's in this directory

```
npc-fin-32b/
├── README.md                       # this file
├── configs/                        # (placeholder for the YAML / DeepSpeed config)
└── paper/
    ├── README.md                   # paper status + reproduction notes
    ├── npc-fin-32b.tex             # LaTeX source (~430 lines)
    └── npc-fin-32b.pdf             # 7-page compiled draft
```

The original training scripts and DeepSpeed config are in the founder's
private archive and have not been redistributed in this repo. The paper
documents the full recipe; a practitioner reproducing the work will need
to write a new training script following the spec in §4 and §5 of the
paper.

## Verified training recipe (from the published HF card + run telemetry)

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-32B-Instruct` |
| Method | QLoRA, 4-bit NF4 base + bf16 LoRA |
| LoRA | r=64, α=128, dropout=0.05, all 7 modules |
| Optimizer | AdamW 8-bit (paged), cosine LR, peak 2e-4, warmup 0.05, wd 0.01 |
| Max seq | 4,096 tokens |
| Epochs | 3 |
| Distributed | **DeepSpeed ZeRO-3 + full CPU offload** (optimizer + parameters) |
| Precision | bf16 throughout |
| Hardware | **12 × NVIDIA H100 SXM5 80 GB** (RunPod single multi-GPU node) |
| Wall clock | ≈72 hours (3 days) |
| Total compute | ≈864 H100-hours |
| Per-GPU micro-batch | 4 |
| Realized effective batch | ≈384 (4 × 12 × 8) |

> **Note on batch-size discrepancy:** the published HF model card lists a
> per-device batch of 4 and grad-accum of 8 with "effective batch 32".
> That number is from the original single-GPU plan; the realized 12-GPU
> run scaled the effective batch by world-size, giving ≈384. The paper
> documents this honestly in §5.3 ("Config drift: planned vs realized
> batch size") and discusses the LR-not-retuned implication in §7.2.

## Training data

- **32,496 SFT examples / 59.7M tokens** across 5 domain tags
- Tags: `crypto_signal`, `crypto_general`, `logic_tree`, `stocks_macro`, `cross_market`
- Synthetic labeling via Qwen2.5-72B-Instruct (HF Inference API)
- Source signals from production MongoDB (`btunified`)
- Filtering: MinHash near-dup (Jaccard 0.85) + automated 3-axis quality scoring

## Eval

| Benchmark | Score |
|---|---|
| CryptoQA (500 questions, internal) | **93.6 %** |

CryptoQA is internal and not redistributed. Per the paper §6.2, no
head-to-head was run against the base model or against public benchmarks
(MMLU / MATH / GSM8K / BBH).

## Honest gaps (per paper §7)

- WandB run-state / loss curve / per-step throughput **not preserved**
- LR was tuned for the planned eff-batch 32, not the realized ≈384; **no LR sweep ran** at the realized batch
- CryptoQA construction details and sample questions **not in this repo**
- **No** function-calling, tool-use, or identity training (this is the SFT base only)

## Companion model

The process reward model trained to verify this reasoner's step-level
output is at [`../npc-fin-prm-7b/`](../npc-fin-prm-7b/) — see the
[Cheap PRMs paper](../npc-fin-prm-7b/paper/) for the verifier recipe.

## Citation

The paper draft in `paper/` will mint a DOI on Zenodo. Until then:

```bibtex
@misc{bachu2026npcfin32b,
  title  = {NPC Fin 32B: A Domain-Specialized Financial Reasoning Model
            via Multi-GPU QLoRA},
  author = {Bachu, Rama Krishna},
  year   = {2026},
  note   = {Preprint, in preparation. Model:
            https://huggingface.co/ramankrishna10/npc-fin-32b-sft},
}
```

## Author

**Rama Krishna Bachu** (`dude.npc`) — founder, Falcon Hash → Bottensor.
[ORCID 0009-0000-1298-0681](https://orcid.org/0009-0000-1298-0681) ·
[ramakrishna.bachu@bottensor.xyz](mailto:ramakrishna.bachu@bottensor.xyz)
