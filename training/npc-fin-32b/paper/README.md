# Paper — NPC Fin 32B

> **Status: draft complete (7 pages). All numbers locked from the published HF model card + verified config-vs-runtime drift discussion. Ready for Zenodo upload.**

LaTeX source and compiled PDF for the third paper in the NPC family.

## Files

| File | What |
|---|---|
| `npc-fin-32b.tex` | LaTeX source (~430 lines, mirrors the family template) |
| `npc-fin-32b.pdf` | Compiled draft, 7 pages, ~92 KB |

## What's filled

All headline numbers verified from the live HF model card + user-confirmed run telemetry:

- **Base model:** Qwen2.5-32B-Instruct
- **Adapter:** QLoRA r=64, α=128, dropout 0.05, all 7 modules
- **Training data:** 32,496 SFT examples / 59.7M tokens / 5 domain tags
- **Synthetic labeling:** Qwen2.5-72B-Instruct via HF Inference API; MinHash dedup + automated quality filter
- **Optimizer:** AdamW 8-bit, cosine LR 2e-4, warmup 0.05, weight decay 0.01
- **Distributed:** DeepSpeed ZeRO-3 + full CPU offload on 12× H100 SXM5 80 GB (RunPod)
- **Wall clock:** ≈72 hours
- **Total compute:** ≈864 H100-hours
- **Eval:** CryptoQA 93.6 % (500-question internal benchmark)

## Headline new finding (paper's signature contribution)

**Config drift between planned and realized batch size.** The training YAML inherited `micro_batch_size: 4` and `gradient_accumulation_steps: 8` (effective batch 32) from an earlier single-GPU plan; the realized run on 12 H100s scaled the effective batch to ≈384 silently (4 × 12 × 8). The peak LR of 2e-4 was tuned for 32, not for 384. Standard scaling rules suggest 7e-4 would have been closer; 2e-4 worked anyway, attributed to ZeRO-3 + CPU-offload averaging behavior plus the small LoRA parameter surface (~0.8 % of base).

This is documented honestly in §5.3 and Limitations §7.2 — the same class of bug as the bootstrap-iteration claim corrected in the companion Cheap PRMs paper.

## Compile

```bash
cd training/npc-fin-32b/paper
pdflatex -interaction=nonstopmode npc-fin-32b.tex
pdflatex -interaction=nonstopmode npc-fin-32b.tex   # 2nd pass for citations
# → npc-fin-32b.pdf, 7 pages, 92 KB
```

Requires `pdflatex` + `txfonts` + `hyperref` + `booktabs` (standard TeX Live).

## Honest gaps (recorded in §7 Limitations)

| Gap | Status |
|---|---|
| WandB run-state / loss curve | Not preserved; documented as a real reproducibility gap |
| Per-step throughput | Not preserved |
| Final train/eval loss | Not preserved |
| LR sweep at the realized batch | Not run; recipe under-tunes LR |
| Head-to-head vs base Qwen2.5-32B | Not run |
| Public-benchmark eval (MMLU, MATH, GSM8K, BBH) | Not run; out-of-domain |
| CryptoQA appendix sample | Internal-only; not redistributed |

## After Zenodo upload

Three things need updating once the DOI mints:

1. **HF model card corrections** (separate from paper):
   - `Hardware: A40 48GB (RunPod)` → `12× H100 SXM5 80GB (RunPod)`
   - `Batch Size: 4 / Gradient Accumulation: 8 / effective batch 32` → realized eff batch ≈384 with explanation
   - `Precision: float16` → `bf16`
2. **Citation block** updated to use minted DOI (replaces placeholder)
3. **Top-level repo README + model README** get DOI badges
