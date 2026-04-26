# NPC Fin-PRM 7B

> **Status: shipped (on HF since 2026-04-09). Eval harness + analysis live here for the upcoming paper "Cheap PRMs: Multi-Dimensional Process Reward Modeling for Domain-Specialized Reasoning."**

A 7B Process Reward Model trained on a single H100 in 17.4 hours. Scores
DeFi/crypto reasoning steps on **four dimensions** — `factual_accuracy`,
`logical_validity`, `completeness`, `risk_awareness` — for use in
Best-of-N sampling, reasoning verification, and PRM-guided search.

**Live model:** [`ramankrishna10/npc-fin-prm-7b`](https://huggingface.co/ramankrishna10/npc-fin-prm-7b)

## What's in this directory

This dir is **not** a full training pipeline (the model was trained on
the founder's H100 in March 2026 with private scaffolding that wasn't
preserved). It contains:

```
npc-fin-prm-7b/
├── README.md                       # this file
├── configs/
│   ├── training_metadata.json      # verified hyperparameters from the trained adapter
│   └── adapter_config.json         # PEFT adapter config (pulled from HF)
├── scripts/
│   └── build_ood_set.py            # rebuild the GSM8K + MATH-500 OOD test set
├── eval/
│   ├── run_prm.py                  # PRM inference harness (PEFT + bnb 4-bit)
│   ├── score_predictions.py        # Spearman, F1, MAE, ECE, confusion matrix
│   ├── eval_ood.py                 # OOD wrapper around run_prm.py
│   └── run_on_pod.sh               # one-shot orchestrator (run on the pod)
└── analysis/
    ├── phase1_findings.json        # all Phase-1 numbers — paper-ready citations
    └── ood_steps.jsonl             # 307-step OOD test set (reproducible from build_ood_set.py)
```

The original training scripts that produced the adapter on HF will be
ported in when the paper's iteration-ablation runs (see "Honest gaps"
below).

## Verified training run

Pulled directly from the published `trainer_state.json` and
`training_metadata.json`:

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Method | QLoRA — 4-bit NF4 base + bf16 LoRA adapters |
| LoRA | r=32, α=64, dropout=0.05, all 7 modules (q/k/v/o + gate/up/down) |
| Optimizer | AdamW 8-bit, cosine LR, lr=2e-4, warmup 0.05, bf16 |
| Effective batch | 16 (per-device 2 × grad-accum 8) |
| Max seq | 2,048 tokens |
| Total steps | 7,041 (3 × 2,346 steps/epoch) |
| Epochs | 3 |
| Train examples | 37,542 |
| Val examples | 4,171 |
| **Final train loss** | **0.2518** |
| **Final eval loss** | **0.2168** |
| Wall clock | 17.4 hours (62,600 s) |
| Hardware | Single NVIDIA H100 80 GB |
| Peak VRAM | 23.69 GB |
| Trained | 2026-03-20 → 2026-03-21 |

Loss curve descended cleanly from `0.4839` (step 200) to `0.2158` (step 6,200) with no
LR resets and no loss spikes — see `analysis/phase1_findings.json`.

## Output schema

```json
{
  "overall_score": 0.85,
  "rating": "STRONG | ACCEPTABLE_WITH_ISSUES | FLAWED",
  "dimensions": {
    "factual_accuracy":  {"score": 0.90, "justification": "..."},
    "logical_validity":  {"score": 0.85, "justification": "..."},
    "completeness":      {"score": 0.80, "justification": "..."},
    "risk_awareness":    {"score": 0.85, "justification": "..."}
  },
  "explanation": "Step correctly identifies X but underweights Y risk.",
  "error_identified": "LOGICAL_FALLACY | OMISSION | UNJUSTIFIED_CONFIDENCE | FACTUAL_ERROR | LOGICAL_LEAP | null"
}
```

## Phase 1 findings — paper-ready (computed from the live val set)

### Distribution
- 4,171 val examples, **0** JSON parse failures (judge produced clean JSON 100%)
- Score is bimodal — peaks at 0.6–0.7 (n=555) and 0.8–0.9 (n=2,121)
- Rating: 55.6 % STRONG · 30.1 % ACCEPTABLE_WITH_ISSUES · 14.3 % FLAWED
- Domain: **DeFi-heavy crypto** (top tokens: APY, TVL, ETH, LTV, USDC, FRAX, GHO, PYUSD, USDT)

### Error taxonomy
| Type | Count | Share |
|---|---:|---:|
| LOGICAL_FALLACY | 985 | 57.2 % |
| OMISSION | 613 | 35.6 % |
| UNJUSTIFIED_CONFIDENCE | 77 | 4.5 % |
| FACTUAL_ERROR | 45 | 2.6 % |
| LOGICAL_LEAP | 1 | 0.1 % |

### Rating ↔ overall_score is a strict partition

| Rating | n | min | max |
|---|---:|---:|---:|
| STRONG | 2,320 | 0.80 | 0.95 |
| ACCEPTABLE_WITH_ISSUES | 1,255 | 0.50 | 0.80 |
| FLAWED | 596 | 0.11 | 0.49 |

Zero overlap → the judge applies a deterministic threshold. Rating
prediction is therefore redundant with score prediction.

### 🔥 The headline result: dimensions are not orthogonal

Pairwise Spearman among the four gold-label dimensions (n=4,171):

| | factual | logical | complete | risk |
|---|---:|---:|---:|---:|
| **factual** | 1.000 | 0.743 | 0.772 | 0.715 |
| **logical** | — | 1.000 | 0.851 | **0.916** |
| **complete** | — | — | 1.000 | 0.863 |
| **risk** | — | — | — | 1.000 |

`logical_validity ↔ risk_awareness` correlate at **0.916**. Three of the
four dimensions form a tight cluster (0.85–0.92).

Spearman of `overall_score` vs each dimension:

| Dimension | rho |
|---|---:|
| logical_validity | **0.955** |
| risk_awareness | 0.949 |
| completeness | 0.864 |
| factual_accuracy | 0.762 |

→ The judge's `overall_score` is **~95 % explained by `logical_validity` alone**. The
4-dimension framing is partially redundant — a cautionary result for the
multi-dimensional PRM design pattern, and the central contribution of the
"Cheap PRMs" paper.

## Phase 2 eval — n=200, MLX 4-bit on Apple M5 (2026-04-26)

Independent re-evaluation of the published model card claims, run on a
laptop (no pod required) using a stratified 200-example val sample
(80 STRONG / 80 ACCEPTABLE_WITH_ISSUES / 40 FLAWED).

| Metric | Card claims | This run (n=200) | Match? |
|---|---:|---:|:---:|
| Spearman (overall_score) | 0.94 | **0.9234** | ✓ |
| Step-level rating accuracy | 89.2 % | **88.50 %** | ✓ |
| Error-detection F1 (FLAWED) | 0.87 | **0.8421** | ✓ |
| MAE on overall_score | — | **0.0404** | — |
| Parse failures | — | **0 / 200** | ✓ |

Per-dimension Spearman:

| Dimension | ρ |
|---|---:|
| factual_accuracy | 0.842 |
| logical_validity | **0.931** |
| completeness | 0.865 |
| risk_awareness | 0.908 |

Confusion matrix (gold → pred):

| Gold \\ Pred | STRONG | ACCEPTABLE | FLAWED |
|---|---:|---:|---:|
| **STRONG** (80) | **79** | 1 | 0 |
| **ACCEPTABLE** (80) | 10 | **66** | 4 |
| **FLAWED** (40) | 0 | 8 | **32** |

The model **never confuses STRONG with FLAWED at the extremes** (zero
off-diagonal between rows 1 and 3). Most errors are STRONG↔ACCEPTABLE
on borderline reasoning — an intuitive failure mode for a graded
classifier.

### Calibration is the headline issue (ECE = 0.21)

```
P(flawed) bin    n    predicted    actual_flawed_rate
[0.1, 0.2)      84      0.146         0.000        ← way over
[0.3, 0.4)      38      0.331         0.053        ← still over
[0.5, 0.6)      20      0.566         0.900        ← under
[0.7, 0.8)       5      0.710         1.000        ← ok
```

The PRM **ranks well (Spearman 0.92)** but the score scale is biased
toward over-flagging in the 0.1–0.5 band and under-flagging at 0.5–0.6.
**Don't use raw `1 - overall_score` as a calibrated probability** —
apply Platt-scaling or isotonic regression first.

Full report: `analysis/eval_report_n200_mlx4bit.txt`.
Per-row predictions: `analysis/eval_preds_n200.jsonl`.

### Reproduction notes

Run on a 24 GB Apple M5 in ~50 minutes:

1. Merge the published PEFT adapter into Qwen2.5-7B-Instruct on CPU (fp16, ~4 min)
2. Convert merged fp16 → MLX 4-bit (`mlx_lm.convert`, ~2 min)
3. Run `eval/run_mlx_eval.py` on a 200-example stratified val sample (~45 min)
4. Score with `eval/score_predictions.py`

The original `eval/run_prm.py` (PyTorch + PEFT + bnb 4-bit) **does NOT
work on Apple Silicon** because Metal caps single-buffer allocations at
~14 GiB and the fp16 base hits this limit. Use the MLX path on Mac;
use `run_prm.py` on CUDA pods.

## Honest gaps still open

| Gap | Status | Plan |
|---|---|---|
| **Original training scripts** | Not in repo | Port in when the paper's iteration-ablation runs |
| **"3 bootstrapped iterations" claim on HF card** | **Wrong.** Trainer state shows 1 continuous run × 3 epochs. No LR resets, no loss spikes. | Fix HF card: replace "3 iterations" with "3 epochs"; or actually run the bootstrap loop and earn the claim |
| **Calibration layer** | Not in shipped model | Train Platt-scaling head on the eval predictions; ship as a sidecar |
| **Human-eval baseline** | Not run | ~3 hr of manual scoring on 200 val steps, by the author |
| **OOD probe (math reasoning)** | **Done.** 5.2% mis-flag on gold-correct GSM8K + MATH-500 (16/307 FLAWED, mean score 0.856). See `analysis/eval_report_ood.txt`. | Cross-domain transfer is better than expected. |
| **Best-of-N downstream impact** | Blocked on NPC Agentic v2 completion | Run after v2 ships |

## Reproducing the eval

The eval harness in `eval/` is GPU-bound (need ~6 GB VRAM for the 7B base
in 4-bit + the LoRA adapter). Run on the same A40 / H100 you have access
to:

```bash
# 1. Stage on the pod
ssh root@<pod>:<port> mkdir -p /workspace/finprm-eval
scp -r training/npc-fin-prm-7b/eval/* root@<pod>:/workspace/finprm-eval/eval/
scp training/npc-fin-prm-7b/analysis/ood_steps.jsonl root@<pod>:/workspace/finprm-eval/ood/

# 2. Pull val examples from HF on the pod (12 MB)
python3 -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('ramankrishna10/npc-fin-prm-7b', 'val_examples.jsonl', \
    repo_type='model', local_dir='/workspace/finprm-eval')"

# 3. Run all evals (~1 hr on A40)
cd /workspace/finprm-eval && bash eval/run_on_pod.sh
```

Outputs land in `/workspace/finprm-eval/results/`:
- `val_preds.jsonl`     — PRM predictions on 500 val examples
- `score_id.txt`        — Spearman / F1 / ECE / confusion matrix
- `ood_preds.jsonl`     — PRM predictions on 307 OOD math-reasoning steps
- `run_*.log`           — full inference logs

## Tech stack

- **Frameworks:** PyTorch 2.10, Transformers ≥ 4.57, PEFT ≥ 0.18, bitsandbytes ≥ 0.49
- **Training:** Unsloth + TRL `SFTTrainer` (per the model card), bf16 mixed precision
- **Inference:** PEFT adapter on bnb 4-bit base → ~6 GB VRAM
- **Tracking:** WandB (private project at training time)

## Use cases

1. **Best-of-N sampling.** Generate N reasoning chains from a generator (NPC Agentic v2,
   Qwen2.5, etc.), score each step with the PRM, pick the chain with highest
   mean step score.
2. **Filter-and-fix loops.** Flag steps where overall_score < threshold, re-prompt the
   generator to redo just that step.
3. **Reasoning evaluation.** Use as an automated verifier in CI for any
   reasoning-model deployment.
4. **Reward model for RL.** Per-step PRM scores can substitute for outcome-only rewards
   in PPO/GRPO training of a reasoning model.

## Limitations (honest)

1. **DeFi/crypto-specific.** Domain transfer to non-crypto reasoning (math, science,
   legal, medical) is unvalidated. The OOD probe in this dir measures exactly that.
2. **Judge ceiling.** Labels come from Qwen2.5-72B; the PRM's Spearman of 0.94 vs the judge
   is bounded by the judge's own quality. Spearman vs human is the missing eval.
3. **Dimensions partially redundant.** See the orthogonality result above.
4. **Score thresholds aren't human-calibrated.** The 0.5 / 0.8 cutoffs are inherited
   from the judge's own distribution; users should re-calibrate for their own pipelines.
5. **Not a replacement for human review** in production financial decisions.

## Citation

> Bachu, R. K. (2026). *Cheap PRMs: Multi-Dimensional Process Reward
> Modeling for Domain-Specialized Reasoning.* Zenodo.
> https://doi.org/10.5281/zenodo.19800784

```bibtex
@misc{bachu2026cheapprms,
  title     = {Cheap PRMs: Multi-Dimensional Process Reward Modeling for
               Domain-Specialized Reasoning},
  author    = {Bachu, Rama Krishna},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19800784},
  url       = {https://doi.org/10.5281/zenodo.19800784},
  note      = {Preprint},
}
```

## Related

- [`npc-fin-32b/`](../npc-fin-32b/) — finance-reasoning generator (retired); the model
  this PRM was originally built to verify
- [`npc-agentic-7b-v2/`](../npc-agentic-7b-v2/) — generalist reasoning model; PRM will
  be re-evaluated against v2 as the OOD downstream-impact target

## Author

**Rama Krishna Bachu** (`dude.npc`) — founder, Falcon Hash → Bottensor.
[ORCID 0009-0000-1298-0681](https://orcid.org/0009-0000-1298-0681) ·
[ramakrishna.bachu@bottensor.xyz](mailto:ramakrishna.bachu@bottensor.xyz)
