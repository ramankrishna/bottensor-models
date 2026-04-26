# Paper draft — Cheap PRMs

> **Status: draft, in preparation. Awaiting OOD eval results to fill three placeholder fields, then Zenodo upload.**

LaTeX source and compiled PDF for the paper accompanying NPC Fin-PRM 7B.

## Files

| File | What |
|---|---|
| `cheap-prms.tex` | LaTeX source (mirrors NPC Fast paper structure / class / hyperref setup) |
| `cheap-prms.pdf` | Compiled draft (8 pages) |

## What's in / what's pending

Filled with **real numbers from Phase 1 + Phase 2** runs:
- All training-config numbers (verified from the published `trainer_state.json`)
- Phase 1 dataset analysis (orthogonality finding, score distributions, error taxonomy)
- Phase 2 in-distribution eval on n=200 (Spearman, F1, MAE, confusion matrix)
- Calibration table (ECE = 0.21) — paper's headline new finding

Placeholders (3 fields, abstract + Section 5.3 OOD table):
- `[OOD_REJECTION_RATE]` — % of gold-correct math reasoning the PRM mis-flags as FLAWED
- `[OOD_FLAGGED_RATE]` — same metric, expressed differently in body text
- `[OOD_MEAN_SCORE]` — mean PRM `overall_score` on OOD math steps

These fill in once the OOD eval (`eval/run_mlx_eval.py` on `analysis/ood_steps.jsonl`) finishes.

## Compile

```bash
cd training/npc-fin-prm-7b/paper
pdflatex -interaction=nonstopmode cheap-prms.tex
# → cheap-prms.pdf, 8 pages
```

Requires `pdflatex` + `txfonts` + `hyperref` + `booktabs` (standard TeX Live install).
