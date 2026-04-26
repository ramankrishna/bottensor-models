# Paper — Cheap PRMs

> **Status: PUBLISHED on Zenodo (2026-04-26). DOI: [10.5281/zenodo.19800784](https://doi.org/10.5281/zenodo.19800784).**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19800784.svg)](https://doi.org/10.5281/zenodo.19800784)

LaTeX source and compiled PDF for the paper accompanying NPC Fin-PRM 7B.

## Cite

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

## Files

| File | What |
|---|---|
| `cheap-prms.tex` | LaTeX source (mirrors NPC Fast paper structure / class / hyperref setup) |
| `cheap-prms.pdf` | Compiled draft, 8 pages, ~96 KB |

## What's filled

All headline numbers from the runs in this directory:

- **Training config** (Section 4) — verified from the published `trainer_state.json`
- **Dataset stats** (Section 3) — score distributions, rating partition, error taxonomy
- **Orthogonality** (Section 3.4) — pairwise Spearman among the 4 dimensions, $\rho \in [0.85, 0.92]$ for the cluster
- **In-distribution eval** (Section 5.1) — Spearman 0.9234, F1 0.8421, MAE 0.0404, confusion matrix
- **Calibration** (Section 5.2) — ECE = 0.21 reliability table (paper's headline new finding)
- **OOD eval** (Section 5.3) — 5.2% mis-flag rate on 307 gold-correct GSM8K+MATH-500 steps, plus the rating-extrapolation observation (`EXCELLENT`/`PERFECT` emitted on 3.9% of OOD steps despite never appearing in training)
- **Limitations** (Section 6) — bootstrap not run, judge ceiling, dimension redundancy, calibration uncorrected, no human eval

## Compile

```bash
cd training/npc-fin-prm-7b/paper
pdflatex -interaction=nonstopmode cheap-prms.tex
# → cheap-prms.pdf, 8 pages
```

Requires `pdflatex` + `txfonts` + `hyperref` + `booktabs` (standard TeX Live install).
