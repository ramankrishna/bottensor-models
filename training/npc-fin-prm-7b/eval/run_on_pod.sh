#!/usr/bin/env bash
# Orchestration script — kick off all PRM evals AFTER v2 finishes (or on a fresh A40).
#
# Run on the pod. Assumes:
#   - HF cache has already pulled Qwen/Qwen2.5-7B-Instruct (it has — used for v2)
#   - HF token at $HF_HOME/token (write scope not needed for read)
#   - val_examples.jsonl + ood/ood_steps.jsonl uploaded to /workspace/finprm-eval/
#
# Outputs land in /workspace/finprm-eval/results/.
# Total wall time ~2-3 hr on a single A40 once v2 frees the GPU.

set -euo pipefail

ROOT=/workspace/finprm-eval
RESULTS=$ROOT/results
mkdir -p "$RESULTS"

source /workspace/venvs/training/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

cd "$ROOT/eval"

# 1. In-distribution Spearman (real number to compare against the 0.94 claim)
python3 run_prm.py \
  --input "$ROOT/val_examples.jsonl" \
  --output "$RESULTS/val_preds.jsonl" \
  --n 500 \
  --batch 4 \
  --load-4bit \
  2>&1 | tee "$RESULTS/run_id.log"

python3 score_predictions.py "$RESULTS/val_preds.jsonl" \
  2>&1 | tee "$RESULTS/score_id.txt"

# 2. OOD probe (math reasoning is gold-correct — what does PRM say?)
python3 eval_ood.py \
  --ood "$ROOT/ood/ood_steps.jsonl" \
  --out "$RESULTS/ood_preds.jsonl" \
  --load-4bit \
  2>&1 | tee "$RESULTS/run_ood.log"

# 3. (Optional) Best-of-N on NPC Agentic v2 — once v2 is merged + pushed
# Requires the v2 merged model to exist
# python3 eval_best_of_n.py ...

echo
echo "All eval results in $RESULTS/"
ls -la "$RESULTS/"
