#!/bin/bash
set -euo pipefail

# Resolve repo root (parent of scripts/)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-npc-fast}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[run_train] starting NPC Fast training..."
python train.py \
    --model_name HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --output_dir "$ROOT/output/checkpoints/" \
    --max_steps 6200 \
    --learning_rate 5e-6

echo "[run_train] training complete — running eval suite..."
bash "$HERE/run_eval.sh"
