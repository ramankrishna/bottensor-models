#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

if [ -f "$ROOT/.env" ]; then
    set -o allexport
    source "$ROOT/.env"
    set +o allexport
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-$ROOT/output/checkpoints/final/}"
if [ ! -d "$CHECKPOINT" ]; then
    echo "[run_eval] checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "[run_eval] BFCL / IFEval / agentic"
python -m eval.benchmarks --model_path "$CHECKPOINT"

echo "[run_eval] needle-in-haystack (up to 128K)"
python -m eval.context_eval --model_path "$CHECKPOINT"

echo "[run_eval] router eval"
python -m eval.router_eval --model_path "$CHECKPOINT"

echo "[run_eval] perplexity"
python -m eval.perplexity --model_path "$CHECKPOINT"

echo "[run_eval] complete — results in output/eval_results.json"
