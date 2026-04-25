#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-$ROOT/output/checkpoints/final/}"
GPTQ_DIR="$ROOT/output/npc-fast-1.7b-gptq/"
GGUF_DIR="$ROOT/output/npc-fast-1.7b-gguf/"

if [ ! -d "$CHECKPOINT" ]; then
    echo "[run_export] checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "[run_export] quantizing to GPTQ 4-bit → $GPTQ_DIR"
python -m export.quantize --model_path "$CHECKPOINT" --output_dir "$GPTQ_DIR"

echo "[run_export] converting to GGUF → $GGUF_DIR"
python -m export.gguf --model_path "$CHECKPOINT" --output_dir "$GGUF_DIR"

echo "[run_export] pushing to HuggingFace"
python -m export.push_hf --model_path "$CHECKPOINT" --repo ramankrishna10/npc-fast-1.7b
python -m export.push_hf --model_path "$GPTQ_DIR"   --repo ramankrishna10/npc-fast-1.7b-gptq
python -m export.push_hf --model_path "$GGUF_DIR"   --repo ramankrishna10/npc-fast-1.7b-gguf

echo "[run_export] complete."
