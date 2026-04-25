#!/bin/bash
set -euo pipefail
cd /workspace/npc-fast-trainer
source ./.env
export TOKENIZERS_PARALLELISM=false
exec python main.py train \
    --resume_from_checkpoint /workspace/npc-fast-trainer/output/checkpoints/checkpoint-2500-resume \
    --target_step 6200
