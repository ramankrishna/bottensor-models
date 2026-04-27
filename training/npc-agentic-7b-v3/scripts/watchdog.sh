#!/bin/bash
cd /workspace/training/active/npc-agentic
LOG=/workspace/training/active/npc-agentic/logs/watchdog.log
echo "[$(date -u +%FT%TZ)] watchdog starting" >> $LOG
MAX_RESTARTS=5
N=0
while true; do
  if pgrep -f "02_train.py" >/dev/null; then
    sleep 60
    continue
  fi
  # If final adapter exists, training finished cleanly — stop the watchdog
  if [ -f /workspace/training/active/npc-agentic/checkpoints/final/adapter_model.safetensors ]; then
    echo "[$(date -u +%FT%TZ)] training finished (final adapter present) — watchdog exiting" >> $LOG
    exit 0
  fi
  N=$((N+1))
  if [ $N -gt $MAX_RESTARTS ]; then
    echo "[$(date -u +%FT%TZ)] too many restarts ($N) — giving up" >> $LOG
    exit 1
  fi
  echo "[$(date -u +%FT%TZ)] trainer not running — launching (attempt $N)" >> $LOG
  source /workspace/venvs/training/bin/activate
  export HF_HOME=/workspace/hf_cache
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export WANDB_PROJECT=npc-agentic
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES=0
  export TMPDIR=/workspace/tmp
  nohup python -u 02_train.py >> logs/train.log 2>&1 &
  sleep 30
done
