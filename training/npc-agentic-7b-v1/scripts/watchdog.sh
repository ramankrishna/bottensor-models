#!/bin/bash
# Keeps training alive. If the trainer dies (but not via clean TRAINING DONE),
# wait 60s then restart. Loops forever until `TRAINING DONE` appears in the log.
cd /workspace/training/active/npc-agentic
source /workspace/venvs/training/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/workspace/tmp
export WANDB_PROJECT=npc-agentic
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

WATCHDOG_LOG=logs/watchdog.log
mkdir -p logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog starting" >> $WATCHDOG_LOG

while true; do
    if grep -q "TRAINING DONE" logs/train.log 2>/dev/null; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TRAINING DONE detected — watchdog exiting" >> $WATCHDOG_LOG
        break
    fi
    if ! pgrep -f "02_train.py" > /dev/null; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] trainer not running — launching" >> $WATCHDOG_LOG
        nohup python -u 02_train.py >> logs/train.log 2>&1 &
        sleep 60
    fi
    sleep 30
done
