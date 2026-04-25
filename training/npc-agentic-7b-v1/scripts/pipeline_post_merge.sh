#!/bin/bash
cd /workspace/training/active/npc-agentic
source /workspace/venvs/training/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/workspace/tmp
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

LOG=/workspace/training/active/npc-agentic/logs/pipeline.log
: > $LOG
echo "[$(date +%H:%M:%S)] PIPELINE RESUME (merge already done)" | tee -a $LOG

if [ ! -f /workspace/training/active/npc-agentic/merged/config.json ]; then
    echo "!! no merged/ — cannot resume" | tee -a $LOG
    exit 2
fi

# Step 6 — GPTQ
if [ ! -f /workspace/training/active/npc-agentic/quantized/config.json ]; then
  echo "[$(date +%H:%M:%S)] ============================================" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] STEP 6: GPTQ W4A16 quantize" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] ============================================" | tee -a $LOG
  if ! python -u 05_quantize.py 2>&1 | tee -a $LOG; then
    echo "[$(date +%H:%M:%S)] !! GPTQ failed — aborting" | tee -a $LOG
    exit 3
  fi
  echo "[$(date +%H:%M:%S)] >> GPTQ done" | tee -a $LOG
else
  echo "[$(date +%H:%M:%S)] STEP 6: skipping (quantized/ exists)" | tee -a $LOG
fi

# Step 6b — GGUF
if ! ls /workspace/training/active/npc-agentic/gguf/*-Q4_K_M.gguf 2>/dev/null | grep -q .; then
  echo "[$(date +%H:%M:%S)] ============================================" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] STEP 6b: GGUF build + quants" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] ============================================" | tee -a $LOG
  if ! python -u 07_gguf.py 2>&1 | tee -a $LOG; then
    echo "[$(date +%H:%M:%S)] !! GGUF failed — continuing to push anyway (will push 3 repos)" | tee -a $LOG
  else
    echo "[$(date +%H:%M:%S)] >> GGUF done" | tee -a $LOG
  fi
else
  echo "[$(date +%H:%M:%S)] STEP 6b: skipping (gguf/ already populated)" | tee -a $LOG
fi

# Step 7 — push
echo "[$(date +%H:%M:%S)] ============================================" | tee -a $LOG
echo "[$(date +%H:%M:%S)] STEP 7: push to HuggingFace" | tee -a $LOG
echo "[$(date +%H:%M:%S)] ============================================" | tee -a $LOG
python -u 06_push.py 2>&1 | tee -a $LOG
echo "[$(date +%H:%M:%S)] >> push exit code: $?" | tee -a $LOG

echo "[$(date +%H:%M:%S)] PIPELINE DONE" | tee -a $LOG
