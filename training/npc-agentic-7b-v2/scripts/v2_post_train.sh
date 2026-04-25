#!/bin/bash
cd /workspace/training/active/npc-agentic
source /workspace/venvs/training/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/workspace/tmp
export TOKENIZERS_PARALLELISM=false
export GGUF_SKIP_SMOKE=1

LOG=/workspace/training/active/npc-agentic/logs/pipeline.log
: > $LOG
echo "[$(date +%H:%M:%S)] v2 orchestrator starting; waiting for training" | tee -a $LOG

# Wait for the final adapter. Training runs for 2+ days, watchdog can restart it.
# We don't abort if trainer+watchdog briefly both miss — we trust the watchdog.
FINAL=/workspace/training/active/npc-agentic/checkpoints/final/adapter_model.safetensors
while ! [ -f "$FINAL" ]; do
  sleep 120
done
echo "[$(date +%H:%M:%S)] >> training finished — final adapter detected" | tee -a $LOG
sleep 15

# Rename to v2 before push (so we don't overwrite v1)
python - <<PY | tee -a $LOG
p = "config.py"
s = open(p).read()
if "MODEL_SHORTNAME = \"npc-agentic-7b\"" in s:
    s = s.replace('MODEL_SHORTNAME = "npc-agentic-7b"', 'MODEL_SHORTNAME = "npc-agentic-7b-v2"')
    open(p, "w").write(s)
    print("  MODEL_SHORTNAME bumped to npc-agentic-7b-v2")
else:
    print("  (v2 suffix already applied or base name differs)")
PY

step() {
  echo "[$(date +%H:%M:%S)] ============ $1 ============" | tee -a $LOG
}

step "STEP 4: eval"
python -u 03_evaluate.py 2>&1 | tee -a $LOG
echo "[$(date +%H:%M:%S)] >> eval done" | tee -a $LOG

step "STEP 5: merge"
python -u 04_merge.py 2>&1 | tee -a $LOG
echo "[$(date +%H:%M:%S)] >> merge done" | tee -a $LOG

step "STEP 6: GPTQ quantize"
python -u 05_quantize.py 2>&1 | tee -a $LOG
echo "[$(date +%H:%M:%S)] >> GPTQ done (or failed)" | tee -a $LOG

step "STEP 6b: GGUF"
python -u 07_gguf.py 2>&1 | tee -a $LOG
echo "[$(date +%H:%M:%S)] >> GGUF done (or failed)" | tee -a $LOG

step "STEP 7: push 4 repos"
python -u 06_push.py 2>&1 | tee -a $LOG
echo "[$(date +%H:%M:%S)] >> push done" | tee -a $LOG

step "PIPELINE DONE"
