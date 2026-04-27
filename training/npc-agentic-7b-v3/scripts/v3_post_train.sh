#!/bin/bash
# v3 post-train orchestrator.
#
# Differences vs v2's version:
#   1. Real exit-code checks per step (v2's "tee -a $LOG | exit_code" mask
#      always returned 0 because tee succeeded, hiding upstream failures).
#   2. Step name "(or failed)" wording removed — failure is now an
#      actual halt, not a silent passthrough.
#   3. 03b_gsm8k_rerun is included as STEP 4b (was missing in v2's
#      orchestrator; we had to scp it onto the pod and run manually).
#   4. Adapter-existence guards in each step's preconditions, so a
#      manual re-run can pick up wherever the last failure left off.
#
# Should be launched with `nohup` + `disown` so an SSH disconnect
# doesn't kill the chain (v2 lesson — see runbook).

set +e   # don't bail on errors at the bash level — handle per-step

cd /workspace/training/active/npc-agentic
source /workspace/venvs/training/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/workspace/tmp
export TOKENIZERS_PARALLELISM=false
export GGUF_SKIP_SMOKE=1

LOG=/workspace/training/active/npc-agentic/logs/pipeline.log
: > $LOG

ts() { date -u +%H:%M:%S; }

run_step() {
  local name=$1
  local script=$2
  echo "[$(ts)] ============ ${name} ============" | tee -a $LOG
  python -u "${script}" 2>&1 | tee -a $LOG
  rc=${PIPESTATUS[0]}
  echo "[$(ts)] >> ${name}: exit=${rc}" | tee -a $LOG
  if [ "${rc}" -ne 0 ]; then
    echo "[$(ts)] !! ${name} FAILED (rc=${rc}). HALTING chain." | tee -a $LOG
    exit "${rc}"
  fi
}

echo "[$(ts)] v3 orchestrator starting; waiting for training" | tee -a $LOG

FINAL=/workspace/training/active/npc-agentic/checkpoints/final/adapter_model.safetensors
while ! [ -f "$FINAL" ]; do
  sleep 120
done
echo "[$(ts)] >> training finished — final adapter detected" | tee -a $LOG
sleep 15

# v3 already has v3 suffix in config.MODEL_SHORTNAME — no bump needed.

run_step "STEP 4: eval"             03_evaluate.py
run_step "STEP 4b: GSM8K re-extract" 03b_gsm8k_rerun.py
run_step "STEP 5: merge"            04_merge.py
run_step "STEP 6: GPTQ quantize"    05_quantize.py
run_step "STEP 6b: GGUF"            07_gguf.py
run_step "STEP 7: push 4 repos"     06_push.py

echo "[$(ts)] ============ PIPELINE DONE ============" | tee -a $LOG
