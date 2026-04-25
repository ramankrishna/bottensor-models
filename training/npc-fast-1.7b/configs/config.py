"""
NPC Fast — central configuration.

All paths, hyperparameters, identity, and the curriculum schedule live here
so every module imports from a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EVAL_RESULTS_PATH = OUTPUT_DIR / "eval_results.json"
NEEDLE_HEATMAP_PATH = OUTPUT_DIR / "needle_heatmap.png"
DATASETS_REGISTRY = ROOT / "datasets.json"
for _d in (OUTPUT_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Identity ----
MODEL_NAME = "NPC Fast"
BUILT_BY = "Bottensor"
PARENT_COMPANY = "Falcon Hash"
CREATOR = "dude.npc"
ROLE = "Fast agentic router — 1.7B, trained to 16K, router head fine-tuned for NPC Fin escalation"
PARTNER_MODEL = "NPC Fin 32B"

# ---- Base model ----
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192
TARGET_MAX_POSITION_EMBEDDINGS = 131_072   # 128K
YARN_FACTOR = float(TARGET_MAX_POSITION_EMBEDDINGS) / float(ORIGINAL_MAX_POSITION_EMBEDDINGS)

# ---- Training ----
TRAINING_CONFIG: dict = {
    # Model
    "model_name": BASE_MODEL,
    "max_position_embeddings": TARGET_MAX_POSITION_EMBEDDINGS,

    # Training
    "num_train_epochs": 5,
    "max_steps": 6_200,
    "learning_rate": 5e-6,
    "min_learning_rate": 1e-7,
    "lr_scheduler_type": "cosine_with_restarts",
    "num_cycles": 3,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,

    # Precision
    "bf16": True,
    "tf32": True,

    # Memory
    "gradient_checkpointing": True,
    "use_flash_attention_2": True,

    # Saving
    "save_steps": 500,
    "save_total_limit": 5,
    "output_dir": str(CHECKPOINT_DIR),

    # Logging
    "logging_steps": 10,
    "report_to": "wandb",
    "wandb_project": "npc-fast",
    "wandb_run_name": "smollm2-1.7b-npc-fast-full",

    # Eval
    "eval_steps": 1000,
    "eval_strategy": "steps",

    # Data
    "dataset_registry": str(DATASETS_REGISTRY),
    "val_split": 0.05,
    "seed": 42,

    # DataLoader
    "dataloader_num_workers": 4,
    "optim": "adamw_torch",
}

# ---- Curriculum schedule ----
# Each stage expands the effective context window while keeping the
# *effective* batch size constant at 32 tokens-per-step batches.
#
# Stages 1-3 are the main continual-pretraining budget (6000 steps,
# up to 32K context). Stage 4 (64K) is skipped — in practice most of
# the learning happens ≤32K and the 64K stage was dominated-cost for
# marginal gain. Stage 5 is a short 200-step *anneal* at full 128K
# context so the model actually exercises the YaRN-scaled RoPE before
# export; without this the model is technically 128K-capable (via
# YaRN) but hasn't been *trained* at long context at all.
CURRICULUM: list[dict] = [
    {"stage": 1, "start": 0,     "end": 2_000, "max_seq_length": 4_096,   "micro_batch": 8, "grad_accum": 4},
    # micro_batch halved vs original spec — even with liger-kernel's fused
    # cross-entropy, 16K×4 samples of activation memory is tight once
    # gradient-checkpointing re-forwards fire under peak load. Keeps eff
    # batch 32 via 2× grad_accum.
    {"stage": 2, "start": 2_000, "end": 4_000, "max_seq_length": 16_384,  "micro_batch": 2, "grad_accum": 16},
    {"stage": 3, "start": 4_000, "end": 6_000, "max_seq_length": 32_768,  "micro_batch": 1, "grad_accum": 32},
    # stage 4 (64K) intentionally omitted — see note above
    {"stage": 5, "start": 6_000, "end": 6_200, "max_seq_length": 65_536,  "micro_batch": 1, "grad_accum": 32},
]

EFFECTIVE_BATCH_SIZE = 32  # micro_batch × grad_accum is constant across stages


def stage_for_step(global_step: int) -> dict:
    """Return the curriculum stage active at this global step."""
    for stage in CURRICULUM:
        if stage["start"] <= global_step < stage["end"]:
            return stage
    # Past the last stage — pin to final
    return CURRICULUM[-1]


# ---- HF publishing targets ----
HF_REPO_FP16 = "ramankrishna10/npc-fast-1.7b"
HF_REPO_GPTQ = "ramankrishna10/npc-fast-1.7b-gptq"
HF_REPO_GGUF = "ramankrishna10/npc-fast-1.7b-gguf"

# ---- Env helpers ----
def hf_token() -> str | None:
    return os.getenv("HF_TOKEN")


def wandb_key() -> str | None:
    return os.getenv("WANDB_API_KEY")
