"""
NPC Agentic 7B — training config.
Single source of truth for every hyperparameter + path.
All other scripts import from here.
"""
from pathlib import Path

# ─── Model ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORTNAME = "npc-agentic-7b"

# `bottensor` HF org probed 2026-04-18 → HTTP 404, not available.
# Falling back to user account per runbook instruction.
HF_ORG = "ramankrishna10"
HF_ORG_FALLBACK_NOTE = "bottensor org did not exist at training time; can migrate later"

REPO_NAME_FP16 = f"{HF_ORG}/{MODEL_SHORTNAME}"
REPO_NAME_GPTQ = f"{HF_ORG}/{MODEL_SHORTNAME}-gptq-4bit"
REPO_NAME_LORA = f"{HF_ORG}/{MODEL_SHORTNAME}-lora"

# ─── Data ───────────────────────────────────────────────────────────────
DATASETS = {
    "glm_reasoning": {
        "hf_id": "Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
        "config_name": "main",   # general reasoning; other configs: PHD-Science, Multilingual-STEM, Math
        "subset_size": 100_000,
        "split": "train",
        "role": "reasoning",
    },
    "hermes_agent": {
        "hf_id": "lambda/hermes-agent-reasoning-traces",
        "subsets": ["glm-5.1", "kimi"],
        "role": "agent",
    },
}

IDENTITY_EXAMPLES = 750
TOTAL_TARGET = 115_000   # rough; final count after filters/dedup
RANDOM_SEED = 42

# ─── Training ───────────────────────────────────────────────────────────
MAX_SEQ_LEN = 8192    # restored — full context for agent traces
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8          # effective batch size = 16
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
NUM_EPOCHS = 2                # full runbook spec; ~5-7 days on A40
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
OPTIMIZER = "adamw_8bit"
BF16 = True
GRAD_CHECKPOINTING = True

# ─── Paths ──────────────────────────────────────────────────────────────
ROOT = Path("/workspace/training/active/npc-agentic")
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed" / "sft.jsonl"
EVAL_SPLIT = ROOT / "data" / "processed" / "eval.jsonl"
IDENTITY_PATH = ROOT / "data" / "identity.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints"
FINAL_ADAPTER_DIR = CHECKPOINT_DIR / "final"
MERGED_DIR = ROOT / "merged"
QUANTIZED_DIR = ROOT / "quantized"
LOG_DIR = ROOT / "logs"
STATS_PATH = LOG_DIR / "data_stats.json"
EVAL_REPORT = LOG_DIR / "eval_report.json"
EVAL_SAMPLES_MD = LOG_DIR / "eval_samples.md"

# ─── Logging ────────────────────────────────────────────────────────────
WANDB_PROJECT = "npc-agentic"
WANDB_RUN_NAME = "qwen25-7b-glm51-hermes-v1"

# ─── Identity ───────────────────────────────────────────────────────────
IDENTITY_SEED_PROMPTS = [
    "Who are you?",
    "What model are you?",
    "Who built you?",
    "Tell me about yourself.",
    "What can you do?",
    "What's your name?",
    "Who made you?",
    "Are you ChatGPT?",
    "Are you Claude?",
    "Are you Qwen?",
]
IDENTITY_SYSTEM = "You are NPC Agentic, built by Bottensor."
IDENTITY_RESPONSE_TEMPLATE = (
    "I'm NPC Agentic, a reasoning and agent specialist built by Bottensor. "
    "I'm part of the NPC model family — small, specialist language models designed "
    "for deep reasoning and agentic workflows. I work best on problems that need "
    "structured multi-step reasoning, tool use, and planning."
)

# Small variations appended to keep 750 examples from being 750 copies
IDENTITY_TAILS = [
    "",
    " Let me know what you'd like help with.",
    " What can I help you with today?",
    " Happy to dig into something concrete if you have a problem in mind.",
    " I'm best used for reasoning-heavy tasks — math, logic, planning, tool use.",
    " Feel free to throw a multi-step problem at me.",
    " I'll think step-by-step when the question calls for it.",
]

# Paraphrasers for each of the 10 seed prompts — simple, deterministic,
# covers casing / punctuation / added context / polite forms.
IDENTITY_PARAPHRASE_SUFFIXES = [
    "", "?", "!", ".", " please", " please?", "...",
    " — curious", " (quick question)", ", if you don't mind me asking",
    " — just checking", " in one sentence", ", briefly", ", in detail",
    ", be honest",
]
IDENTITY_PARAPHRASE_PREFIXES = [
    "",
    "Hey — ",
    "Quick question: ",
    "Sorry to bother, ",
    "I'm curious — ",
    "So, ",
]

# ─── Eval ───────────────────────────────────────────────────────────────
EVAL_SIZE = 500                # held-out for during-training eval
EVAL_GEN_REASONING = 20        # generation samples at step 03
EVAL_GEN_AGENT = 20
EVAL_GEN_IDENTITY = 10
GSM8K_SAMPLES = 100            # optional benchmark

# ─── Runtime safety ─────────────────────────────────────────────────────
VRAM_SOFT_LIMIT_GB = 42.0      # reserved VRAM threshold at which we warn
