"""Cross-model helpers shared across the NPC family."""

# ── SFT label masking ────────────────────────────────────────────────
from .sft import (
    build_assistant_spans,
    preprocess_example,
    sanity_check_mask_share,
    verify_eos_in_loss,
)

# ── HuggingFace push + cards ─────────────────────────────────────────
from .hf_push import (
    CARDS,
    DEFAULT_IGNORE,
    folder_size_gb,
    push_folder,
    render_card,
    stage_lora_adapter,
)

# ── GPTQ W4A16 ───────────────────────────────────────────────────────
from .gptq import (
    build_calib_dataset,
    quantize_w4a16,
    smoke_test as gptq_smoke_test,
)

# ── llama.cpp / GGUF ─────────────────────────────────────────────────
from .gguf import (
    DEFAULT_QUANTS,
    build_quants,
    convert_to_fp16,
    ensure_llama_cpp,
    ensure_python_deps,
    quantize_one,
    smoke_test as gguf_smoke_test,
)

# ── Eval harnesses ───────────────────────────────────────────────────
from . import eval as eval_harnesses  # noqa: F401  (sub-package import)

__all__ = [
    # sft
    "build_assistant_spans",
    "preprocess_example",
    "sanity_check_mask_share",
    "verify_eos_in_loss",
    # hf_push
    "CARDS",
    "DEFAULT_IGNORE",
    "folder_size_gb",
    "push_folder",
    "render_card",
    "stage_lora_adapter",
    # gptq
    "build_calib_dataset",
    "quantize_w4a16",
    "gptq_smoke_test",
    # gguf
    "DEFAULT_QUANTS",
    "build_quants",
    "convert_to_fp16",
    "ensure_llama_cpp",
    "ensure_python_deps",
    "quantize_one",
    "gguf_smoke_test",
    # eval sub-package
    "eval_harnesses",
]
