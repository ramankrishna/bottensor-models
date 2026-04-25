"""Cross-model helpers shared across the NPC family."""

from .sft_masking import (
    build_assistant_spans,
    preprocess_example,
    sanity_check_mask_share,
)

__all__ = [
    "build_assistant_spans",
    "preprocess_example",
    "sanity_check_mask_share",
]
