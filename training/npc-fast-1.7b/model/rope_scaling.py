"""
YaRN RoPE configuration for context extension.

SmolLM2-1.7B-Instruct ships with a 8K-token RoPE base. We apply a YaRN
scaling factor to project attention onto a 128K context during continual
pre-training. The hyperparameters below are the defaults reported by the
original YaRN paper (Peng et al., 2023) for large factors.
"""

from __future__ import annotations

from config import (
    ORIGINAL_MAX_POSITION_EMBEDDINGS,
    TARGET_MAX_POSITION_EMBEDDINGS,
    YARN_FACTOR,
)


def yarn_config() -> dict:
    return {
        "type": "yarn",
        "factor": YARN_FACTOR,
        "original_max_position_embeddings": ORIGINAL_MAX_POSITION_EMBEDDINGS,
        "beta_fast": 32,
        "beta_slow": 1,
    }


def target_max_position_embeddings() -> int:
    return TARGET_MAX_POSITION_EMBEDDINGS
