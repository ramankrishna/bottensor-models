"""Shared eval harnesses for the NPC family."""

from .gsm8k import (
    extract_gsm_answer,
    load_gsm8k_samples,
    eval_gsm8k,
)
from .identity import (
    score_identity,
    DEFAULT_IDENTITY_KEYWORDS,
)
from .math_bench import (
    extract_boxed_answer,
    is_math_correct,
)

__all__ = [
    # gsm8k
    "extract_gsm_answer",
    "load_gsm8k_samples",
    "eval_gsm8k",
    # identity
    "score_identity",
    "DEFAULT_IDENTITY_KEYWORDS",
    # math
    "extract_boxed_answer",
    "is_math_correct",
]
