"""
MATH benchmark helpers — answer extraction + equivalence check.

For the Hendrycks MATH benchmark we extract the final answer from a
``\\boxed{...}`` envelope and compare strings after a normalization
pass (strip whitespace, drop ``\\$`` / ``$`` wrappers, normalize a few
common LaTeX equivalents). This isn't sympy-level rigorous — it
matches MATH's own reference checker for ~95% of cases and is fast.

For full correctness on edge cases (mixed numbers, units, LaTeX
equivalents like ``\\dfrac`` vs ``\\frac``), upgrade to the
``minerva_math`` evaluator from the LM eval harness.
"""
from __future__ import annotations

import re


_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed_answer(text: str) -> str | None:
    """
    Return the contents of the LAST ``\\boxed{...}`` in ``text``, with
    one level of nested braces preserved (e.g. ``\\frac{1}{2}``).
    Returns ``None`` if no boxed expression is found.
    """
    if not text:
        return None
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


# ─────────────────────────────────────────────────────────────────────
# Light normalization
# ─────────────────────────────────────────────────────────────────────
_DOLLAR_WRAP = re.compile(r"^\$+|\$+$")
_LEFT_RIGHT = re.compile(r"\\(?:left|right)")
_DFRAC      = re.compile(r"\\dfrac")
_TFRAC      = re.compile(r"\\tfrac")
_DOUBLE_WS  = re.compile(r"\s+")


def normalize_math(s: str) -> str:
    """Lightweight normalization for MATH answer comparison."""
    if s is None:
        return ""
    s = s.strip()
    s = _DOLLAR_WRAP.sub("", s)
    s = _LEFT_RIGHT.sub("", s)
    s = _DFRAC.sub(r"\\frac", s)
    s = _TFRAC.sub(r"\\frac", s)
    s = s.replace(" ", "")
    # Trailing punctuation from sentence-ending answers
    s = s.rstrip(".,;:!?")
    return s


def is_math_correct(prediction: str, gold: str) -> bool:
    """
    Check whether ``prediction`` matches ``gold`` after normalization.
    Both arguments should be the contents of a ``\\boxed{...}`` (i.e.
    the output of :func:`extract_boxed_answer`).
    """
    return normalize_math(prediction) == normalize_math(gold)
