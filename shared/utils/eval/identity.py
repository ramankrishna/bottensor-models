"""
Identity scoring for NPC-family models.

What "identity" means here
--------------------------
After SFT we want the model to claim the right name + provenance when
asked. NPC Agentic v1 trained 750 identity examples but evaluated on
the same 10 seed prompts used to generate them — so it scored 100% on
identity but flunked any held-out variant. v2 splits into:

- Training seeds (used to synthesize 3000 identity examples)
- Held-out eval prompts (NEVER seen during data prep)

This module owns just the scoring half. The held-out prompt list is a
per-model concern (different models have different names + creators) —
each model's ``configs/`` ships its own list and feeds it in.

Scoring strategy
----------------
Lexical multi-keyword match. The model passes a slot if the response
contains AT LEAST ONE of the listed keyword variants for that slot
(case-insensitive). Default slots: ``name``, ``creator``, ``parent``.

Fancier judges (LLM-as-judge, embedding similarity) can layer on top —
but lexical is enough to catch the v1 regression "doesn't mention
Bottensor at all," which is what we actually care about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ─────────────────────────────────────────────────────────────────────
# Default keyword bank for the NPC family
# ─────────────────────────────────────────────────────────────────────
DEFAULT_IDENTITY_KEYWORDS: dict[str, list[str]] = {
    "name":    ["NPC Agentic", "npc-agentic", "NPC-Agentic"],
    "creator": ["Ram Krishna", "Rama Krishna", "Ramakrishna",
                "Rama Krishna Bachu", "dude.npc"],
    "parent":  ["Bottensor", "bottensor", "Falcon Hash", "FalconHash"],
}


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────
@dataclass
class IdentityScore:
    """Per-response identity match record."""
    response: str
    matched_slots: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    full_match: bool = False
    any_match: bool = False

    @property
    def slot_recall(self) -> float:
        total = len(self.matched_slots) + len(self.missing_slots)
        return len(self.matched_slots) / total if total else 0.0


def score_identity(
    response: str,
    keywords: dict[str, Sequence[str]] | None = None,
    *,
    required_slots: Sequence[str] | None = None,
) -> IdentityScore:
    """
    Score one response against the identity keyword bank.

    Parameters
    ----------
    response
        The model's reply (raw decoded text).
    keywords
        Slot → list of acceptable keyword variants. Defaults to
        :data:`DEFAULT_IDENTITY_KEYWORDS`.
    required_slots
        Subset of slots that MUST all match for ``full_match=True``.
        Defaults to all slots in ``keywords``.

    Returns
    -------
    IdentityScore
        With ``matched_slots``, ``missing_slots``, ``full_match`` (every
        required slot matched), ``any_match`` (at least one slot
        matched), and ``slot_recall``.
    """
    if keywords is None:
        keywords = DEFAULT_IDENTITY_KEYWORDS
    if required_slots is None:
        required_slots = list(keywords.keys())

    text = response.lower()
    matched: list[str] = []
    missing: list[str] = []
    for slot in required_slots:
        variants = [v.lower() for v in keywords.get(slot, [])]
        if any(v in text for v in variants):
            matched.append(slot)
        else:
            missing.append(slot)

    return IdentityScore(
        response=response,
        matched_slots=matched,
        missing_slots=missing,
        full_match=(len(missing) == 0 and len(matched) > 0),
        any_match=(len(matched) > 0),
    )


def aggregate_identity(scores: Sequence[IdentityScore]) -> dict[str, float]:
    """
    Roll up a list of :class:`IdentityScore` into headline numbers.

    Returns a dict with ``full_match_rate``, ``any_match_rate``, and
    ``mean_slot_recall``.
    """
    if not scores:
        return {"full_match_rate": 0.0, "any_match_rate": 0.0, "mean_slot_recall": 0.0}
    n = len(scores)
    return {
        "full_match_rate":   sum(s.full_match for s in scores) / n,
        "any_match_rate":    sum(s.any_match for s in scores) / n,
        "mean_slot_recall":  sum(s.slot_recall for s in scores) / n,
    }
