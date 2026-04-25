"""
Hash-based deduplication on the concatenated user content.

Training on duplicate prompts both wastes compute and biases eval splits.
We key on a normalized form of every ``user`` message in an example (lowercased,
whitespace collapsed) and drop later occurrences.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable

from .loader import NormalizedExample

LOG = logging.getLogger("npc-fast.dedup")

_WS_RE = re.compile(r"\s+")


def _user_key(example: NormalizedExample) -> str:
    parts = [
        _WS_RE.sub(" ", m["content"].strip().lower())
        for m in example.messages
        if m["role"] == "user"
    ]
    blob = "\n".join(parts)
    return hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()


def dedupe(examples: Iterable[NormalizedExample]) -> list[NormalizedExample]:
    seen: set[str] = set()
    out: list[NormalizedExample] = []
    collisions = 0
    for ex in examples:
        k = _user_key(ex)
        if k in seen:
            collisions += 1
            continue
        seen.add(k)
        out.append(ex)
    LOG.info("Deduped %d -> %d (%d duplicates removed)", len(out) + collisions, len(out), collisions)
    return out
