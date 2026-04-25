"""
Multi-dataset weighted sampling + interleaving.

Steps:
  1. Bucket examples by source.
  2. Compute per-source target counts using the configured weights,
     capped by the actual available size of each source.
  3. Truncate each bucket to its target.
  4. Interleave round-robin (oldest-first when a bucket is exhausted).
  5. Deterministic shuffle with the provided seed.
  6. 95/5 train/val split.

Also logs the final composition, e.g.:
    "Dataset composition: 60% datagen, 24% hermes, 16% claude-opus"
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Iterable

from .dedup import dedupe
from .loader import NormalizedExample

LOG = logging.getLogger("npc-fast.mixer")


def _source_short(src: str) -> str:
    """Human-friendly short name for composition logging."""
    tail = src.split("/")[-1]
    return tail.split("::")[0]


def _interleave(buckets: list[list[NormalizedExample]]) -> list[NormalizedExample]:
    pointers = [0] * len(buckets)
    out: list[NormalizedExample] = []
    total = sum(len(b) for b in buckets)
    while len(out) < total:
        for i, bucket in enumerate(buckets):
            if pointers[i] < len(bucket):
                out.append(bucket[pointers[i]])
                pointers[i] += 1
    return out


def mix(
    examples: Iterable[NormalizedExample],
    *,
    seed: int = 42,
    val_split: float = 0.05,
) -> tuple[list[NormalizedExample], list[NormalizedExample]]:
    all_examples = list(examples)
    if not all_examples:
        raise ValueError("No examples to mix — did the loader return anything?")

    # 1. Dedupe before mixing so weights apply to unique content.
    all_examples = dedupe(all_examples)

    # 2. Bucket by source.
    by_src: dict[str, list[NormalizedExample]] = defaultdict(list)
    for ex in all_examples:
        by_src[ex.source].append(ex)

    # 3. Compute per-source targets from weights.
    #    Normalize weights so they sum to 1.0 across present sources.
    src_weights = {src: max(0.0, b[0].weight) for src, b in by_src.items() if b}
    total_w = sum(src_weights.values())
    if total_w <= 0:
        raise ValueError("All dataset weights are zero.")
    src_weights = {k: v / total_w for k, v in src_weights.items()}

    # Target total examples: cap by the largest honorable sample count so we
    # don't upsample any source beyond what the weight can support.
    present_sizes = {src: len(b) for src, b in by_src.items()}
    # Compute the biggest total that still respects every weight.
    candidate_totals = [
        present_sizes[src] / src_weights[src]
        for src in src_weights
        if src_weights[src] > 0
    ]
    target_total = int(min(candidate_totals)) if candidate_totals else 0

    buckets: list[list[NormalizedExample]] = []
    for src, bucket in by_src.items():
        cap = int(round(target_total * src_weights[src]))
        rng = random.Random(seed + hash(src) % (2**31))
        rng.shuffle(bucket)
        buckets.append(bucket[:cap])

    # 4. Interleave to avoid catastrophic in-batch homogeneity.
    interleaved = _interleave(buckets)

    # 5. Deterministic shuffle for the Trainer's sampler.
    rng = random.Random(seed)
    rng.shuffle(interleaved)

    # 6. Log composition.
    counts: dict[str, int] = defaultdict(int)
    for ex in interleaved:
        counts[_source_short(ex.source)] += 1
    total = max(1, len(interleaved))
    pieces = [
        f"{pct:.0f}% {name}"
        for name, pct in sorted(
            ((n, 100.0 * c / total) for n, c in counts.items()),
            key=lambda t: -t[1],
        )
    ]
    LOG.info("Dataset composition: %s  (n=%d)", ", ".join(pieces), total)

    # 7. Train/val split.
    split_idx = max(1, int(len(interleaved) * (1.0 - val_split)))
    train = interleaved[:split_idx]
    val = interleaved[split_idx:]
    LOG.info("Split: train=%d, val=%d", len(train), len(val))
    return train, val
