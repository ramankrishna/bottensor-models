"""
Utilities for saving full-weight checkpoints.

The HuggingFace Trainer already handles periodic saves via `save_steps`, but
we expose a helper for the final "merged" export so callers don't have to
know the on-disk layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger("npc-fast.save")


def save_full_checkpoint(model, tokenizer, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Saving full-weight model → %s", output_dir)
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(str(output_dir))
    LOG.info("Save complete.")
    return output_dir
