"""
Curriculum learning: progressive context length.

``CurriculumCallback`` is a HuggingFace ``TrainerCallback`` that inspects the
global step at the start of every epoch/step and, when we cross a stage
boundary, rewires:

  * ``dataset.set_max_seq_length(new_len)`` — triggers re-packing
  * ``trainer.args.per_device_train_batch_size`` — micro batch
  * ``trainer.args.gradient_accumulation_steps`` — preserve effective batch
  * wandb log line for the transition

The effective batch (``micro_batch × grad_accum``) is held constant across
stages so the optimizer sees a stable update magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from config import CURRICULUM, stage_for_step

LOG = logging.getLogger("npc-fast.curriculum")


@dataclass
class CurriculumCallback(TrainerCallback):
    """Flip context length + batch config at stage boundaries.

    The callback needs a reference to both the packed dataset (so it can
    change max_seq_length) and the trainer (so it can rebuild the
    dataloader). Both are wired in ``train.py`` after the Trainer exists.
    """

    dataset: object = None             # PackedDataset — set post-init
    trainer: object = None             # Trainer instance — set post-init
    _current_stage: Optional[int] = None

    def _apply_stage(self, stage: dict, args: TrainingArguments) -> None:
        if self.dataset is not None and hasattr(self.dataset, "set_max_seq_length"):
            self.dataset.set_max_seq_length(stage["max_seq_length"])
        args.per_device_train_batch_size = int(stage["micro_batch"])
        args.gradient_accumulation_steps = int(stage["grad_accum"])

        # Force the trainer to rebuild its dataloader with the new batch size
        # and re-packed dataset. Safe at step boundary.
        if self.trainer is not None and hasattr(self.trainer, "_get_train_sampler"):
            try:
                self.trainer.train_dataset = self.dataset
                # Clear cached dataloader if present
                if hasattr(self.trainer, "_train_dataloader"):
                    self.trainer._train_dataloader = None
            except Exception as e:  # noqa: BLE001
                LOG.warning("Could not refresh dataloader: %s", e)

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "curriculum/stage": stage["stage"],
                    "curriculum/max_seq_length": stage["max_seq_length"],
                    "curriculum/micro_batch": stage["micro_batch"],
                    "curriculum/grad_accum": stage["grad_accum"],
                })
        except Exception:
            pass

        LOG.info(
            "[curriculum] entered stage %d: max_seq=%d, micro=%d, grad_accum=%d",
            stage["stage"], stage["max_seq_length"],
            stage["micro_batch"], stage["grad_accum"],
        )
        self._current_stage = stage["stage"]

    # ---- TrainerCallback hooks ----
    def on_train_begin(self, args, state, control, **kwargs):
        stage = stage_for_step(state.global_step)
        self._apply_stage(stage, args)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        stage = stage_for_step(state.global_step)
        if stage["stage"] != self._current_stage:
            self._apply_stage(stage, args)
        return control


def describe_schedule() -> str:
    """Human-readable schedule dump used by `main.py status`."""
    lines = ["Curriculum schedule:"]
    for s in CURRICULUM:
        eff = s["micro_batch"] * s["grad_accum"]
        lines.append(
            f"  stage {s['stage']}: steps {s['start']:>5d}-{s['end']:>5d} | "
            f"ctx={s['max_seq_length']:>7d} | micro={s['micro_batch']} "
            f"grad_accum={s['grad_accum']} (eff {eff})"
        )
    return "\n".join(lines)
