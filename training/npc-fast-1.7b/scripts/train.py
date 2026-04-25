"""
NPC Fast training entrypoint.

Full-weight continual pre-training of SmolLM2-1.7B-Instruct with 128K YaRN
RoPE, a 5-stage context-length curriculum, flash-attention-2, bf16, and
gradient checkpointing. Single H200 SXM — no DeepSpeed, no PEFT.

Runnable directly or via ``main.py train``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from config import (
    BASE_MODEL,
    CHECKPOINT_DIR,
    CURRICULUM,
    DATASETS_REGISTRY,
    TRAINING_CONFIG,
    stage_for_step,
    wandb_key,
)
from data.curriculum import CurriculumCallback
from data.loader import load_registry
from data.mixer import mix
from data.preprocessing import (
    DynamicPadCollator,
    PackedDataset,
    tokenize_examples,
)
from model.setup import build_model_and_tokenizer

LOG = logging.getLogger("npc-fast.train")


import transformers as _hf_tr


class EarlyStopAtStep(_hf_tr.TrainerCallback):
    """Stops training when global_step reaches target_step.

    Recovery for the curriculum/num_train_epochs interaction bug: HF's
    Trainer locks ``num_train_epochs = ceil(max_steps/steps_per_epoch)`` at
    startup using the *stage-1* dataloader. When CurriculumCallback later
    enlarges max_seq_length (= fewer batches per epoch), the outer epoch
    loop hits its ceiling before max_steps. We set max_steps=-1 +
    num_train_epochs=huge so the outer loop never exits, and use this
    callback as the real stopping rule.
    """

    def __init__(self, target_step: int):
        self.target_step = int(target_step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.target_step:
            LOG.info(
                "[early-stop] global_step=%d reached target=%d — stopping.",
                state.global_step, self.target_step,
            )
            control.should_training_stop = True
        return control


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _maybe_init_wandb() -> None:
    if wandb_key() is None:
        LOG.info("WANDB_API_KEY not set — wandb reporting disabled.")
        os.environ["WANDB_DISABLED"] = "true"
        return
    import wandb
    wandb.init(
        project=TRAINING_CONFIG["wandb_project"],
        name=TRAINING_CONFIG["wandb_run_name"],
        config=TRAINING_CONFIG,
        reinit=True,
    )


def _build_datasets(tokenizer):
    LOG.info("Loading dataset registry: %s", DATASETS_REGISTRY)
    raw = load_registry(DATASETS_REGISTRY)
    train_ex, val_ex = mix(
        raw,
        seed=TRAINING_CONFIG["seed"],
        val_split=TRAINING_CONFIG["val_split"],
    )
    LOG.info("Tokenizing train split...")
    train_tok = tokenize_examples(train_ex, tokenizer)
    LOG.info("Tokenizing val split...")
    val_tok = tokenize_examples(val_ex, tokenizer)

    # Seed packing with stage-1 context; CurriculumCallback will update it.
    initial_seq = CURRICULUM[0]["max_seq_length"]
    train_ds = PackedDataset(train_tok, initial_seq)
    val_ds = PackedDataset(val_tok, initial_seq)
    return train_ds, val_ds


def _training_arguments(override_lr: float | None, override_max_steps: int | None, target_step: int | None = None) -> TrainingArguments:
    s0 = CURRICULUM[0]
    args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        max_steps=override_max_steps or TRAINING_CONFIG["max_steps"],
        per_device_train_batch_size=s0["micro_batch"],
        gradient_accumulation_steps=s0["grad_accum"],
        learning_rate=override_lr or TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        lr_scheduler_kwargs={"num_cycles": TRAINING_CONFIG["num_cycles"]},
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        bf16=TRAINING_CONFIG["bf16"],
        tf32=TRAINING_CONFIG["tf32"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        # Mid-training eval disabled: liger-kernels fused_linear_cross_entropy
        # only activates in model.training=True mode. During Trainer.evaluate()
        # the model is put in .eval() and liger falls back to HFs default loss
        # which materialises a full [B, T, V] fp32 logits tensor -> OOMs at
        # >=16K context on 80GB. We rely on the post-training eval suite
        # (eval/*.py) which controls memory explicitly.
        eval_strategy="no",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        report_to=[TRAINING_CONFIG["report_to"]] if wandb_key() else ["none"],
        run_name=TRAINING_CONFIG["wandb_run_name"],
        seed=TRAINING_CONFIG["seed"],
        dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
        optim=TRAINING_CONFIG["optim"],
        save_safetensors=True,
        remove_unused_columns=False,
    )
    if target_step is not None:
        # CURRICULUM-RESUME MODE: HF's _inner_training_loop computes
        # num_train_epochs = ceil(max_steps / steps_per_epoch) at startup
        # using the *initial* dataloader. When CurriculumCallback later
        # shrinks steps_per_epoch (longer context = fewer batches/epoch),
        # the outer epoch loop exits before max_steps is reached. The fix:
        # run with max_steps=-1 + huge num_train_epochs so the outer loop
        # never exits, and rely on EarlyStopAtStep(target_step) to stop us.
        args.max_steps = -1
        args.num_train_epochs = 10_000
        args.warmup_steps = 0  # we are far past warmup on resume
    return args


def run(
    *,
    model_name: str = BASE_MODEL,
    resume_from_checkpoint: str | None = None,
    output_dir: str | None = None,
    learning_rate: float | None = None,
    max_steps: int | None = None,
    target_step: int | None = None,
) -> None:
    _setup_logging()
    set_seed(TRAINING_CONFIG["seed"])
    _maybe_init_wandb()

    if torch.cuda.is_available():
        LOG.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        LOG.warning("CUDA unavailable — training on CPU will be unreasonably slow.")

    model, tokenizer = build_model_and_tokenizer(
        model_name,
        use_flash_attention_2=TRAINING_CONFIG["use_flash_attention_2"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
    )

    train_ds, val_ds = _build_datasets(tokenizer)

    # ---- Curriculum-aware resume ----
    # If resuming, pre-apply the stage that matches the checkpoint's
    # global_step BEFORE the Trainer builds its dataloader. Otherwise the
    # Trainer caches num_update_steps_per_epoch for stage 1, then the
    # CurriculumCallback re-packs the dataset on_train_begin and the cached
    # sampler indices fall outside the new packed length -> epoch_iterator
    # yields zero samples -> "stopping training at step N" false-positive.
    resume_stage = None
    if resume_from_checkpoint:
        try:
            import json as _json
            ts = _json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text())
            resume_step = int(ts.get("global_step", 0))
            resume_stage = stage_for_step(resume_step)
            LOG.info(
                "[resume] checkpoint at step %d -> stage %d (max_seq=%d, micro=%d, ga=%d). "
                "Pre-applying to dataset and TrainingArguments.",
                resume_step, resume_stage["stage"],
                resume_stage["max_seq_length"], resume_stage["micro_batch"],
                resume_stage["grad_accum"],
            )
            train_ds.set_max_seq_length(resume_stage["max_seq_length"])
            val_ds.set_max_seq_length(resume_stage["max_seq_length"])
        except Exception as e:  # noqa: BLE001
            LOG.warning("[resume] could not pre-apply stage: %s", e)

    args = _training_arguments(learning_rate, max_steps, target_step=target_step)
    if resume_stage is not None:
        args.per_device_train_batch_size = int(resume_stage["micro_batch"])
        args.gradient_accumulation_steps = int(resume_stage["grad_accum"])
    if output_dir:
        args.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    collator = DynamicPadCollator(pad_token_id=tokenizer.pad_token_id)
    curriculum_cb = CurriculumCallback(dataset=train_ds)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[curriculum_cb] + ([EarlyStopAtStep(target_step)] if target_step else []),
    )
    # Wire callback → trainer so it can rebuild dataloaders at stage changes.
    curriculum_cb.trainer = trainer

    LOG.info("Starting trainer.train()...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = Path(args.output_dir) / "final"
    LOG.info("Saving final model to %s", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    LOG.info("Training complete. Final model: %s", final_dir)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=BASE_MODEL)
    ap.add_argument("--output_dir", default=str(CHECKPOINT_DIR))
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--target_step", type=int, default=None,
                    help="Stop when global_step reaches this. Use with curriculum-resume; pairs with max_steps=-1 internally.")
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--resume_from_checkpoint", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(
        model_name=a.model_name,
        resume_from_checkpoint=a.resume_from_checkpoint,
        output_dir=a.output_dir,
        learning_rate=a.learning_rate,
        max_steps=a.max_steps,
        target_step=a.target_step,
    )
