"""
Step 3 — QLoRA SFT of Qwen2.5-7B-Instruct on the prepared SFT dataset.

Uses Unsloth for 4-bit base loading + TRL 0.24's SFTTrainer. Loss is masked
on non-assistant tokens via a custom preprocessing step that computes each
token's character offset and only unmasks tokens whose span falls inside an
assistant turn.

Why not TRL's `DataCollatorForCompletionOnlyLM`? Removed in trl 0.24.
Why not `SFTConfig(assistant_only_loss=True)`? Requires `{% generation %}`
markers in the chat template, which Qwen2.5 doesn't ship.

Launch with `nohup python 02_train.py > logs/train.log 2>&1 &`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Unsloth patches torch/transformers — must be imported BEFORE trl/transformers.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("WANDB_PROJECT", "npc-agentic")

import torch
from unsloth import FastLanguageModel  # must be first
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ────────────────────────────────────────────────────────────────────────
# Label masking — unmask only assistant-turn tokens
# ────────────────────────────────────────────────────────────────────────
def build_assistant_spans(
    tokenizer, messages: List[Dict[str, str]]
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Render the chat template, then identify character-offset spans that
    correspond to each assistant message's body (NOT the `<|im_start|>assistant\\n`
    header nor the `<|im_end|>` closer — those are template furniture and
    shouldn't be trained on either).

    Returns (rendered_text, list_of_(start_char, end_char)).
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )

    spans: List[Tuple[int, int]] = []
    cursor = 0
    header = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"

    for m in messages:
        if m["role"] != "assistant":
            continue
        # Find the next `<|im_start|>assistant\n` after cursor
        header_pos = rendered.find(header, cursor)
        if header_pos < 0:
            # Assistant message exists in messages but not in rendered output —
            # unexpected. Skip rather than miscount.
            continue
        body_start = header_pos + len(header)
        body_end = rendered.find(end_marker, body_start)
        if body_end < 0:
            # Unterminated — use end of string
            body_end = len(rendered)
        spans.append((body_start, body_end))
        cursor = body_end + len(end_marker)

    return rendered, spans


def preprocess_example(
    tokenizer, example: Dict[str, Any], max_len: int
) -> Dict[str, List[int]]:
    """
    Tokenize one example, producing input_ids / attention_mask / labels.
    `labels` is set to -100 for every token whose character span doesn't
    fall inside any assistant-body region.
    """
    rendered, spans = build_assistant_spans(tokenizer, example["messages"])

    enc = tokenizer(
        rendered,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]

    labels = [-100] * len(input_ids)
    # A token is in-span if its offset range intersects any assistant span.
    # Most tokenizers produce (start, end) with end > start; some special
    # tokens give (0, 0) which we treat as not-in-span.
    for i, (ts, te) in enumerate(offsets):
        if ts == te:
            continue
        for sa, sb in spans:
            # intersection test
            if te > sa and ts < sb:
                labels[i] = input_ids[i]
                break

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }


# ────────────────────────────────────────────────────────────────────────
# WandB setup
# ────────────────────────────────────────────────────────────────────────
def configure_wandb() -> str:
    netrc = Path.home() / ".netrc"
    if netrc.exists() or os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb.init(
                project=config.WANDB_PROJECT,
                name=config.WANDB_RUN_NAME,
                config={
                    "base_model": config.BASE_MODEL,
                    "max_seq_len": config.MAX_SEQ_LEN,
                    "lora_rank": config.LORA_RANK,
                    "lora_alpha": config.LORA_ALPHA,
                    "effective_batch_size": config.PER_DEVICE_BATCH_SIZE * config.GRAD_ACCUM_STEPS,
                    "lr": config.LEARNING_RATE,
                    "epochs": config.NUM_EPOCHS,
                },
                resume="allow",
            )
            log(f"  WandB initialized: {wandb.run.url}")
            return "wandb"
        except Exception as exc:
            log(f"  WandB init failed ({exc}) — falling back to report_to=none")
    return "none"


# ────────────────────────────────────────────────────────────────────────
# Callbacks
# ────────────────────────────────────────────────────────────────────────
class VRAMMonitor(TrainerCallback):
    """Log VRAM stats every 100 steps. Flag over soft-limit."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 != 0 or state.global_step == 0:
            return
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_reserved() / 1e9
        prefix = "  VRAM"
        if reserved > config.VRAM_SOFT_LIMIT_GB:
            prefix = "  VRAM ⚠"
        log(f"{prefix} step={state.global_step}  alloc={alloc:.1f}GB  "
            f"reserved={reserved:.1f}GB  peak={peak:.1f}GB")


class EarlyStopIfEvalRises(TrainerCallback):
    """Stop training after eval loss increases for 2 consecutive evaluations."""
    def __init__(self):
        self.history = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        self.history.append(eval_loss)
        if len(self.history) >= 3:
            if self.history[-1] > self.history[-2] > self.history[-3]:
                log(f"  EARLY STOP: eval_loss rose for 2 consecutive evals: "
                    f"{self.history[-3]:.4f} → {self.history[-2]:.4f} → {self.history[-1]:.4f}")
                control.should_training_stop = True


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    log("=" * 64)
    log("NPC Agentic 7B — QLoRA SFT")
    log("=" * 64)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — cannot train")
    prop = torch.cuda.get_device_properties(0)
    log(f"  GPU: {prop.name}  VRAM: {prop.total_memory/1e9:.1f} GB  sm_{prop.major}{prop.minor}")

    # ── Load base model in 4-bit via Unsloth ───────────────────────────
    log("")
    log("== Loading Qwen2.5-7B-Instruct in 4-bit (Unsloth) ==")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL,
        max_seq_length=config.MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    if tokenizer.chat_template is None:
        raise RuntimeError("Qwen2.5 tokenizer arrived without chat_template — aborting.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    log(f"  vocab_size: {tokenizer.vocab_size}")
    log(f"  pad_token_id: {tokenizer.pad_token_id}  eos_token_id: {tokenizer.eos_token_id}")

    # ── Apply LoRA ─────────────────────────────────────────────────────
    log("")
    log(f"== Applying LoRA r={config.LORA_RANK}  α={config.LORA_ALPHA} ==")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.RANDOM_SEED,
    )
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    log(f"  trainable params: {trainable/1e6:.1f}M / total {total/1e9:.2f}B  "
        f"({100*trainable/total:.2f}%)")

    # ── Load datasets ──────────────────────────────────────────────────
    log("")
    log("== Loading dataset ==")
    train_ds = load_dataset("json", data_files=str(config.DATA_PROCESSED), split="train")
    eval_ds = load_dataset("json", data_files=str(config.EVAL_SPLIT), split="train")
    log(f"  train: {len(train_ds):,}  eval: {len(eval_ds):,}")

    # ── Preprocess (tokenize + compute masked labels) ──────────────────
    log("")
    log("== Preprocessing: tokenize + label-mask ==")
    t0 = time.time()

    def _map_fn(example):
        return preprocess_example(tokenizer, example, config.MAX_SEQ_LEN)

    drop_cols = [c for c in train_ds.column_names if c not in ("input_ids", "attention_mask", "labels")]
    train_ds = train_ds.map(
        _map_fn, remove_columns=drop_cols,
        num_proc=4, desc="tokenize/mask train",
    )
    eval_ds = eval_ds.map(
        _map_fn, remove_columns=[c for c in eval_ds.column_names if c not in ("input_ids","attention_mask","labels")],
        num_proc=4, desc="tokenize/mask eval",
    )
    log(f"  preprocess wall-clock: {(time.time()-t0)/60:.1f} min")
    log(f"  train rows after: {len(train_ds):,}  eval rows after: {len(eval_ds):,}")

    # Sanity: fraction of tokens unmasked on a few samples
    import random as _rand
    _rand.seed(0)
    samples = [train_ds[i] for i in _rand.sample(range(len(train_ds)), min(20, len(train_ds)))]
    total_tok = sum(len(s["input_ids"]) for s in samples)
    unmasked = sum(sum(1 for x in s["labels"] if x != -100) for s in samples)
    log(f"  loss-mask sanity: {unmasked:,} unmasked / {total_tok:,} total "
        f"({100*unmasked/max(1,total_tok):.1f}% assistant-token share)")

    # ── SFTConfig (replaces TrainingArguments in TRL 0.24) ─────────────
    report_to = configure_wandb()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(config.CHECKPOINT_DIR),
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim=config.OPTIMIZER,
        bf16=config.BF16,
        fp16=False,
        gradient_checkpointing=config.GRAD_CHECKPOINTING,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH_SIZE,
        load_best_model_at_end=False,
        report_to=report_to,
        run_name=config.WANDB_RUN_NAME,
        logging_dir=str(config.LOG_DIR),
        seed=config.RANDOM_SEED,
        data_seed=config.RANDOM_SEED,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        # SFTConfig-specific:
        max_length=config.MAX_SEQ_LEN,
        packing=False,
        completion_only_loss=False,      # we handled masking manually
        assistant_only_loss=False,       # ditto
        dataset_kwargs={"skip_prepare_dataset": True},  # we pre-processed
    )

    # ── Trainer ────────────────────────────────────────────────────────
    # DataCollatorForSeq2Seq pads input_ids AND labels to the same length,
    # filling labels with -100 so loss ignores padding.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        args=sft_args,
        callbacks=[VRAMMonitor(), EarlyStopIfEvalRises()],
    )

    # ── Auto-resume from latest checkpoint if one exists ───────────────
    resume = False
    ckpt_dir = Path(str(config.CHECKPOINT_DIR))
    if ckpt_dir.exists():
        ckpts = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if ckpts:
            latest = ckpts[-1]
            log(f"== Resuming from {latest} ==")
            resume = str(latest)

    # ── Go ─────────────────────────────────────────────────────────────
    log("")
    log("== Starting training ==")
    log(f"  effective batch size: {config.PER_DEVICE_BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    log(f"  total optimizer steps (estimated): "
        f"{(len(train_ds) * config.NUM_EPOCHS) // (config.PER_DEVICE_BATCH_SIZE * config.GRAD_ACCUM_STEPS):,}")
    if resume:
        log(f"  RESUMING from {resume}")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume if resume else None)
    elapsed = time.time() - t0
    log(f"  training wall-clock: {elapsed/3600:.2f}h")

    # ── Save final adapter ─────────────────────────────────────────────
    log("")
    log("== Saving final adapter ==")
    config.FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.FINAL_ADAPTER_DIR))

    # v2 fix: write generation_config.json with anti-degeneracy defaults so
    # downstream users (vLLM, transformers, llama.cpp) inherit sane sampling
    # and we don't have to rely on callers setting repetition_penalty themselves.
    gen_defaults = getattr(config, "GENERATION_DEFAULTS", None)
    if gen_defaults:
        gen_cfg_path = config.FINAL_ADAPTER_DIR / "generation_config.json"
        # Preserve base model's existing generation_config fields (eos/pad/bos ids)
        base_gen: Dict[str, Any] = {}
        if gen_cfg_path.exists():
            try:
                base_gen = json.loads(gen_cfg_path.read_text())
            except Exception:
                base_gen = {}
        # Merge: our defaults override but preserve special tokens from base
        merged_gen = {**base_gen, **gen_defaults}
        # Keep transformers_version from base if present
        if "transformers_version" in base_gen:
            merged_gen["transformers_version"] = base_gen["transformers_version"]
        gen_cfg_path.write_text(json.dumps(merged_gen, indent=2) + "\n")
        log(f"  wrote generation_config.json with defaults: {list(gen_defaults)}")

    log(f"  saved → {config.FINAL_ADAPTER_DIR}")

    # ── Summary ────────────────────────────────────────────────────────
    last_eval = None
    for entry in reversed(trainer.state.log_history):
        if "eval_loss" in entry:
            last_eval = entry["eval_loss"]
            break

    summary = {
        "final_train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "final_eval_loss": last_eval,
        "total_steps": trainer.state.global_step,
        "wall_clock_seconds": elapsed,
        "wall_clock_hours": elapsed / 3600,
        "peak_vram_gb": torch.cuda.max_memory_reserved() / 1e9,
        "wandb_run_url": None,
        "trainable_params": trainable,
        "base_model": config.BASE_MODEL,
    }
    try:
        import wandb
        if wandb.run is not None:
            summary["wandb_run_url"] = wandb.run.url
            wandb.finish()
    except Exception:
        pass

    summary_path = config.LOG_DIR / "train_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log(f"  summary → {summary_path}")
    log("")
    log("=" * 64)
    log("TRAINING DONE.")
    log(f"  final train loss: {summary['final_train_loss']:.4f}")
    log(f"  final eval loss:  {summary['final_eval_loss']}")
    log(f"  wall-clock:       {summary['wall_clock_hours']:.2f}h")
    log(f"  peak VRAM:        {summary['peak_vram_gb']:.1f} GB")
    log(f"  wandb:            {summary['wandb_run_url']}")
    log(f"  adapter saved at: {config.FINAL_ADAPTER_DIR}")
    log("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
