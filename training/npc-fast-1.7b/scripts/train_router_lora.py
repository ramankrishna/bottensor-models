"""Quick LoRA SFT for router task. Fix: use string chat-template then tokenize."""
from __future__ import annotations
import json, logging
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOG = logging.getLogger("router-sft")

BASE = "/workspace/npc-fast-trainer/output/checkpoints/final"
DATA = "/workspace/npc-fast-trainer/data/router_sft/train.jsonl"
OUT  = "/workspace/npc-fast-trainer/output/router_lora"
MAX  = 512

LOG.info("Loading tokenizer + model")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

examples = [json.loads(l) for l in open(DATA)]
LOG.info("Loaded %d raw examples", len(examples))

def encode_one(messages):
    # String-level templates, then .encode() to get plain lists of ints
    full_str = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=False)
    prompt_str = tok.apply_chat_template(messages[:-1], tokenize=False,
                                          add_generation_prompt=True)
    full_ids = tok.encode(full_str, add_special_tokens=False)
    prompt_ids = tok.encode(prompt_str, add_special_tokens=False)
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    full_ids = full_ids[:MAX]
    labels = labels[:MAX]
    attention = [1] * len(full_ids)
    pad = MAX - len(full_ids)
    full_ids += [tok.pad_token_id] * pad
    labels += [-100] * pad
    attention += [0] * pad
    return {"input_ids": full_ids, "attention_mask": attention, "labels": labels}

encoded = [encode_one(ex["messages"]) for ex in examples]
LOG.info("Encoded %d examples. Sample lengths: %s",
         len(encoded), [sum(e["attention_mask"]) for e in encoded[:5]])

ds = Dataset.from_list(encoded)

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model(OUT)
tok.save_pretrained(OUT)
LOG.info("Saved LoRA adapter to %s", OUT)
