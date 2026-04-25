# NPC Fin 32B

> **Status: retired (legacy reference).** Code not redistributed in this
> repo — listed for completeness of the model family.

NPC Fin was the **first** model in the family — a finance-focused
reasoning model trained on Qwen2.5-32B-Instruct with QLoRA SFT. It served
as the "heavy" leg behind NPC MoM router (NPC Fast handled fast routes,
NPC Fin handled deep finance reasoning).

## Specs

| Field           | Value                                             |
|-----------------|---------------------------------------------------|
| Base model      | `Qwen/Qwen2.5-32B-Instruct`                       |
| Method          | QLoRA (4-bit NF4 + bf16 LoRA adapters)            |
| LoRA            | r=64, alpha=128, dropout=0.05, all 7 modules      |
| Optimizer       | `adamw_8bit` (paged 8-bit), cosine LR + warmup    |
| Hardware        | RunPod A40 (48GB VRAM)                            |
| Framework       | Unsloth + TRL SFTTrainer                          |
| Domain          | Finance reasoning (synthetic + curated)           |
| HF target       | `ramankrishna10/npc-fin-32b-sft` (private)        |

## Why retired

The MoM-style architecture (router + heavy expert) was replaced by the
single-model **NPC Agentic 7B** generalist. NPC Fin scripts and weights
remain on the founder's archive but are not part of the active stack.

## Reference

Hyperparameters and the data pipeline are essentially the same as
[`npc-agentic-7b-v1/`](../npc-agentic-7b-v1/) with a finance-only dataset
mix. Read that dir for the live equivalent.
