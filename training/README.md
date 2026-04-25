# training/

One subdirectory per model. Each subdirectory is self-contained — its own
README, configs, scripts, and (where applicable) `requirements.txt`.

| Dir                    | Model                | Method                       | HW       |
|------------------------|----------------------|------------------------------|----------|
| `npc-fast-1.7b/`       | NPC Fast 1.7B        | Full-weight CPT (no PEFT)    | H200 SXM |
| `npc-fin-32b/`         | NPC Fin 32B          | QLoRA SFT (finance)          | A40      |
| `npc-mom-router/`      | NPC MoM (FastAPI)    | Code only — routing gateway  | CPU      |
| `npc-agentic-7b-v1/`   | NPC Agentic 7B v1    | QLoRA SFT (reasoning)        | A40      |
| `npc-agentic-7b-v2/`   | NPC Agentic 7B v2    | QLoRA SFT (v1 fixes baked in)| A40      |

See the top-level [README](../README.md) for the family overview and
shared tech stack.
