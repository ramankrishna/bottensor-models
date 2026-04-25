# bottensor-models

Training scripts and configs for the **NPC Model Family** by
**Bottensor** (a Falcon Hash company).

This repo is the source-of-truth for how each NPC model was built — the
data prep, training loop, evaluation, and export pipeline. It does **not**
contain training data, checkpoints, or model weights; those live on
HuggingFace under [`ramankrishna10/*`](https://huggingface.co/ramankrishna10).

## Model family

| Model              | Base                          | Method                    | Role                               | Status        |
|--------------------|-------------------------------|---------------------------|------------------------------------|---------------|
| **NPC Fast 1.7B**  | SmolLM2-1.7B-Instruct         | Full-weight CPT, 128K YaRN | Fast agentic router (`self` vs escalate) | Shipped       |
| **NPC Fin 32B**    | Qwen2.5-32B-Instruct          | QLoRA SFT (finance)       | Heavy finance reasoning            | Retired (legacy) |
| **NPC MoM Router** | n/a (FastAPI gateway)         | Code, not a model         | Routes traffic Fast → Fin          | Retired (replaced by direct vLLM) |
| **NPC Agentic 7B v1** | Qwen2.5-7B-Instruct        | QLoRA SFT (reasoning)     | General multi-step reasoning       | Privatized (quality issues) |
| **NPC Agentic 7B v2** | Qwen2.5-7B-Instruct        | QLoRA SFT (reasoning, v1 fixes baked in) | General multi-step reasoning   | Training (this branch) |

HuggingFace targets:

- `ramankrishna10/npc-fast-1.7b`, `-gptq`, `-gguf`
- `ramankrishna10/npc-agentic-7b` (+ `-merged`, `-gptq`, `-gguf`) — currently private until v2 ships

## Layout

```
bottensor-models/
├── training/
│   ├── npc-fast-1.7b/        # Full-weight CPT pipeline (H200)
│   ├── npc-fin-32b/          # README only — legacy reference
│   ├── npc-mom-router/       # FastAPI gateway (retired)
│   ├── npc-agentic-7b-v1/    # First reasoning run (lessons learned)
│   └── npc-agentic-7b-v2/    # Current reasoning run with v1 fixes
├── shared/
│   └── utils/                # (placeholder for future cross-model helpers)
├── LICENSE                   # Apache-2.0
├── .gitignore
├── .gitattributes
├── .env.example              # All env vars referenced across the repo
└── README.md
```

## Tech stack (across the family)

- **Frameworks:** PyTorch 2.10 (CUDA 12.8), Transformers 4.57+, TRL 0.24,
  PEFT 0.18, bitsandbytes 0.49, [Unsloth](https://github.com/unslothai/unsloth) for QLoRA loops
- **Adapters:** LoRA (rank 64, alpha 128) on q/k/v/o + gate/up/down projections
- **Quantization:** NF4 double-quant (training), GPTQ W4A16 via
  `llm-compressor` 0.10 (inference), GGUF Q4_K_M / Q5_K_M / Q8_0 via
  `llama.cpp`
- **Precision:** bf16 mixed precision throughout
- **Optimizer:** `adamw_8bit` (paged 8-bit) with cosine LR + warmup
- **Long context:** YaRN RoPE scaling (NPC Fast → 128K)
- **Serving:** vLLM 0.18+ with `--enable-auto-tool-choice --tool-call-parser hermes`
- **Tracking:** Weights & Biases (optional; auto-disables if no key set)

## Setup

```bash
git clone https://github.com/ramankrishna/bottensor-models.git
cd bottensor-models
cp .env.example .env                 # fill HF_TOKEN, WANDB_API_KEY, etc.
```

Each model dir is independent and has its own `requirements.txt`. Pick
the one you want to reproduce:

```bash
cd training/npc-fast-1.7b
pip install -r requirements.txt
bash scripts/run_train.sh
```

See each model's README for hardware, dataset mix, and runtime.

## Reproducibility notes

- Scripts contain hardcoded `/workspace/...` paths from the RunPod
  training environment. They're documented as-is so the runtime layout
  is preserved; adjust to your own paths before running.
- All secrets are loaded via `os.getenv()`. No tokens or keys are
  committed.
- Training data is **not** redistributed. Each script pulls source
  datasets from HF directly (HuggingFaceTB, openthoughts, etc.).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Author

**Rama Krishna Bachu** (`dude.npc`) — founder, Falcon Hash → Bottensor.
