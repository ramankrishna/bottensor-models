# shared/

Cross-model utilities. Currently a placeholder — each model dir is
self-contained today.

Planned migrations into this dir as the family grows:

- `utils/hf_push.py` — common HF Hub upload + model card generator
- `utils/gptq.py`    — `llm-compressor` wrapper with shared calibration
- `utils/gguf.py`    — `llama.cpp` convert + quantize wrapper
- `utils/eval/`      — shared eval harnesses (GSM8K, MATH, identity)
- `utils/sft.py`     — char-offset label-masking helper for chat templates
                        without `{% generation %}` markers (Qwen, Llama-3)
