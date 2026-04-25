"""
Full-weight model + tokenizer initialization.

This is the canonical NPC Fast load path. It:
  * loads SmolLM2-1.7B-Instruct in bf16 with Flash Attention 2
  * applies the YaRN 128K RoPE config
  * ensures every parameter is trainable (full-weight continual pre-training)
  * enables gradient checkpointing so the 141GB H200 can run stage-5 sequences
  * sets pad_token = eos_token on the tokenizer when missing
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL
from .rope_scaling import target_max_position_embeddings, yarn_config

LOG = logging.getLogger("npc-fast.model")


def build_model_and_tokenizer(
    model_name: str = BASE_MODEL,
    *,
    use_flash_attention_2: bool = True,
    gradient_checkpointing: bool = True,
    device: str = "cuda",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    LOG.info("Loading base model: %s", model_name)

    # ---- Liger-kernel fused kernels ----
    # Must be applied BEFORE the model is instantiated so the monkey-patches
    # take effect at class level. Critical for long-context training: replaces
    # HF's ForCausalLMLoss (which materializes a full [B,T,V] fp32 logits
    # tensor and OOMs at ≥16K context on 80GB H100s) with a fused triton
    # kernel that computes CE without materializing the logits.
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        apply_liger_kernel_to_llama(
            rope=True,
            cross_entropy=False,            # superseded by fused_linear_cross_entropy
            fused_linear_cross_entropy=True,
            rms_norm=True,
            swiglu=True,
        )
        LOG.info("Applied liger-kernel patches: rope, fused_linear_CE, rms_norm, swiglu")
    except ImportError:
        LOG.warning(
            "liger-kernel not installed — long-context stages will likely OOM. "
            "Install with: pip install 'liger-kernel==0.4.2'"
        )

    attn_impl = "flash_attention_2" if use_flash_attention_2 else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- YaRN 128K RoPE ----
    model.config.rope_scaling = yarn_config()
    model.config.max_position_embeddings = target_max_position_embeddings()
    LOG.info(
        "Applied YaRN: factor=%.1f, original=%d, target=%d",
        model.config.rope_scaling["factor"],
        model.config.rope_scaling["original_max_position_embeddings"],
        model.config.max_position_embeddings,
    )

    # ---- Full-weight training ----
    trainable = 0
    for p in model.parameters():
        p.requires_grad = True
        trainable += p.numel()
    LOG.info("Trainable parameters: %.2fB", trainable / 1e9)

    # ---- Gradient checkpointing ----
    if gradient_checkpointing:
        # non-reentrant plays nicer with flash-attn
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Required so activation tensors retain grad when the checkpoint
        # reruns forward in backward.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        LOG.info("Gradient checkpointing enabled (non-reentrant)")

    # Ensure the model's generation config respects the new max length
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = target_max_position_embeddings()

    model = model.to(device)
    return model, tokenizer
