"""
Char-offset label masking for chat-template SFT.

Why this exists
---------------
TRL 0.24 dropped ``DataCollatorForCompletionOnlyLM``, and the Qwen2.5
chat template (also Llama-3, Gemma, etc.) does not contain the
``{% generation %}`` markers required by
``SFTConfig(assistant_only_loss=True)`` — using that flag silently
produces ``sum(mask) == 0``, training on nothing.

The fix: render the chat template, locate each assistant message's body
by its ``<|im_start|>assistant\\n`` header and ``<|im_end|>`` closer,
tokenize with ``return_offsets_mapping=True``, and unmask only those
tokens whose character span intersects an assistant-body region.

Both NPC Agentic v1 and v2 train with this exact helper. Future Qwen-
template runs should import from here rather than reinventing it.

Sanity probe — assistant-token share
------------------------------------
A correct mask leaves ~25-45 % of tokens unmasked on a typical
multi-turn chat dataset. ``sanity_check_mask_share`` returns that ratio
so the trainer can fail fast if the chat template's special tokens
change and break offset matching.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def build_assistant_spans(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    header: str = "<|im_start|>assistant\n",
    end_marker: str = "<|im_end|>",
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Render the chat template, then locate the character-offset spans of
    every assistant message's body (excluding the role header and the
    closing marker — those are template furniture, not generation).

    Defaults match Qwen2.5 / Qwen2 / Qwen3 chat templates. For
    Llama-3-Instruct, override::

        header="<|start_header_id|>assistant<|end_header_id|>\\n\\n"
        end_marker="<|eot_id|>"

    Returns
    -------
    (rendered_text, [(start_char, end_char), ...])
        ``rendered_text`` is the chat template applied with
        ``add_generation_prompt=False`` (we want the closer in the
        rendered string so we can find it). The span list has one entry
        per assistant turn in ``messages``.
    """
    rendered: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )

    spans: List[Tuple[int, int]] = []
    cursor = 0

    for m in messages:
        if m.get("role") != "assistant":
            continue
        header_pos = rendered.find(header, cursor)
        if header_pos < 0:
            # Assistant turn declared in messages but not present in the
            # rendered output — skip rather than miscount.
            continue
        body_start = header_pos + len(header)
        body_end = rendered.find(end_marker, body_start)
        if body_end < 0:
            body_end = len(rendered)
        spans.append((body_start, body_end))
        cursor = body_end + len(end_marker)

    return rendered, spans


def preprocess_example(
    tokenizer,
    example: Dict[str, Any],
    max_len: int,
    *,
    header: str = "<|im_start|>assistant\n",
    end_marker: str = "<|im_end|>",
    messages_key: str = "messages",
) -> Dict[str, List[int]]:
    """
    Tokenize one example into ``{input_ids, attention_mask, labels}``.

    A token's label is unmasked (set to its ``input_id``) iff its
    character span intersects any assistant-body region returned by
    :func:`build_assistant_spans`. All other tokens get ``-100`` so they
    are ignored by the cross-entropy loss.

    Parameters
    ----------
    tokenizer
        Must support ``apply_chat_template`` and
        ``return_offsets_mapping=True``. Most fast tokenizers do.
    example
        A row from the SFT dataset. ``example[messages_key]`` is a list
        of ``{"role": ..., "content": ...}`` dicts.
    max_len
        Truncation length. The chat template is rendered first then
        truncated, which can clip a long assistant turn — that's
        intentional and matches HF SFTTrainer behaviour.
    """
    rendered, spans = build_assistant_spans(
        tokenizer, example[messages_key],
        header=header, end_marker=end_marker,
    )

    enc = tokenizer(
        rendered,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids: List[int] = enc["input_ids"]
    attn:      List[int] = enc["attention_mask"]
    offsets:   List[Tuple[int, int]] = enc["offset_mapping"]

    labels = [-100] * len(input_ids)
    for i, (ts, te) in enumerate(offsets):
        # (0, 0) is the convention for special / added tokens — never
        # part of an assistant body.
        if ts == te:
            continue
        for sa, sb in spans:
            if te > sa and ts < sb:   # any intersection
                labels[i] = input_ids[i]
                break

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }


def sanity_check_mask_share(samples: Sequence[Dict[str, List[int]]]) -> float:
    """
    Fraction of tokens with ``label != -100`` across ``samples``.

    Healthy range: ~0.20 - 0.50 for typical multi-turn chat SFT data.
    A value at or near 0.0 means the chat-template markers don't match
    the ``header`` / ``end_marker`` you passed in; fail fast there
    rather than burning a training run.
    """
    total = sum(len(s["labels"]) for s in samples)
    if total == 0:
        return 0.0
    unmasked = sum(sum(1 for x in s["labels"] if x != -100) for s in samples)
    return unmasked / total
