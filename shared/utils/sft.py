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
tokenize with ``return_offsets_mapping=True``, and unmask the tokens
whose character span intersects an assistant region — **including the
closing turn marker**.

Critical bug history (v1 + v2 vs v3)
------------------------------------
NPC Agentic v1 and v2 used a build of this helper that EXCLUDED the
closing ``<|im_end|>`` token from the unmasked span. That meant the EOS
token never received gradient during training, and the shipped models
could not terminate cleanly at inference time — they would generate
their answer correctly, then lapse into base-Qwen pretraining priors:
synthetic ``Human:``/``Assistant:`` continuations, repeated
``</details>``, hallucinated ``<thoughts>``/``<chat_history>`` tags.

v3 onwards includes the closer in the span. The fix is one line:

    body_end = marker_pos + len(end_marker)   # was: marker_pos

Always run :func:`verify_eos_in_loss` on a sample of the preprocessed
training set before kicking off a long run; an EOS-coverage of zero
means the bug has regressed.

Sanity probes
-------------
- :func:`sanity_check_mask_share`: ~25-45 % of tokens unmasked on a
  typical chat dataset. Below 5 % means the chat-template markers don't
  match (templates drift between Qwen versions); fix offsets fast.
- :func:`verify_eos_in_loss`: every assistant turn's closing
  ``<|im_end|>`` should be unmasked. If not, you have the v1/v2 bug.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def build_assistant_spans(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    header: str = "<|im_start|>assistant\n",
    end_marker: str = "<|im_end|>",
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Render the chat template, then locate the character-offset spans of
    every assistant message's body **including the closing turn marker**
    (e.g. ``<|im_end|>``) so the EOS token is in the trained loss.

    .. warning::

       NPC Agentic v1 and v2 used a version of this function that
       *excluded* the closing marker from the span. The result was that
       the EOS token never received gradient during training, and the
       shipped models could not terminate cleanly at inference time —
       they would lapse into base-model pretraining priors after
       finishing their content (synthetic ``Human:``/``Assistant:``
       continuations, repeated ``</details>``, hallucinated
       ``<thoughts>``/``<chat_history>`` tags). v3 onwards includes the
       closer. See the npc-agentic-7b-v3 paper §4 for the post-mortem.

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
        per assistant turn in ``messages``. ``end_char`` is the position
        **just after** the closing marker, so the marker itself is
        included in the unmasked span.
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
        marker_pos = rendered.find(end_marker, body_start)
        if marker_pos < 0:
            # Unterminated turn — extend the unmasked region to end of string.
            body_end = len(rendered)
        else:
            # INCLUDE the closing marker in the unmasked span so the EOS
            # token receives gradient during training. (v1/v2 bug fix.)
            body_end = marker_pos + len(end_marker)
        spans.append((body_start, body_end))
        cursor = body_end

    return rendered, spans


def verify_eos_in_loss(
    tokenizer,
    samples: Sequence[Dict[str, List[int]]],
    *,
    eos_token_id: Optional[int] = None,
) -> Dict[str, int]:
    """
    Verification probe: confirm the EOS token (`<|im_end|>` for Qwen) is
    actually in the trained loss after :func:`preprocess_example`.

    For each sample's ``input_ids`` + ``labels``, find every position
    where the input is the EOS token. Count how many are unmasked
    (``label != -100``) vs masked.

    A correct setup has **all** assistant-turn-closing EOS tokens
    unmasked (system + user closers stay masked since those tokens
    aren't part of any assistant span). A broken setup (the v1/v2 bug)
    has **zero** assistant-turn EOS tokens unmasked.

    Returns ``{"eos_total": N, "eos_unmasked": K, "samples": M}``. The
    caller should assert ``K > 0`` before kicking off training.
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    total = 0
    unmasked = 0
    for s in samples:
        ids = s["input_ids"]
        labels = s["labels"]
        for tok, lbl in zip(ids, labels):
            if tok == eos_token_id:
                total += 1
                if lbl != -100:
                    unmasked += 1
    return {
        "eos_total": total,
        "eos_unmasked": unmasked,
        "samples": len(samples),
    }


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
