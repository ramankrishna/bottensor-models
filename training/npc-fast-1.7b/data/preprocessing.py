"""
Preprocessing: chat-template → tokenize → pack into fixed-length sequences.

Exposes a ``PackedDataset`` whose ``max_seq_length`` is mutable — the
curriculum callback flips it at stage boundaries. Each ``__getitem__`` call
returns a dict with ``input_ids``, ``attention_mask``, and ``labels`` for
causal LM loss.

Packing: we greedily concatenate normalized examples separated by EOS until
the combined length reaches (or exceeds) ``max_seq_length``, then emit the
packed sequence. This is the standard trick to avoid padding waste on
variable-length SFT data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset

from .loader import NormalizedExample

LOG = logging.getLogger("npc-fast.preproc")


@dataclass
class TokenizedExample:
    input_ids: list[int]
    source: str
    tags: list[str]


def tokenize_examples(
    examples: list[NormalizedExample],
    tokenizer,
) -> list[TokenizedExample]:
    """Apply the SmolLM2 chat template and tokenize each example once."""
    out: list[TokenizedExample] = []
    eos_id = tokenizer.eos_token_id
    for ex in examples:
        text = tokenizer.apply_chat_template(
            ex.messages, tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        # Ensure each example ends with EOS so packing has a clean delimiter.
        if eos_id is not None and ids[-1] != eos_id:
            ids.append(eos_id)
        out.append(TokenizedExample(input_ids=ids, source=ex.source, tags=ex.tags))
    LOG.info("Tokenized %d examples (avg len %.1f)",
             len(out),
             sum(len(e.input_ids) for e in out) / max(1, len(out)))
    return out


def pack_sequences(
    tokenized: list[TokenizedExample],
    max_seq_length: int,
) -> list[list[int]]:
    """Greedy concat until we hit max_seq_length; emit and start a fresh buffer.

    Sequences longer than max_seq_length on their own are truncated to
    max_seq_length so no example is silently dropped.
    """
    packs: list[list[int]] = []
    buf: list[int] = []
    for ex in tokenized:
        ids = ex.input_ids
        if len(ids) >= max_seq_length:
            # Flush current buffer, then truncate and emit oversize example solo.
            if buf:
                packs.append(buf)
                buf = []
            packs.append(ids[:max_seq_length])
            continue
        if len(buf) + len(ids) > max_seq_length:
            packs.append(buf)
            buf = list(ids)
        else:
            buf.extend(ids)
    if buf:
        packs.append(buf)
    LOG.info("Packed %d tokenized examples into %d sequences at max_seq=%d",
             len(tokenized), len(packs), max_seq_length)
    return packs


class PackedDataset(Dataset):
    """Torch dataset over pre-tokenized examples with mutable max_seq_length.

    Because packing depends on ``max_seq_length`` and that changes per
    curriculum stage, we re-pack on-demand when the value changes. The packed
    list is cached until the next change.
    """

    def __init__(
        self,
        tokenized: list[TokenizedExample],
        initial_max_seq_length: int,
    ) -> None:
        self._tokenized = tokenized
        self._max_seq_length = initial_max_seq_length
        self._packed: Optional[list[list[int]]] = None
        self._repack()

    # ---- public API ----
    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def set_max_seq_length(self, new_len: int) -> None:
        if new_len == self._max_seq_length:
            return
        LOG.info("max_seq_length %d -> %d (re-packing)", self._max_seq_length, new_len)
        self._max_seq_length = new_len
        self._repack()

    def __len__(self) -> int:
        assert self._packed is not None
        return len(self._packed)

    def __getitem__(self, idx: int) -> dict:
        assert self._packed is not None
        ids = self._packed[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def iter_tokens(self) -> Iterator[int]:
        assert self._packed is not None
        for seq in self._packed:
            yield from seq

    # ---- internal ----
    def _repack(self) -> None:
        self._packed = pack_sequences(self._tokenized, self._max_seq_length)


class DynamicPadCollator:
    """Right-pad a batch of variable-length packed sequences.

    All packed sequences within a stage target the same ``max_seq_length`` so
    in practice they are close to the same length — but stage transitions can
    produce a batch straddling the boundary. We handle that by padding to the
    longest item in the batch.
    """

    def __init__(self, pad_token_id: int, label_pad_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(int(f["input_ids"].shape[0]) for f in features)

        input_ids = torch.full(
            (len(features), max_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full(
            (len(features), max_len), self.label_pad_id, dtype=torch.long
        )

        for i, f in enumerate(features):
            n = int(f["input_ids"].shape[0])
            input_ids[i, :n] = f["input_ids"]
            attention_mask[i, :n] = f["attention_mask"]
            labels[i, :n] = f["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
