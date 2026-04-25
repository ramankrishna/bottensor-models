"""
Load registered datasets from HuggingFace Hub and normalize via adapters.

Reads ``datasets.json`` → for each entry, downloads via ``datasets.load_dataset``
(honoring ``subset`` for configs and ``private`` via the HF token), pipes every
row through its adapter, and returns a list of ``NormalizedExample`` dicts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError

from .adapters import get_adapter

LOG = logging.getLogger("npc-fast.loader")


@dataclass
class NormalizedExample:
    source: str                # dataset id (+ subset if applicable)
    format_id: str             # "sharegpt" | "openai"
    weight: float              # mixing weight from the registry
    messages: list[dict]       # normalized chat messages
    tags: list[str] = field(default_factory=list)


def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _load_one(entry: dict) -> list[NormalizedExample]:
    ds_id = entry["id"]
    fmt = entry["type"]
    split = entry.get("split", "train")
    subset = entry.get("subset")
    weight = float(entry.get("weight", 1.0))
    private = bool(entry.get("private", False))

    load_kwargs = {"split": split}
    if subset:
        load_kwargs["name"] = subset
    if private:
        token = _hf_token()
        if not token:
            raise RuntimeError(
                f"Dataset {ds_id} is private but HF_TOKEN is not set."
            )
        load_kwargs["token"] = token

    LOG.info("Loading %s (subset=%s, split=%s)...", ds_id, subset, split)
    try:
        ds = load_dataset(ds_id, **load_kwargs)
    except (DatasetGenerationError, TypeError, ValueError) as e:
        # Declared feature schema can disagree with actual parquet contents
        # (e.g. some conversation turns carry an extra `tool` key). Fall
        # back to a raw parquet read with schema inferred directly from
        # the files, bypassing the dataset's feature declaration.
        LOG.warning(
            "  %s: schema-strict load failed (%s) — falling back to raw files.",
            ds_id, e.__class__.__name__,
        )
        ds = _load_raw_files(ds_id, subset=subset, split=split, token=load_kwargs.get("token"))
    adapter = get_adapter(fmt)

    src = ds_id if not subset else f"{ds_id}::{subset}"
    out: list[NormalizedExample] = []
    kept = dropped = 0
    for row in ds:
        messages = adapter.normalize(row)
        if messages is None:
            dropped += 1
            continue
        tags = _derive_tags(row, messages)
        out.append(
            NormalizedExample(
                source=src, format_id=fmt, weight=weight,
                messages=messages, tags=tags,
            )
        )
        kept += 1
    LOG.info("  %s: kept %d / %d (%.1f%%)", src, kept, kept + dropped,
             100.0 * kept / max(1, kept + dropped))
    return out


def _load_raw_files(ds_id: str, subset: Optional[str], split: str,
                     token: Optional[str]) -> Iterable[dict]:
    """Read raw parquet / jsonl files directly from the HF repo.

    Bypasses ``datasets`` feature-schema enforcement so rows with extra
    struct fields (e.g. a ``tool`` key present on some turns but not others)
    don't trigger a cast failure. Handles both parquet shards and JSONL
    dumps (the common hand-uploaded layout).
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = api.list_repo_files(ds_id, repo_type="dataset", token=token)

    parquet_files = [f for f in files if f.endswith(".parquet")]
    jsonl_files = [f for f in files if f.endswith(".jsonl") or f.endswith(".json")]

    # Filter parquet by split + optional subset in path.
    pq_wanted: list[str] = []
    for f in parquet_files:
        low = f.lower()
        if subset and f"/{subset.lower()}/" not in low and not low.startswith(f"{subset.lower()}/"):
            continue
        if split.lower() not in low:
            continue
        pq_wanted.append(f)

    rows: list[dict] = []
    if pq_wanted:
        import pyarrow.parquet as pq
        pq_wanted.sort()
        LOG.info("  %s: raw-parquet fallback over %d file(s)", ds_id, len(pq_wanted))
        for rel in pq_wanted:
            local = hf_hub_download(
                repo_id=ds_id, filename=rel, repo_type="dataset", token=token,
            )
            rows.extend(pq.read_table(local).to_pylist())
        return rows

    # JSONL fallback — common for hand-uploaded datasets. Each shard is one
    # category; union them all for the train split.
    if jsonl_files:
        jsonl_files.sort()
        LOG.info("  %s: raw-jsonl fallback over %d file(s)", ds_id, len(jsonl_files))
        for rel in jsonl_files:
            local = hf_hub_download(
                repo_id=ds_id, filename=rel, repo_type="dataset", token=token,
            )
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    raise RuntimeError(
        f"No parquet or jsonl files found for {ds_id} (subset={subset}, split={split})."
    )


def _derive_tags(row: dict, messages: list[dict]) -> list[str]:
    """Light heuristics so perplexity can be sliced per-tag later."""
    tags: list[str] = []
    text = " ".join(m["content"] for m in messages).lower()
    if any(t.get("role") == "tool" for t in messages):
        tags.append("tool_use")
    if "<think>" in text or "step 1" in text or "step 2" in text:
        tags.append("reasoning")
    if len(messages) >= 4:
        tags.append("multi_step")
    if "i'm npc" in text or "built by bottensor" in text:
        tags.append("identity")
    # Pull category hint from ShareGPT-style rows
    cat = row.get("category") or row.get("task")
    if isinstance(cat, str) and cat.strip():
        tags.append(f"cat:{cat.strip().lower().replace(' ', '_')}")
    return tags


def load_registry(registry_path: Path | str) -> list[NormalizedExample]:
    registry_path = Path(registry_path)
    with registry_path.open() as f:
        registry = json.load(f)

    all_examples: list[NormalizedExample] = []
    for entry in registry["datasets"]:
        try:
            all_examples.extend(_load_one(entry))
        except Exception as e:  # noqa: BLE001 — log and continue
            LOG.error("Failed to load %s: %s", entry["id"], e)
            raise

    LOG.info("Total normalized examples across all datasets: %d", len(all_examples))
    return all_examples


def preview(registry_path: Path | str, k: int = 5) -> None:
    """Show k random examples per dataset (used by `main.py preview`)."""
    import random

    with Path(registry_path).open() as f:
        registry = json.load(f)
    for entry in registry["datasets"]:
        try:
            ex = _load_one(entry)
        except Exception as e:  # noqa: BLE001
            print(f"[preview] {entry['id']}: FAILED ({e})")
            continue
        src = ex[0].source if ex else entry["id"]
        print(f"\n=== {src} ({len(ex)} examples) ===")
        sample = random.sample(ex, min(k, len(ex)))
        for i, item in enumerate(sample, 1):
            print(f"--- sample {i} (tags={item.tags}) ---")
            for m in item.messages[:6]:
                head = m["content"][:220].replace("\n", " ")
                print(f"  [{m['role']:>9}] {head}")
