"""
HuggingFace Hub publish.

Creates the repo if it doesn't exist, writes a model card, uploads the
folder contents. Supports three artifact types:

  * bf16 full model          → ramankrishna10/npc-fast-1.7b
  * GPTQ 4-bit                → ramankrishna10/npc-fast-1.7b-gptq
  * GGUF (Q4_K_M + Q8_0)      → ramankrishna10/npc-fast-1.7b-gguf

Use ``--repo`` to override the default target.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from config import BUILT_BY, CREATOR, MODEL_NAME, PARENT_COMPANY, PARTNER_MODEL, ROLE

LOG = logging.getLogger("npc-fast.export.push")


MODEL_CARD_TEMPLATE = f"""---
license: apache-2.0
language:
- en
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
library_name: transformers
tags:
- agentic
- router
- long-context
- 128k
- {{artifact_tag}}
---

# {MODEL_NAME}

Fast agentic router model with 128K context window.

## Architecture
- Base: SmolLM2-1.7B-Instruct
- Training: Full-weight continual pre-training
- Context: 128K tokens via YaRN RoPE scaling
- Precision: BF16

## Training Data
- ~60.8K examples from 3 sources
- Agentic tool-use, function calling, reasoning traces
- 5-stage curriculum learning (4K → 128K context)

## Intended Use
- {ROLE}
- Decides when to handle requests vs escalate to {PARTNER_MODEL}
- Tool selection, function calling, structured output
- Multi-step planning and task decomposition

## Artifact
`{{artifact_tag}}`

## Built by
{BUILT_BY} (a {PARENT_COMPANY} company) — creator: {CREATOR}
"""


def _write_card(folder: Path, artifact_tag: str) -> None:
    card_path = folder / "README.md"
    if card_path.exists():
        return
    card_path.write_text(MODEL_CARD_TEMPLATE.format(artifact_tag=artifact_tag))
    LOG.info("Wrote model card → %s", card_path)


def _infer_artifact_tag(folder: Path) -> str:
    names = {p.name.lower() for p in folder.iterdir()}
    if any(n.endswith(".gguf") for n in names):
        return "gguf"
    if "quantize_config.json" in names or any("gptq" in n for n in names):
        return "gptq-4bit"
    return "bf16"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--repo", required=True,
                    help="e.g. ramankrishna10/npc-fast-1.7b")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit_message", default="NPC Fast — initial upload")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    from huggingface_hub import HfApi

    folder = Path(args.model_path).resolve()
    if not folder.is_dir():
        raise RuntimeError(f"Model path is not a directory: {folder}")

    tag = _infer_artifact_tag(folder)
    _write_card(folder, tag)

    api = HfApi(token=token)
    LOG.info("Creating repo %s (private=%s)", args.repo, args.private)
    api.create_repo(repo_id=args.repo, exist_ok=True, private=args.private)

    LOG.info("Uploading %s → %s", folder, args.repo)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(folder),
        commit_message=args.commit_message,
    )
    LOG.info("Upload complete: https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
