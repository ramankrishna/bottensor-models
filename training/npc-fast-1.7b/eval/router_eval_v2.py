"""
Router eval v2 — rewritten prompt to fix the schema-echo bug.

Original prompt used pipe notation `"self" | "npc_fin"` in the schema which
the 1.7B model was echoing literally into its output. v2 uses two concrete
few-shot examples and removes all pipe-notation templating.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Optional

from ._utils import extract_json, generate, load_model, merge_results
from .router_eval import _SELF_SEED, _FIN_SEED, _build_dataset, _score

LOG = logging.getLogger("npc-fast.eval.router_v2")

ROUTER_SYSTEM = (
    "You are NPC Fast, a 1.7B router. For each user request, output one line "
    "of JSON with two fields: route (a string, either self or npc_fin) and "
    "reason (a short phrase).\n\n"
    "Choose npc_fin when the task needs deep multi-step reasoning, specialized "
    "financial analysis, long-document synthesis, or heavy math.\n"
    "Choose self for simple lookup, format conversion, short code, tool calls "
    "with obvious arguments, identity questions, translation, or chit-chat.\n\n"
    "Examples:\n"
    "User: What is 2+2?\n"
    "Assistant: {\"route\": \"self\", \"reason\": \"trivial arithmetic\"}\n\n"
    "User: Build a DCF model for TSLA with 3 scenarios.\n"
    "Assistant: {\"route\": \"npc_fin\", \"reason\": \"multi-step financial model\"}\n\n"
    "Output only the JSON object for the user's request. Nothing else."
)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_each", type=int, default=100)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model, tok = load_model(args.model_path)

    dataset = _build_dataset(args.n_each)
    results: list[dict] = []
    for i, row in enumerate(dataset):
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": row["query"]},
        ]
        out = generate(model, tok, messages, max_new_tokens=60, temperature=0.0)
        parsed = extract_json(out)
        pred = "unknown"
        conf: Optional[float] = None
        if parsed and parsed.get("route") in {"self", "npc_fin"}:
            pred = parsed["route"]
            c = parsed.get("confidence")
            if isinstance(c, (int, float)):
                conf = float(c)
        results.append({"query": row["query"], "label": row["label"],
                        "predicted": pred, "confidence": conf, "raw": out})
        if (i + 1) % 25 == 0:
            partial = _score(results)
            LOG.info("[%d/%d] acc=%.3f miss=%.3f fp=%.3f",
                     i + 1, len(dataset), partial["accuracy"],
                     partial["missed_escalation_rate"], partial["false_escalation_rate"])

    summary = {"router_v2": {**_score(results), "examples": results[:20]}}
    merge_results(summary)
    LOG.info("Router v2 summary: %s",
             {k: v for k, v in summary["router_v2"].items() if k != "examples"})


if __name__ == "__main__":
    main()
