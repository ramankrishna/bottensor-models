"""Router eval v3 — bias toward self, more self examples, explicit default."""

from __future__ import annotations
import argparse, logging
from typing import Optional
from ._utils import extract_json, generate, load_model, merge_results
from .router_eval import _build_dataset, _score

LOG = logging.getLogger("npc-fast.eval.router_v3")

ROUTER_SYSTEM = (
    "You are NPC Fast, a capable 1.7B model. Handle most requests yourself. "
    "Only forward to the larger NPC Fin 32B model when a task truly requires "
    "deep multi-step financial analysis that you cannot do well alone.\n\n"
    "Default: route=self.\n"
    "Escalate to npc_fin ONLY if ALL of these are true:\n"
    "  - the task is about finance, markets, banking, derivatives, or valuation\n"
    "  - it requires multi-step quantitative reasoning or deep domain knowledge\n"
    "  - a short answer would be wrong or superficial\n\n"
    "Non-financial reasoning, arithmetic, coding, translation, formatting, "
    "tool calls, identity questions, definitions, summarization, and chit-chat "
    "are ALL self, even if they look complex.\n\n"
    "Output exactly one JSON object with fields route and reason. Examples:\n\n"
    "User: What is 17 * 23?\n"
    "Assistant: {\"route\": \"self\", \"reason\": \"arithmetic\"}\n\n"
    "User: Translate good morning to French.\n"
    "Assistant: {\"route\": \"self\", \"reason\": \"translation\"}\n\n"
    "User: Explain recursion in one sentence.\n"
    "Assistant: {\"route\": \"self\", \"reason\": \"definition\"}\n\n"
    "User: List the first 5 prime numbers.\n"
    "Assistant: {\"route\": \"self\", \"reason\": \"simple list\"}\n\n"
    "User: Build a DCF model for TSLA with 3 scenarios.\n"
    "Assistant: {\"route\": \"npc_fin\", \"reason\": \"multi-step finance model\"}\n\n"
    "Output only the JSON for the user's request. Nothing else."
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
        if parsed and parsed.get("route") in {"self", "npc_fin"}:
            pred = parsed["route"]
        results.append({"query": row["query"], "label": row["label"],
                        "predicted": pred, "raw": out})
        if (i + 1) % 25 == 0:
            partial = _score(results)
            LOG.info("[%d/%d] acc=%.3f miss=%.3f fp=%.3f",
                     i+1, len(dataset), partial["accuracy"],
                     partial["missed_escalation_rate"], partial["false_escalation_rate"])

    summary = {"router_v3": {**_score(results), "examples": results[:20]}}
    merge_results(summary)
    LOG.info("Router v3 summary: %s",
             {k: v for k, v in summary["router_v3"].items() if k != "examples"})


if __name__ == "__main__":
    main()
