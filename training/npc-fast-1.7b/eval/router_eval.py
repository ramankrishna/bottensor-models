"""
Router eval — can NPC Fast correctly decide "self" vs "npc_fin"?

We hand-construct 200 queries with ground-truth labels:
  * "self"    — NPC Fast should handle directly (short lookup, light reasoning,
                trivial tool calls, identity, formatting)
  * "npc_fin" — escalate to NPC Fin 32B (multi-step finance reasoning,
                market analysis, multi-document synthesis, heavy math)

The model is prompted to emit a single JSON object:
  {"route": "self"|"npc_fin", "reason": "...", "confidence": 0.0-1.0}

We report accuracy, false escalation rate, and missed escalation rate.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Optional

from ._utils import extract_json, generate, load_model, log_wandb, merge_results

LOG = logging.getLogger("npc-fast.eval.router")

ROUTER_SYSTEM = (
    "You are NPC Fast, a 1.7B router model. Decide whether to handle a user "
    "request yourself or forward it to NPC Fin 32B. Forward to 'npc_fin' when "
    "the task requires deep multi-step reasoning, specialized financial "
    "analysis, long-context document synthesis, or heavy math. Handle 'self' "
    "when the task is simple lookup, format conversion, short code, tool "
    "calling with obvious arguments, identity/meta questions, or chit-chat.\n\n"
    "Respond with exactly one JSON object:\n"
    '{"route": "self" | "npc_fin", "reason": "<short>", "confidence": 0.0-1.0}'
)

# 100 self + 100 npc_fin examples, expanded below with synonyms to reach 200.
_SELF_SEED = [
    "What's your name?",
    "Convert 98 F to Celsius.",
    "Reply in all lowercase.",
    "Who built you?",
    "Summarize this sentence in 5 words: '{s}'",
    "What is 17 * 23?",
    "Send a hello email to alice@x.com.",
    "Define: mitochondrion.",
    "Output the JSON: {\"ok\": true}.",
    "Translate 'good morning' to French.",
    "What tools do you have access to?",
    "Reverse the string 'anthropic'.",
    "Is Python interpreted or compiled?",
    "Search the web for 'weather NYC'.",
    "List the first 5 prime numbers.",
    "Capitalize: 'npc fast is fast'.",
    "Open the file ~/notes.txt.",
    "Give me a haiku about rain.",
    "Explain recursion in one sentence.",
    "Tell me the date format ISO-8601 uses.",
]

_FIN_SEED = [
    "Given TSLA's last 10K, analyze revenue concentration risk across segments.",
    "Build a DCF model for NVDA assuming 20% FCF growth tapering to 5%.",
    "Compare the systemic risk of US regional banks to money-center banks after SVB.",
    "Given Fed dot-plot projections, model yield curve impact on a 60/40 portfolio.",
    "Analyze options flow for AMD over the last 30 days and infer institutional positioning.",
    "Walk through a merger arbitrage strategy for the MSFT-ATVI deal step by step.",
    "Quantify basis risk in a Treasury futures hedge of a corporate bond portfolio.",
    "Decompose JPM's net interest margin into volume, rate, and mix components.",
    "Model the counterparty risk exposure for a prime broker during a liquidity crisis.",
    "Derive the gamma exposure profile for market-makers given current SPX positioning.",
    "Scenario-analyze a 200bps rate shock on a leveraged mortgage REIT.",
    "Do a sum-of-the-parts valuation on META including Reality Labs, Family of Apps, and AI.",
    "Apply the Merton model to estimate Credit Suisse's implied default probability in March 2023.",
    "Perform a factor attribution decomposition on a long/short tech book.",
    "Work through a convex bond hedging problem using duration and convexity.",
    "Analyze the capital structure implications of a $50B share buyback financed by debt.",
    "Explain the math behind VWAP slippage on a block trade with time-weighted execution.",
    "Derive the Black-Scholes implied volatility surface skew for SPY 30-day options.",
    "Model the liquidity coverage ratio impact of a run-off scenario on a mid-sized bank.",
    "Compare the stress-test assumptions in CCAR 2024 vs 2023 and quantify capital shortfalls.",
]


def _build_dataset(n_each: int = 100) -> list[dict]:
    rng_self = _SELF_SEED * ((n_each // len(_SELF_SEED)) + 1)
    rng_fin = _FIN_SEED * ((n_each // len(_FIN_SEED)) + 1)
    dataset: list[dict] = []
    for q in rng_self[:n_each]:
        dataset.append({"query": q.replace("{s}", "The quick brown fox jumps."),
                        "label": "self"})
    for q in rng_fin[:n_each]:
        dataset.append({"query": q, "label": "npc_fin"})
    return dataset


def _score(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["label"])
    # Escalation = routed to npc_fin
    tp = sum(1 for r in results if r["predicted"] == "npc_fin" and r["label"] == "npc_fin")
    fn = sum(1 for r in results if r["predicted"] != "npc_fin" and r["label"] == "npc_fin")
    fp = sum(1 for r in results if r["predicted"] == "npc_fin" and r["label"] == "self")
    tn = sum(1 for r in results if r["predicted"] != "npc_fin" and r["label"] == "self")
    return {
        "n": total,
        "accuracy": correct / max(1, total),
        "false_escalation_rate": fp / max(1, fp + tn),   # self→npc_fin errors
        "missed_escalation_rate": fn / max(1, fn + tp),  # npc_fin→self errors
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_each", type=int, default=100)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model, tok = load_model(args.model_path)

    dataset = _build_dataset(args.n_each)
    results: list[dict] = []
    for row in dataset:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": row["query"]},
        ]
        out = generate(model, tok, messages, max_new_tokens=120, temperature=0.0)
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

    summary = {"router": {**_score(results), "examples": results[:20]}}
    merge_results(summary)
    log_wandb({
        "eval/router.accuracy": summary["router"]["accuracy"],
        "eval/router.false_escalation": summary["router"]["false_escalation_rate"],
        "eval/router.missed_escalation": summary["router"]["missed_escalation_rate"],
    })
    LOG.info("Router summary: %s", {k: v for k, v in summary["router"].items() if k != "examples"})


if __name__ == "__main__":
    main()
