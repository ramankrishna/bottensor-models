"""Router eval OOD — 60 semantically novel queries not in training seed pool.

The training data was derived from router_eval._SELF_SEED / _FIN_SEED.
This OOD set uses completely different surface forms + topics to test
real generalization of the LoRA router, not memorization.
"""
from __future__ import annotations
import argparse, logging
from typing import Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from ._utils import extract_json, merge_results
from .router_eval_v4 import SYSTEM, generate
from .router_eval import _score

LOG = logging.getLogger("npc-fast.eval.router_ood")

# 30 novel self queries (different topics, different surface forms from training)
OOD_SELF = [
    "Spell check this: recieve",
    "What year did WWII end?",
    "Round pi to 2 decimals.",
    "Give me the hex color for red.",
    "Name a famous painter.",
    "What is the melting point of ice in Celsius?",
    "Write a one-line poem about stars.",
    "What language is Linux written in?",
    "Is Mars a planet or a star?",
    "List three animals that hibernate.",
    "What does the acronym API stand for?",
    "Convert 100 km/h to mph.",
    "How do you spell the plural of goose?",
    "Name a chemical element starting with C.",
    "Say goodbye politely in Italian.",
    "Echo this back: purple elephant.",
    "Turn this into uppercase: claude is cool",
    "What continent is Egypt in?",
    "Give me a synonym for happy.",
    "What do you call a baby dog?",
    "Is the sky actually blue or does it just appear blue?",
    "Short joke about computers please.",
    "Invert this list: [a, b, c, d].",
    "How many sides does a hexagon have?",
    "Cube 4.",
    "Write a haiku about a bicycle.",
    "What does DNA stand for?",
    "Name any Greek philosopher.",
    "Format the number 1234567 with commas.",
    "What is the longest river in Africa?",
]

# 30 novel fin queries (genuinely novel finance tasks, not in training seed pool)
OOD_FIN = [
    "Walk through the Black-Litterman framework applied to a 5-asset portfolio.",
    "Derive the replicating portfolio for a down-and-in barrier option.",
    "Model the funding P&L of a repo trade during a quarter-end liquidity stress.",
    "Stress-test a CLO equity tranche under a credit-spread widening of 300bp.",
    "Quantify the relative value of NKY vs SPX futures given correlation breaks.",
    "Model an insurance company surplus note under a 1-in-200-year catastrophe scenario.",
    "Estimate the DV01 and convexity of a callable municipal bond portfolio.",
    "Analyze the convexity hedging flows from mortgage servicers when rates rise 50bp.",
    "Derive the fair value of a perpetual AT1 bank CoCo using the Merton approach.",
    "Build a multi-factor attribution for a global macro fund across 4 sleeves.",
    "Compute the expected shortfall for a levered short-vol strategy at 99%.",
    "Value a SPAC warrant using Monte Carlo with earn-out triggers.",
    "Model the cross-gamma between SPX and VIX for a dispersion trade.",
    "Analyze the liquidity premium embedded in on-the-run vs off-the-run Treasuries.",
    "Build a discounted cash flow for a biotech with 3 phase-dependent scenarios.",
    "Decompose the tracking error of an emerging markets equity index fund.",
    "Quantify the wrong-way risk in a CDX.HY short exposure during a credit event.",
    "Model the prepayment speed on an agency MBS pool using a 4-factor PSA model.",
    "Stress the solvency capital requirement of a European insurer under Solvency II.",
    "Derive the theta decay profile for a calendar spread into an FOMC meeting.",
    "Analyze cash-and-carry arbitrage between gold spot and 6M futures.",
    "Build a 5-year earnings bridge for a capex-heavy industrial company.",
    "Model the net interest income sensitivity to a 100bp parallel rate shift for a regional bank.",
    "Decompose a private equity fund IRR into leverage, multiple, and operational components.",
    "Compute the risk contribution of each factor in a 6-factor equity risk model.",
    "Stress-test a cross-currency swap book under a EUR/USD spot move of 20%.",
    "Derive the convexity adjustment for a CMS-linked swap payoff.",
    "Model the fail-to-deliver dynamics in a stressed repo market.",
    "Analyze the term structure of implied correlation in an index options book.",
    "Build a multi-period liability-driven investment glide path for a DB pension.",
]


def build_ood():
    return (
        [{"query": q, "label": "self"} for q in OOD_SELF] +
        [{"query": q, "label": "npc_fin"} for q in OOD_FIN]
    )


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    dataset = build_ood()
    results: list[dict] = []
    for i, row in enumerate(dataset):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["query"]},
        ]
        out = generate(model, tok, messages, max_new_tokens=60)
        parsed = extract_json(out)
        pred = "unknown"
        if parsed and parsed.get("route") in {"self", "npc_fin"}:
            pred = parsed["route"]
        results.append({"query": row["query"], "label": row["label"],
                        "predicted": pred, "raw": out})
        if (i + 1) % 15 == 0:
            partial = _score(results)
            LOG.info("[%d/%d] acc=%.3f miss=%.3f fp=%.3f",
                     i+1, len(dataset), partial["accuracy"],
                     partial["missed_escalation_rate"], partial["false_escalation_rate"])

    summary = {"router_ood": {**_score(results), "examples": results}}
    merge_results(summary)
    LOG.info("Router OOD summary: %s",
             {k: v for k, v in summary["router_ood"].items() if k != "examples"})
    # Print the errors so we can see failure modes
    errors = [r for r in results if r["predicted"] != r["label"]]
    if errors:
        LOG.info("--- %d errors ---", len(errors))
        for e in errors[:15]:
            LOG.info("label=%s pred=%s q=%r raw=%r",
                     e["label"], e["predicted"], e["query"][:50], e["raw"][:80])


if __name__ == "__main__":
    main()
