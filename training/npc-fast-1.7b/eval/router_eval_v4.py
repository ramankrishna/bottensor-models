"""Router eval v4 — with LoRA adapter loaded on top of base."""
from __future__ import annotations
import argparse, logging
from typing import Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from ._utils import extract_json, merge_results
from .router_eval import _build_dataset, _score

LOG = logging.getLogger("npc-fast.eval.router_v4")

# Match the SYSTEM used in training
SYSTEM = (
    "You are NPC Fast, a capable 1.7B model. Handle most requests yourself. "
    "Only forward to the larger NPC Fin 32B model when a task truly requires "
    "deep multi-step financial analysis that you cannot do well alone.\n\n"
    "Default: route=self.\n"
    "Escalate to npc_fin ONLY if ALL of these are true:\n"
    "  - the task is about finance, markets, banking, derivatives, or valuation\n"
    "  - it requires multi-step quantitative reasoning or deep domain knowledge\n"
    "  - a short answer would be wrong or superficial\n\n"
    "Output exactly one JSON object with fields route and reason."
)


@torch.no_grad()
def generate(model, tok, messages, max_new_tokens=60):
    enc = tok.apply_chat_template(messages, tokenize=True,
                                   add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens,
                          do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[0][enc["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=True)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n_each", type=int, default=100)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    LOG.info("Loading base: %s", args.base)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
        device_map="auto",
    )
    LOG.info("Loading adapter: %s", args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    dataset = _build_dataset(args.n_each)
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
        if (i + 1) % 25 == 0:
            partial = _score(results)
            LOG.info("[%d/%d] acc=%.3f miss=%.3f fp=%.3f",
                     i+1, len(dataset), partial["accuracy"],
                     partial["missed_escalation_rate"], partial["false_escalation_rate"])

    summary = {"router_v4_lora": {**_score(results), "examples": results[:20]}}
    merge_results(summary)
    LOG.info("Router v4 (LoRA) summary: %s",
             {k: v for k, v in summary["router_v4_lora"].items() if k != "examples"})


if __name__ == "__main__":
    main()
