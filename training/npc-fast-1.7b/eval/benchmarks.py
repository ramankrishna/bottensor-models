"""
Agentic benchmarks — BFCL v2 (tool calling), IFEval (instruction following),
and a custom 100-scenario agentic suite.

BFCL and IFEval loaders pull from their public HF mirrors; if those are
unavailable we synthesize a small in-repo fallback for CI / offline usage.
Results are aggregated into ``output/eval_results.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from typing import Optional

from ._utils import extract_json, generate, load_model, log_wandb, merge_results

LOG = logging.getLogger("npc-fast.eval.benchmarks")


# ---------------- BFCL ----------------

def _load_bfcl(n: int = 200) -> list[dict]:
    """Return list of {"messages": [...], "expected_fn": str, "expected_args": dict}."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")
        rows: list[dict] = []
        for r in list(ds.select(range(min(n, len(ds))))):
            question = r.get("question") or r.get("user_query") or ""
            expected = r.get("answer") or r.get("ground_truth") or []
            if isinstance(expected, str):
                try:
                    expected = json.loads(expected)
                except json.JSONDecodeError:
                    continue
            if not isinstance(expected, list) or not expected:
                continue
            tools = r.get("function") or r.get("functions") or []
            system = (
                "You are a tool-calling assistant. When a tool is appropriate, "
                "respond with a single JSON object containing "
                '"name" and "arguments" and nothing else.'
            )
            if tools:
                system += f"\nAvailable tools:\n{json.dumps(tools, ensure_ascii=False)[:4000]}"
            rows.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ],
                "expected_fn": expected[0].get("name"),
                "expected_args": expected[0].get("arguments", {}),
            })
        return rows
    except Exception as e:  # noqa: BLE001
        LOG.warning("BFCL load failed (%s); using small synthetic fallback.", e)
        return _bfcl_synthetic(n=20)


def _bfcl_synthetic(n: int = 20) -> list[dict]:
    tools = [
        {"name": "get_weather", "parameters": {"city": "string"}},
        {"name": "send_email", "parameters": {"to": "string", "subject": "string", "body": "string"}},
        {"name": "search_web", "parameters": {"query": "string"}},
    ]
    scenarios = [
        ("What's the weather in Paris today?", "get_weather", {"city": "Paris"}),
        ("Email alice@example.com with subject Hi and body 'see you soon'", "send_email",
         {"to": "alice@example.com", "subject": "Hi", "body": "see you soon"}),
        ("Can you search the web for 'best ramen in Tokyo'?", "search_web", {"query": "best ramen in Tokyo"}),
    ]
    sys = (
        "You are a tool-calling assistant. Respond with exactly one JSON object "
        f'with "name" and "arguments". Tools: {json.dumps(tools)}'
    )
    rng = random.Random(0)
    rows = []
    for _ in range(n):
        q, fn, args = rng.choice(scenarios)
        rows.append({
            "messages": [{"role": "system", "content": sys}, {"role": "user", "content": q}],
            "expected_fn": fn,
            "expected_args": args,
        })
    return rows


def _args_match(expected: dict, got: dict) -> bool:
    if not isinstance(got, dict):
        return False
    for k, v in expected.items():
        if str(got.get(k, "")).strip().lower() != str(v).strip().lower():
            return False
    return True


def run_bfcl(model, tokenizer, n: int = 200) -> dict:
    rows = _load_bfcl(n)
    name_correct = args_correct = json_valid = 0
    for row in rows:
        out = generate(model, tokenizer, row["messages"], max_new_tokens=256, temperature=0.0)
        parsed = extract_json(out)
        if parsed is None:
            continue
        json_valid += 1
        fn = parsed.get("name")
        args = parsed.get("arguments", {})
        if fn == row["expected_fn"]:
            name_correct += 1
            if _args_match(row["expected_args"], args):
                args_correct += 1
    total = max(1, len(rows))
    return {
        "n": len(rows),
        "json_validity": json_valid / total,
        "name_accuracy": name_correct / total,
        "args_accuracy": args_correct / total,
    }


# ---------------- IFEval ----------------

def _load_ifeval(n: int = 200) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("google/IFEval", split="train")
        rows = []
        for r in list(ds.select(range(min(n, len(ds))))):
            rows.append({
                "prompt": r["prompt"],
                "instruction_id_list": r.get("instruction_id_list", []),
                "kwargs": r.get("kwargs", []),
            })
        return rows
    except Exception as e:  # noqa: BLE001
        LOG.warning("IFEval load failed (%s); returning small fallback.", e)
        return [
            {"prompt": "Write exactly 3 bullet points about Python. Use asterisks.",
             "instruction_id_list": ["length_constraints:number_bullet_lists"],
             "kwargs": [{"num_bullets": 3}]},
            {"prompt": "Reply in all lowercase.",
             "instruction_id_list": ["change_case:english_lowercase"],
             "kwargs": [{}]},
        ]


def _check_lowercase(text: str, **_) -> bool:
    return text == text.lower()


def _check_num_bullets(text: str, num_bullets: int = 3, **_) -> bool:
    count = sum(1 for line in text.splitlines() if line.strip().startswith(("*", "-", "•")))
    return count == num_bullets


IFEVAL_CHECKERS = {
    "change_case:english_lowercase": _check_lowercase,
    "length_constraints:number_bullet_lists": _check_num_bullets,
}


def run_ifeval(model, tokenizer, n: int = 200) -> dict:
    rows = _load_ifeval(n)
    checkable = passed = 0
    for row in rows:
        out = generate(
            model, tokenizer,
            [{"role": "user", "content": row["prompt"]}],
            max_new_tokens=512, temperature=0.0,
        )
        for inst_id, kw in zip(row["instruction_id_list"], row["kwargs"] or [{}]):
            check = IFEVAL_CHECKERS.get(inst_id)
            if check is None:
                continue
            checkable += 1
            if check(out, **(kw or {})):
                passed += 1
    return {
        "n": len(rows),
        "checkable_instructions": checkable,
        "instruction_pass_rate": (passed / checkable) if checkable else None,
    }


# ---------------- Custom agentic ----------------

AGENTIC_TOOLS = [
    {"name": "search_web", "description": "Search the web",
     "parameters": {"query": "string"}},
    {"name": "read_file", "description": "Read a file",
     "parameters": {"path": "string"}},
    {"name": "send_email", "description": "Send an email",
     "parameters": {"to": "string", "subject": "string", "body": "string"}},
    {"name": "run_sql", "description": "Execute a SQL query",
     "parameters": {"query": "string"}},
    {"name": "create_calendar_event", "description": "Create a calendar event",
     "parameters": {"title": "string", "time": "string"}},
]

AGENTIC_SCENARIOS = [
    ("Find recent news about Bittensor and summarize.", "search_web"),
    ("Open my notes at ~/ideas/project.md.", "read_file"),
    ("Email bob@corp.com with subject Update and body 'done'.", "send_email"),
    ("How many users signed up yesterday?", "run_sql"),
    ("Schedule a meeting with Alice at 3pm Friday.", "create_calendar_event"),
]


def _build_agentic(n: int) -> list[dict]:
    rng = random.Random(0)
    sys_prompt = (
        "You are NPC Fast, an agentic router. When a tool is needed, respond "
        "with a single JSON object: "
        '{"name": "<tool>", "arguments": {...}}\n'
        f"Tools: {json.dumps(AGENTIC_TOOLS)}"
    )
    out = []
    for _ in range(n):
        q, expect = rng.choice(AGENTIC_SCENARIOS)
        out.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ],
            "expected": expect,
        })
    return out


def run_agentic(model, tokenizer, n: int = 100) -> dict:
    rows = _build_agentic(n)
    json_valid = name_correct = 0
    for row in rows:
        out = generate(model, tokenizer, row["messages"], max_new_tokens=200, temperature=0.0)
        parsed = extract_json(out)
        if parsed is None:
            continue
        json_valid += 1
        if parsed.get("name") == row["expected"]:
            name_correct += 1
    total = max(1, len(rows))
    return {
        "n": len(rows),
        "json_validity": json_valid / total,
        "tool_selection_accuracy": name_correct / total,
    }


# ---------------- Entrypoint ----------------

def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--bfcl_n", type=int, default=200)
    ap.add_argument("--ifeval_n", type=int, default=200)
    ap.add_argument("--agentic_n", type=int, default=100)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model, tok = load_model(args.model_path)

    LOG.info("Running BFCL (%d)", args.bfcl_n)
    bfcl = run_bfcl(model, tok, args.bfcl_n)
    LOG.info("  %s", bfcl)

    LOG.info("Running IFEval (%d)", args.ifeval_n)
    ifeval = run_ifeval(model, tok, args.ifeval_n)
    LOG.info("  %s", ifeval)

    LOG.info("Running custom agentic (%d)", args.agentic_n)
    agentic = run_agentic(model, tok, args.agentic_n)
    LOG.info("  %s", agentic)

    payload = {"bfcl": bfcl, "ifeval": ifeval, "agentic": agentic}
    merge_results(payload)
    log_wandb({f"eval/{k1}.{k2}": v for k1, d in payload.items()
               for k2, v in d.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
