"""Generate router SFT dataset: 300 self + 200 npc_fin pairs.

Each example is a chat: {system, user, assistant} where the assistant
emits exactly one JSON object matching the schema. We hand-curate a
diverse seed pool and expand with deterministic variations.
"""
from __future__ import annotations
import json, random
from pathlib import Path

random.seed(42)

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

# --- SELF examples -----------------------------------------------------------
# Categories: arithmetic, translation, formatting, identity, trivia, code,
# tool-call, definition, summarization, rewriting, chit-chat.

SELF_PAIRS = [
    ("What is 17 * 23?", "arithmetic"),
    ("Compute 144 / 12.", "arithmetic"),
    ("Add 7, 14, 21, 28.", "arithmetic"),
    ("What is 2 to the 10th power?", "arithmetic"),
    ("Convert 98 F to Celsius.", "unit conversion"),
    ("Convert 5 miles to kilometers.", "unit conversion"),
    ("Convert 1 GB to MB.", "unit conversion"),
    ("Translate good morning to French.", "translation"),
    ("Translate hello to Spanish.", "translation"),
    ("Translate thank you to Japanese.", "translation"),
    ("How do you say water in German?", "translation"),
    ("What is your name?", "identity"),
    ("Who built you?", "identity"),
    ("What model are you?", "identity"),
    ("What version are you?", "identity"),
    ("What tools do you have access to?", "identity"),
    ("What can you do?", "identity"),
    ("Define mitochondrion.", "definition"),
    ("Define photosynthesis.", "definition"),
    ("Define recursion.", "definition"),
    ("What does TCP stand for?", "definition"),
    ("What is a haiku?", "definition"),
    ("Reverse the string anthropic.", "string manipulation"),
    ("Reverse the string hello.", "string manipulation"),
    ("Capitalize: npc fast is fast.", "string manipulation"),
    ("Lowercase this: HELLO WORLD.", "string manipulation"),
    ("Count the letters in mississippi.", "string manipulation"),
    ("Is the word level a palindrome?", "string check"),
    ("Reply in all lowercase.", "formatting"),
    ("Reply in all caps.", "formatting"),
    ("Write a sentence without the letter e.", "formatting"),
    ("Output the JSON: {\"ok\": true}.", "simple JSON"),
    ("Output the JSON: {\"x\": 1, \"y\": 2}.", "simple JSON"),
    ("Return a JSON list of primary colors.", "simple JSON"),
    ("Give me a haiku about rain.", "creative short"),
    ("Give me a haiku about autumn.", "creative short"),
    ("Write a two-line poem about coffee.", "creative short"),
    ("Tell me a one-sentence joke.", "creative short"),
    ("Summarize this in 5 words: The quick brown fox jumps.", "short summary"),
    ("Summarize: Cats are small carnivorous mammals.", "short summary"),
    ("Paraphrase: It is raining heavily outside.", "paraphrase"),
    ("List the first 5 prime numbers.", "simple list"),
    ("List three primary colors.", "simple list"),
    ("List the days of the week.", "simple list"),
    ("Name three planets.", "simple list"),
    ("Is Python interpreted or compiled?", "simple factual"),
    ("Is the Earth round?", "simple factual"),
    ("What is the capital of France?", "simple factual"),
    ("What is the speed of light?", "simple factual"),
    ("Open the file ~/notes.txt.", "tool call"),
    ("Read the config file.", "tool call"),
    ("Send a hello email to alice@x.com.", "tool call"),
    ("Search the web for weather NYC.", "tool call"),
    ("Ping google.com.", "tool call"),
    ("Run ls in the current directory.", "tool call"),
    ("Create a new folder called tmp.", "tool call"),
    ("Delete the file old.log.", "tool call"),
    ("Explain recursion in one sentence.", "short explanation"),
    ("Explain what a cache is.", "short explanation"),
    ("Explain polymorphism briefly.", "short explanation"),
    ("What is Big-O notation?", "short explanation"),
    ("Tell me the date format ISO-8601 uses.", "factual lookup"),
    ("What year did the moon landing happen?", "factual lookup"),
    ("Who wrote Hamlet?", "factual lookup"),
    ("What is the largest planet?", "factual lookup"),
    ("Write a Python function that adds two numbers.", "short code"),
    ("Write a one-line Python to reverse a list.", "short code"),
    ("Write a bash command to list hidden files.", "short code"),
    ("Write a regex for a 5-digit zip code.", "short code"),
    ("What is 2+2?", "trivial arithmetic"),
    ("What is 10 minus 3?", "trivial arithmetic"),
    ("How many days in a week?", "trivial factual"),
    ("How many hours in a day?", "trivial factual"),
    ("Is 17 a prime number?", "simple factual"),
    ("Is 21 a prime number?", "simple factual"),
    ("What is the square root of 81?", "arithmetic"),
    ("Round 3.7 to the nearest integer.", "arithmetic"),
    ("Say hello.", "chit-chat"),
    ("How are you?", "chit-chat"),
    ("Thanks!", "chit-chat"),
    ("Tell me something interesting.", "chit-chat"),
    ("What time is it?", "simple lookup"),
    ("What day is today?", "simple lookup"),
    ("Format this as markdown bold: hello.", "formatting"),
    ("Add a comma after each word: apple banana cherry.", "formatting"),
    ("Remove duplicates from [1,2,2,3,3,3].", "simple list op"),
    ("Sort this list ascending: [5,2,8,1].", "simple list op"),
    ("Find the max of 4, 9, 2, 7.", "simple list op"),
    ("What is the factorial of 5?", "arithmetic"),
    ("Is 100 even or odd?", "simple factual"),
    ("Convert 3pm to 24-hour format.", "simple conversion"),
    ("Convert 100 USD to a generic label (no FX needed).", "simple conversion"),
    ("Write a git commit message for a bugfix.", "short writing"),
    ("Suggest a short variable name for a timestamp.", "short writing"),
    ("Generate a random 6-letter word.", "creative short"),
    ("Pick a random color name.", "creative short"),
    ("Name a common Python library.", "simple lookup"),
    ("Name a popular web framework.", "simple lookup"),
    ("What HTTP status means not found?", "simple factual"),
    ("What is HTML short for?", "simple factual"),
    ("Describe the color blue.", "short explanation"),
    ("Describe what a tree is.", "short explanation"),
]

# --- NPC_FIN examples --------------------------------------------------------
# Deep multi-step finance tasks only.
FIN_PAIRS = [
    ("Given TSLA 10-K, analyze revenue concentration risk across segments.", "multi-step financial analysis"),
    ("Build a DCF model for NVDA with terminal growth of 3%.", "DCF model"),
    ("Compare systemic risk of US regional banks vs money-center banks.", "deep banking analysis"),
    ("Model yield-curve impact of Fed dot plot on a 60/40 portfolio.", "multi-step macro model"),
    ("Analyze options flow for AMD and infer institutional positioning.", "options flow analysis"),
    ("Walk through merger arbitrage for the MSFT-ATVI deal.", "merger arb step-through"),
    ("Quantify basis risk in a Treasury futures hedge of a corp bond book.", "quantitative hedging"),
    ("Decompose JPM net interest margin into volume, rate, and mix.", "NIM decomposition"),
    ("Model counterparty risk for a prime broker in a liquidity crisis.", "counterparty risk model"),
    ("Derive gamma exposure profile for market-makers on SPX.", "options gamma analytics"),
    ("Scenario-analyze a 200bp rate shock on a leveraged mortgage REIT.", "rate shock scenario"),
    ("Sum-of-the-parts valuation on META including Reality Labs and AI.", "SOTP valuation"),
    ("Apply the Merton model to estimate a firm default probability.", "credit modeling"),
    ("Perform factor attribution on a long/short tech book.", "factor attribution"),
    ("Convex bond hedging problem using duration and convexity.", "bond hedging math"),
    ("Capital structure implications of a $50B debt-funded buyback.", "capital structure analysis"),
    ("Explain VWAP slippage math on a block trade with TWAP execution.", "execution microstructure"),
    ("Derive Black-Scholes IV surface skew for SPY 30d options.", "options volatility modeling"),
    ("Model LCR impact of a run-off scenario on a mid-sized bank.", "regulatory liquidity model"),
    ("Compare CCAR 2024 vs 2023 stress assumptions and quantify shortfalls.", "regulatory stress analysis"),
    ("Build a 3-statement model for AAPL with quarterly projections.", "3-statement model"),
    ("Value a PIK-toggle note under three default scenarios.", "credit structured product"),
    ("Analyze cross-currency basis swap P&L under a dollar funding squeeze.", "FX/rates analytics"),
    ("Derive the CVA adjustment for an uncollateralized IRS.", "derivatives CVA"),
    ("Model the portfolio Sharpe under dynamic correlation regime shifts.", "portfolio optimization"),
    ("Build an LBO model with 5x leverage and 5-year exit.", "LBO model"),
    ("Analyze the term premium component of the 10Y UST yield.", "fixed-income term-premium"),
    ("Quantify wrong-way risk in a CDS written on the counterparty.", "wrong-way risk"),
    ("Estimate market impact of a $500M buy program in a mid-cap stock.", "market impact model"),
    ("Compute the FTP (funds transfer pricing) for a 5Y mortgage.", "FTP math"),
    ("Model the P&L of a calendar spread into an earnings event.", "options strategy P&L"),
    ("Quantify the basis risk between EUR/USD spot and 3M forward under a dollar shortage.", "FX basis risk"),
    ("Analyze the credit impairment cascade of a CRE REIT under 30% rent declines.", "CRE stress"),
    ("Derive duration-matched immunization for a pension liability.", "ALM immunization"),
    ("Model the tail risk of a concentrated equity position using EVT.", "tail risk model"),
    ("Compute risk-neutral default probabilities from a CDS curve.", "credit curve math"),
    ("Build a multi-factor risk model for an emerging-markets bond book.", "EM risk model"),
    ("Perform attribution on a multi-strategy hedge fund across 6 sleeves.", "multi-strat attribution"),
    ("Model the carry-and-roll P&L of a 10Y-2Y steepener trade.", "rates trade P&L"),
    ("Analyze the liquidity cost of unwinding a $2B high-yield position.", "liquidity cost analysis"),
]

# --- Expand with rephrasings --------------------------------------------------
SELF_PREFIXES = ["", "Please ", "Can you ", "Hey, ", "Quick: ", "Just "]
FIN_PREFIXES = ["", "I need you to ", "Please help me: ", "Work through this: ", "Task: "]

def expand(pairs, prefixes, target_n):
    out = []
    i = 0
    while len(out) < target_n:
        q, reason = pairs[i % len(pairs)]
        p = prefixes[(i // len(pairs)) % len(prefixes)]
        out.append((p + q, reason))
        i += 1
    return out[:target_n]

self_full = expand(SELF_PAIRS, SELF_PREFIXES, 300)
fin_full = expand(FIN_PAIRS, FIN_PREFIXES, 200)

examples = []
for q, reason in self_full:
    examples.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": json.dumps({"route": "self", "reason": reason})},
    ]})
for q, reason in fin_full:
    examples.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": json.dumps({"route": "npc_fin", "reason": reason})},
    ]})

random.shuffle(examples)

out_path = Path("/workspace/npc-fast-trainer/data/router_sft/train.jsonl")
with out_path.open("w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

# Also dump a 50-example held-out set with different prefixes for smoke-check
heldout = []
for q, reason in SELF_PAIRS[:30]:
    heldout.append({"query": "Hmm — " + q, "label": "self"})
for q, reason in FIN_PAIRS[:20]:
    heldout.append({"query": "I want to: " + q, "label": "npc_fin"})
with open("/workspace/npc-fast-trainer/data/router_sft/heldout.jsonl", "w") as f:
    for h in heldout:
        f.write(json.dumps(h) + "\n")

print(f"Wrote {len(examples)} training examples to {out_path}")
print(f"Wrote {len(heldout)} held-out examples")
