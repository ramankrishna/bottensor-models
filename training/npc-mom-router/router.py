"""
Routing decision — call NPC Fast with a router system prompt, parse the JSON.

Design notes:
  * The routing call is ALWAYS non-streaming so we can read the full JSON
    decision in one hop. The final generation (self or npc_fin) is then
    streamed separately by pipeline.py.
  * We use max_tokens=64 and temperature=0 for the routing call to keep it
    deterministic and fast (<200ms on a 1.7B model).
  * JSON extraction is forgiving: we try strict parse, then a regex fallback,
    then a keyword fallback. If all fail we default to "self" — this matches
    the product rule "Default: route=self".
  * Reasons are clipped to 200 chars for header safety (HTTP headers don't
    like arbitrary long strings).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from config import CONFIG
from models import model_manager

log = logging.getLogger("npc-mom")


ROUTER_SYSTEM_PROMPT = """You are NPC Fast, a routing model for the NPC Model Family by Bottensor.

Decide where to send the user query:
- route: "self"     → general knowledge, simple tasks, tool calls, code, math, translation, facts, definitions, everyday questions.
- route: "npc_fin"  → ANY query about: stocks, tickers, equities, bonds, DCF, valuation, portfolio, risk, beta, Sharpe, CAPM, options, derivatives, Black-Scholes, Greeks, crypto, BTC, ETH, tokens, on-chain, yield, staking, macro, CPI, rates, Fed, recession, forex, commodities, earnings, P/E, EV/EBITDA, M&A, IPO, SPAC, hedge fund, private equity, VC, TVL, DeFi, AMM, perps, LBO, WACC, NPV, IRR, cash flow, balance sheet, financial statements, accounting, audit, tax strategy, compliance, SEC, quantitative finance, trading strategy, technical analysis.

Rule: if the query touches any financial, investing, trading, or macroeconomic concept — choose npc_fin.
Otherwise choose self.

Respond in the exact JSON schema provided: {"route": "self" | "npc_fin", "reason": "<brief reason>"}"""


# Heuristic rescue: when the model picks `self` but its own reason text clearly
# describes a finance query, flip the route. This is a common 4-bit-quant
# failure mode and the reason-text signal is very reliable when it triggers.
_FIN_KEYWORDS = [
    "dcf", "valuation", "equity", "stock", "ticker", "portfolio", "capm",
    "beta", "sharpe", "black-scholes", "black scholes", "option", "deriv",
    "crypto", "bitcoin", "btc", "ethereum", "eth", "token", "on-chain",
    "on chain", "yield", "staking", "defi", "amm", "perp", "perps",
    "macro", "inflation", "cpi", "fed ", "fomc", "recession", "forex",
    "commodit", "earnings", "p/e", "pe ratio", "ev/ebitda", "ebitda",
    "m&a", "ipo", "spac", "hedge fund", "lbo", "wacc", "npv", "irr",
    "cash flow", "balance sheet", "financial", "finance", "trading",
    "technical analysis", "fundamental analysis", "market analysis",
    "quantitative", "investment", "investing",
    # Rates / fixed income vocabulary
    "interest rate", "rate cut", "rate hike", "rate rise", "rate fall",
    " rates ", "yield curve", "bond", "treasury", "treasuries",
    "duration", "convexity", "credit spread", "coupon",
    # Equity research verbs
    "buy the", "sell the", "short the", "long the", "overweight",
    "underweight", "rebalance", "asset allocation",
    # Market-related phrases (anchored to reduce false positives)
    "market outlook", "market analysis", "market view", "price action",
    # Currency pairs (major forex)
    "eur/usd", "gbp/usd", "usd/jpy", "usd/cad", "aud/usd", "nzd/usd",
    "eur/jpy", "gbp/jpy", "eur/gbp",
    "currency pair", " fx ", "exchange rate",
]


VALID_ROUTES = {"self", "npc_fin"}
_REASON_MAX = 200


# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------
@dataclass
class RouteDecision:
    route: str                  # "self" or "npc_fin"
    reason: str                 # free-form, <= 200 chars, header-safe
    route_time_ms: float        # wall-clock of the routing call
    raw: Optional[str] = None   # raw router response (for debugging)


# ----------------------------------------------------------------------
# Main entrypoint
# ----------------------------------------------------------------------
async def decide_route(messages: List[Dict[str, Any]]) -> RouteDecision:
    """
    Make a routing decision. On any error, defaults to "self" so the request
    still succeeds (degraded mode).
    """
    started = time.monotonic()

    # Build the routing prompt. We prepend our system prompt and keep the
    # user's messages verbatim. Any system message from the user is demoted
    # to a user-role message so our router prompt is authoritative.
    routing_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ]
    for m in messages:
        role = m.get("role", "user")
        # Demote user system prompts so they don't confuse the router.
        if role == "system":
            routing_messages.append({
                "role": "user",
                "content": _as_text(m.get("content")),
            })
        else:
            routing_messages.append({
                "role": role,
                "content": m.get("content"),
            })

    # Force valid JSON shape via vLLM structured-outputs (guided_json).
    # Falls back gracefully on upstreams that don't honor the field.
    _ROUTE_SCHEMA = {
        "type": "object",
        "properties": {
            "route":  {"type": "string", "enum": ["self", "npc_fin"]},
            "reason": {"type": "string", "maxLength": 200},
        },
        "required": ["route", "reason"],
        "additionalProperties": False,
    }

    payload: Dict[str, Any] = {
        "model": model_manager.upstream_model_id(model_manager.NPC_FAST),
        "messages": routing_messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "n": 1,
        "stream": False,
        # vLLM 0.18+ structured outputs (preferred)
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "route_decision",
            "schema": _ROUTE_SCHEMA,
            "strict": True,
        }},
        # Legacy vLLM guided_json parameter (harmless if unknown)
        "guided_json": _ROUTE_SCHEMA,
    }

    try:
        resp = await model_manager.chat_completion(
            model_manager.NPC_FAST,
            payload,
            timeout_s=CONFIG["router_timeout_s"],
        )
    except httpx.RequestError as exc:
        elapsed = (time.monotonic() - started) * 1000
        log.warning(
            "router-upstream-error",
            extra={"error": str(exc), "route_time_ms": elapsed},
        )
        return RouteDecision(
            route="self",
            reason="router-unavailable-defaulting-self",
            route_time_ms=elapsed,
        )

    elapsed_ms = (time.monotonic() - started) * 1000

    if resp.status_code != 200:
        log.warning(
            "router-bad-status",
            extra={"status": resp.status_code, "body": resp.text[:500]},
        )
        return RouteDecision(
            route="self",
            reason=f"router-http-{resp.status_code}",
            route_time_ms=elapsed_ms,
        )

    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        log.warning("router-malformed-response", extra={"error": str(exc)})
        return RouteDecision(
            route="self",
            reason="router-malformed-defaulting-self",
            route_time_ms=elapsed_ms,
        )

    route, reason = _parse_route(content)

    # Rescue pass: 4-bit-quant drift often sets route=self while the user
    # query (or the model's own reason) clearly describes a finance topic.
    # Check BOTH signals and flip when either trips.
    rescued = False
    if route == "self":
        user_text = _last_user_text(messages)
        reason_fin = _looks_financial(reason)
        query_fin = _looks_financial(user_text)
        if reason_fin or query_fin:
            route = "npc_fin"
            tag = []
            if reason_fin: tag.append("reason")
            if query_fin:  tag.append("query")
            reason = f"rescued-via-{'-'.join(tag)}-keywords | {reason}"
            rescued = True

    if reason.startswith("router-default") or rescued:
        log.info(
            "router-decision",
            extra={
                "route": route, "reason": reason, "rescued": rescued,
                "raw": (content or "")[:400], "elapsed_ms": elapsed_ms,
            },
        )

    return RouteDecision(
        route=route,
        reason=reason,
        route_time_ms=elapsed_ms,
        raw=content,
    )


def _looks_financial(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(kw in low for kw in _FIN_KEYWORDS)


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract the most recent user turn as plain text."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return _as_text(m.get("content"))
    return ""


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)
_ROUTE_KW = re.compile(r'"?route"?\s*[:=]\s*"?(self|npc[_-]?fin)"?', re.IGNORECASE)
_REASON_KW = re.compile(r'"?reason"?\s*[:=]\s*"([^"]*)"', re.IGNORECASE)


def _parse_route(text: str) -> tuple[str, str]:
    """
    Try multiple parsers in order of strictness. Always returns a valid
    (route, reason) pair.
    """
    if not text:
        return "self", "empty-router-output"

    # 1) Strict JSON parse (first {...} block or the whole thing)
    candidates: List[str] = []
    text_strip = text.strip()
    if text_strip.startswith("{"):
        candidates.append(text_strip)
    candidates.extend(_JSON_BLOCK.findall(text))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        route = _normalize_route(obj.get("route"))
        reason = _clip_reason(obj.get("reason"))
        if route:
            return route, reason or "router-no-reason"

    # 2) Regex keyword match (tolerates pipe-notation / loose YAML)
    m = _ROUTE_KW.search(text)
    if m:
        route = _normalize_route(m.group(1))
        reason_m = _REASON_KW.search(text)
        reason = _clip_reason(reason_m.group(1) if reason_m else None)
        if route:
            return route, reason or "router-regex-match"

    # 3) Fallback: look for bare keyword "npc_fin" anywhere
    low = text.lower()
    if "npc_fin" in low or "npc-fin" in low:
        return "npc_fin", "router-keyword-npc_fin"

    return "self", "router-default-self"


def _normalize_route(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower().replace("-", "_")
    if s in VALID_ROUTES:
        return s
    if s in ("fin", "npcfin"):
        return "npc_fin"
    return None


def _clip_reason(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw)
    # Remove CR/LF/tab so it's safe as an HTTP header
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
    if len(s) > _REASON_MAX:
        s = s[: _REASON_MAX - 1] + "…"
    return s


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI-style multipart content → join text parts
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "\n".join(parts)
    return str(content) if content is not None else ""
