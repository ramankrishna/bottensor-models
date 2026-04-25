"""
Request orchestration.

Three pipelines, one entrypoint per public model name:

  * "npc-fin-32b"  → direct_fin   (BACKWARD COMPAT — existing prod behavior)
  * "npc-fast"     → direct_fast
  * "npc"          → MoM (router + conditional escalation)

All pipelines return an object that the main layer turns into either a
regular JSON response or a StreamingResponse. The pipeline layer owns:
  - upstream selection
  - fallback-on-failure (NPC Fin OOM → NPC Fast with a note)
  - stats accounting
  - extra response headers (X-NPC-Model / X-NPC-Route-Reason / X-NPC-Route-Time-Ms)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from config import CONFIG
from models import model_manager
from router import decide_route, RouteDecision

log = logging.getLogger("npc-mom")


UNAVAILABLE_NOTE = (
    "\n\n[Note: NPC Fin is temporarily unavailable, "
    "answering with NPC Fast instead.]"
)


# ----------------------------------------------------------------------
# Result wrappers
# ----------------------------------------------------------------------
@dataclass
class PipelineResult:
    """
    Non-streaming result. Caller serializes `body` (already OpenAI-shaped)
    and emits `headers` alongside.
    """
    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    model_served: str = ""
    route_decision: Optional[str] = None
    route_reason: Optional[str] = None
    route_time_ms: Optional[float] = None
    generation_time_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class PipelineStream:
    """Streaming result. `stream` yields raw SSE bytes."""
    status_code: int
    stream: AsyncGenerator[bytes, None]
    headers: Dict[str, str] = field(default_factory=dict)
    model_served: str = ""
    route_decision: Optional[str] = None
    route_reason: Optional[str] = None
    route_time_ms: Optional[float] = None


# ----------------------------------------------------------------------
# Direct pipelines
# ----------------------------------------------------------------------
async def direct(
    upstream: str,       # model_manager.NPC_FAST | NPC_FIN
    request_body: Dict[str, Any],
    public_model: str,   # the value to echo back in `model` field
    stream: bool,
    fallback_on_error: bool = False,
) -> PipelineResult | PipelineStream:
    """
    Forward a chat completion to a specific upstream. No routing.

    If `fallback_on_error` and the upstream fails, falls back to NPC Fast
    and prepends a note to the response (used only for `model="npc"` →
    npc_fin when NPC Fin is down; direct npc-fin-32b failures surface as 5xx).
    """
    started = time.monotonic()

    if stream:
        return await _stream_from_upstream(
            upstream=upstream,
            request_body=request_body,
            public_model=public_model,
            started=started,
            fallback_on_error=fallback_on_error,
            route_decision=None,
            route_reason=None,
            route_time_ms=None,
        )

    return await _nonstream_from_upstream(
        upstream=upstream,
        request_body=request_body,
        public_model=public_model,
        started=started,
        fallback_on_error=fallback_on_error,
        route_decision=None,
        route_reason=None,
        route_time_ms=None,
    )


# ----------------------------------------------------------------------
# MoM pipeline
# ----------------------------------------------------------------------
async def mom(
    request_body: Dict[str, Any],
    stream: bool,
) -> PipelineResult | PipelineStream:
    """
    Mixture of Models:
      1) Call NPC Fast with ROUTER_SYSTEM_PROMPT (non-stream) → RouteDecision
      2) route=self    → NPC Fast generates final answer (no router prompt)
      3) route=npc_fin → NPC Fin generates; fallback to NPC Fast on failure.
    """
    messages = request_body.get("messages") or []
    decision: RouteDecision = await decide_route(messages)
    await model_manager.stats.incr_route(decision.route, decision.route_time_ms)

    if decision.route == "self":
        upstream = model_manager.NPC_FAST
        public_served = "npc-fast"
        fallback = False
    else:
        upstream = model_manager.NPC_FIN
        public_served = "npc-fin-32b"
        fallback = True   # fallback to npc-fast if fin errors

    started = time.monotonic()

    # Build the forward body: strip our MoM "npc" model name (upstream wouldn't
    # recognize it) — the forwarder re-writes `model` to the upstream id anyway.
    forward_body = dict(request_body)
    forward_body.pop("model", None)

    if stream:
        return await _stream_from_upstream(
            upstream=upstream,
            request_body=forward_body,
            public_model=public_served,
            started=started,
            fallback_on_error=fallback,
            route_decision=decision.route,
            route_reason=decision.reason,
            route_time_ms=decision.route_time_ms,
        )

    return await _nonstream_from_upstream(
        upstream=upstream,
        request_body=forward_body,
        public_model=public_served,
        started=started,
        fallback_on_error=fallback,
        route_decision=decision.route,
        route_reason=decision.reason,
        route_time_ms=decision.route_time_ms,
    )


# ----------------------------------------------------------------------
# Non-streaming
# ----------------------------------------------------------------------
async def _nonstream_from_upstream(
    *,
    upstream: str,
    request_body: Dict[str, Any],
    public_model: str,
    started: float,
    fallback_on_error: bool,
    route_decision: Optional[str],
    route_reason: Optional[str],
    route_time_ms: Optional[float],
) -> PipelineResult:
    try:
        resp = await model_manager.chat_completion(
            upstream_name=upstream,
            payload=request_body,
            timeout_s=CONFIG["generation_timeout_s"],
        )
        if resp.status_code != 200:
            if fallback_on_error and upstream == model_manager.NPC_FIN:
                log.warning(
                    "npc-fin-failed-falling-back",
                    extra={"status": resp.status_code, "body": resp.text[:500]},
                )
                return await _nonstream_fallback_fast(
                    request_body=request_body,
                    started=started,
                    route_decision=route_decision,
                    route_reason=route_reason,
                    route_time_ms=route_time_ms,
                )
            return _error_result(
                status_code=resp.status_code,
                message=_safe_text(resp.text),
                type_="upstream_error",
                public_model=public_model,
                route_decision=route_decision,
                route_reason=route_reason,
                route_time_ms=route_time_ms,
            )
    except httpx.RequestError as exc:
        log.warning(
            "upstream-request-error",
            extra={"upstream": upstream, "error": str(exc)},
        )
        if fallback_on_error and upstream == model_manager.NPC_FIN:
            return await _nonstream_fallback_fast(
                request_body=request_body,
                started=started,
                route_decision=route_decision,
                route_reason=route_reason,
                route_time_ms=route_time_ms,
            )
        return _error_result(
            status_code=503,
            message=f"Upstream unreachable: {type(exc).__name__}",
            type_="upstream_unreachable",
            public_model=public_model,
            route_decision=route_decision,
            route_reason=route_reason,
            route_time_ms=route_time_ms,
        )

    body = resp.json()
    body["model"] = public_model

    # Token accounting — vLLM echoes OpenAI-style `usage`
    usage = body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    headers = _make_headers(
        public_model=public_model,
        route_reason=route_reason,
        route_time_ms=route_time_ms,
        include_route_headers=route_decision is not None,
    )

    generation_time_ms = round((time.monotonic() - started) * 1000, 2)

    return PipelineResult(
        status_code=200,
        body=body,
        headers=headers,
        model_served=public_model,
        route_decision=route_decision,
        route_reason=route_reason,
        route_time_ms=route_time_ms,
        generation_time_ms=generation_time_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def _nonstream_fallback_fast(
    *,
    request_body: Dict[str, Any],
    started: float,
    route_decision: Optional[str],
    route_reason: Optional[str],
    route_time_ms: Optional[float],
) -> PipelineResult:
    try:
        resp = await model_manager.chat_completion(
            upstream_name=model_manager.NPC_FAST,
            payload=request_body,
            timeout_s=CONFIG["generation_timeout_s"],
        )
    except httpx.RequestError as exc:
        return _error_result(
            status_code=503,
            message=f"Both upstreams unavailable: {type(exc).__name__}",
            type_="all_upstreams_down",
            public_model="npc-fast",
            route_decision=route_decision,
            route_reason=route_reason,
            route_time_ms=route_time_ms,
        )

    if resp.status_code != 200:
        return _error_result(
            status_code=resp.status_code,
            message=_safe_text(resp.text),
            type_="upstream_error",
            public_model="npc-fast",
            route_decision=route_decision,
            route_reason=route_reason,
            route_time_ms=route_time_ms,
        )

    body = resp.json()
    body["model"] = "npc-fast"

    # Prepend fallback note to the assistant reply
    try:
        msg = body["choices"][0]["message"]
        original_content = msg.get("content", "") or ""
        msg["content"] = UNAVAILABLE_NOTE.strip() + "\n\n" + original_content
    except (KeyError, IndexError, TypeError):
        pass

    usage = body.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    headers = _make_headers(
        public_model="npc-fast",
        route_reason=(route_reason or "") + " | fallback-npc-fin-down",
        route_time_ms=route_time_ms,
        include_route_headers=True,
    )
    headers["X-NPC-Fallback"] = "npc-fin-unavailable"

    return PipelineResult(
        status_code=200,
        body=body,
        headers=headers,
        model_served="npc-fast",
        route_decision=route_decision,
        route_reason=(route_reason or "") + " | fallback-npc-fin-down",
        route_time_ms=route_time_ms,
        generation_time_ms=round((time.monotonic() - started) * 1000, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------
async def _stream_from_upstream(
    *,
    upstream: str,
    request_body: Dict[str, Any],
    public_model: str,
    started: float,
    fallback_on_error: bool,
    route_decision: Optional[str],
    route_reason: Optional[str],
    route_time_ms: Optional[float],
) -> PipelineStream:
    """
    Returns a streaming response. The generator rewrites each SSE chunk's
    `model` field to the public name, and on upstream connection failure,
    optionally falls back to NPC Fast.
    """

    async def _gen() -> AsyncGenerator[bytes, None]:
        emitted_any = False
        try:
            async for raw in model_manager.chat_completion_stream(
                upstream_name=upstream,
                payload=request_body,
                timeout_s=CONFIG["generation_timeout_s"],
            ):
                emitted_any = True
                yield _rewrite_stream_chunk(raw, public_model)
            return
        except httpx.RequestError as exc:
            log.warning(
                "stream-upstream-error",
                extra={"upstream": upstream, "error": str(exc), "emitted": emitted_any},
            )

        # Connection failed. If no bytes were emitted yet and we're allowed to
        # fallback, stream from NPC Fast with a prepended note.
        if not emitted_any and fallback_on_error and upstream == model_manager.NPC_FIN:
            # Emit the note as a first chunk
            note_chunk = _make_sse_chunk(
                public_model="npc-fast",
                delta_content=UNAVAILABLE_NOTE.strip() + "\n\n",
            )
            yield note_chunk
            try:
                async for raw in model_manager.chat_completion_stream(
                    upstream_name=model_manager.NPC_FAST,
                    payload=request_body,
                    timeout_s=CONFIG["generation_timeout_s"],
                ):
                    yield _rewrite_stream_chunk(raw, "npc-fast")
                return
            except httpx.RequestError as exc2:
                err = _make_sse_error(503, f"All upstreams down: {type(exc2).__name__}")
                yield err
                yield b"data: [DONE]\n\n"
                return

        # No fallback — surface error
        err = _make_sse_error(503, "Upstream unavailable")
        yield err
        yield b"data: [DONE]\n\n"

    headers = _make_headers(
        public_model=public_model,
        route_reason=route_reason,
        route_time_ms=route_time_ms,
        include_route_headers=route_decision is not None,
    )

    return PipelineStream(
        status_code=200,
        stream=_gen(),
        headers=headers,
        model_served=public_model,
        route_decision=route_decision,
        route_reason=route_reason,
        route_time_ms=route_time_ms,
    )


# ----------------------------------------------------------------------
# SSE helpers
# ----------------------------------------------------------------------
def _rewrite_stream_chunk(raw: bytes, public_model: str) -> bytes:
    """
    Rewrite the `model` field of each SSE data frame to the public name.
    Leaves non-data lines ([DONE], keepalives, comments) untouched.

    Input may contain one or more frames; we process line-by-line.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw

    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        if not line.startswith("data:"):
            out.append(line)
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            out.append(line)
            continue
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict) and "model" in obj:
                obj["model"] = public_model
            out.append("data: " + json.dumps(obj))
        except json.JSONDecodeError:
            out.append(line)
    return ("\n".join(out)).encode("utf-8")


def _make_sse_chunk(public_model: str, delta_content: str) -> bytes:
    chunk = {
        "id": "chatcmpl-fallback-note",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": delta_content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")


def _make_sse_error(status_code: int, message: str) -> bytes:
    err = {
        "error": {
            "message": message,
            "type": "upstream_error",
            "code": str(status_code),
        }
    }
    return f"data: {json.dumps(err)}\n\n".encode("utf-8")


# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------
def _make_headers(
    *,
    public_model: str,
    route_reason: Optional[str],
    route_time_ms: Optional[float],
    include_route_headers: bool,
) -> Dict[str, str]:
    h = {"X-NPC-Model": public_model}
    if include_route_headers:
        if route_reason is not None:
            h["X-NPC-Route-Reason"] = route_reason
        if route_time_ms is not None:
            h["X-NPC-Route-Time-Ms"] = str(int(round(route_time_ms)))
    return h


def _error_result(
    *,
    status_code: int,
    message: str,
    type_: str,
    public_model: str,
    route_decision: Optional[str],
    route_reason: Optional[str],
    route_time_ms: Optional[float],
) -> PipelineResult:
    body = {
        "error": {
            "message": message,
            "type": type_,
            "code": str(status_code),
        }
    }
    headers = _make_headers(
        public_model=public_model,
        route_reason=route_reason,
        route_time_ms=route_time_ms,
        include_route_headers=route_decision is not None,
    )
    return PipelineResult(
        status_code=status_code,
        body=body,
        headers=headers,
        model_served=public_model,
        route_decision=route_decision,
        route_reason=route_reason,
        route_time_ms=route_time_ms,
    )


def _safe_text(s: str) -> str:
    s = s.strip()
    return s[:2000] if len(s) > 2000 else s
