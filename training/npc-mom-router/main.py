"""
NPC Mixture of Models — FastAPI entrypoint.

Bottensor API | dude.npc | Bottensor (a Falcon Hash company)

Endpoints:
    POST /v1/chat/completions   — OpenAI-compatible, auth required
    GET  /v1/models             — no auth
    GET  /health                — no auth, full stats
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from auth import AuthError, auth_service
from config import CONFIG, PUBLIC_MODELS
from logging_config import configure_logging, log_request
from models import model_manager
from pipeline import PipelineResult, PipelineStream, direct, mom
from rate_limiter import RateLimitExceeded, ReservationReceipt, rate_limiter
from schemas import (
    ChatCompletionRequest,
    ErrorBody,
    ErrorEnvelope,
    HealthResponse,
    ModelHealth,
    ModelInfo,
    ModelList,
)

# ----------------------------------------------------------------------
# Startup / shutdown
# ----------------------------------------------------------------------
configure_logging(level=CONFIG["log_level"], request_log_file=CONFIG["log_file"])
log = logging.getLogger("npc-mom")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "startup",
        extra={
            "api_name": CONFIG["brand"]["api_name"],
            "company": CONFIG["brand"]["company"],
            "models": PUBLIC_MODELS,
        },
    )
    auth_service.connect()
    await model_manager.start()
    try:
        yield
    finally:
        log.info("shutdown")
        await model_manager.stop()
        auth_service.close()


app = FastAPI(
    title="Bottensor API — NPC Mixture of Models",
    version="1.0.0",
    description=(
        "Unified OpenAI-compatible endpoint serving NPC Fast (1.7B) and "
        "NPC Fin 32B behind a single API. The `npc` model auto-routes; "
        "`npc-fast` and `npc-fin-32b` are direct. Bottensor (a Falcon Hash "
        "company) | creator: dude.npc."
    ),
    lifespan=lifespan,
)


# ----------------------------------------------------------------------
# Error translation helpers
# ----------------------------------------------------------------------
def _error_json(status: int, message: str, type_: str = "invalid_request_error",
                code: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorEnvelope(
            error=ErrorBody(message=message, type=type_, code=code)
        ).model_dump(),
    )


@app.exception_handler(AuthError)
async def _auth_error_handler(_request: Request, exc: AuthError) -> JSONResponse:
    return _error_json(exc.status, str(exc), type_="authentication_error",
                       code="unauthorized")


@app.exception_handler(RateLimitExceeded)
async def _rate_error_handler(_request: Request, exc: RateLimitExceeded) -> JSONResponse:
    resp = _error_json(
        429,
        f"Rate limit exceeded ({exc.reason})",
        type_="rate_limit_error",
        code="rate_limited",
    )
    resp.headers["Retry-After"] = str(exc.retry_after)
    return resp


# ----------------------------------------------------------------------
# /v1/models — public
# ----------------------------------------------------------------------
@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    created = int(time.time())
    return ModelList(
        data=[
            ModelInfo(id=m, created=created, owned_by=CONFIG["brand"]["owned_by"])
            for m in PUBLIC_MODELS
        ]
    )


# ----------------------------------------------------------------------
# /health — public
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    fast_status, fast_vram = await model_manager.health(model_manager.NPC_FAST)
    fin_status, fin_vram = await model_manager.health(model_manager.NPC_FIN)

    stats = model_manager.stats

    return HealthResponse(
        status="ok" if fast_status == "loaded" and fin_status == "loaded" else "degraded",
        models={
            "npc-fast": ModelHealth(
                status=fast_status,
                vram_mb=fast_vram if fast_vram is not None else 3400,
                port=model_manager.port(model_manager.NPC_FAST),
            ),
            "npc-fin-32b": ModelHealth(
                status=fin_status,
                vram_mb=fin_vram if fin_vram is not None else 20000,
                port=model_manager.port(model_manager.NPC_FIN),
            ),
        },
        uptime_seconds=int(time.time() - stats.started_at),
        requests_total=stats.requests_total,
        requests_by_model=dict(stats.requests_by_model),
        route_stats={
            "self": stats.route_stats.self_,
            "npc_fin": stats.route_stats.npc_fin,
        },
        avg_route_time_ms=stats.avg_route_time_ms(),
    )


# ----------------------------------------------------------------------
# /v1/chat/completions — authed
# ----------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    t0 = time.monotonic()

    # 1) Parse the body early (we need it for both validation and reservation)
    try:
        raw_body = await request.json()
    except json.JSONDecodeError as exc:
        return _error_json(400, f"Invalid JSON: {exc.msg}", code="invalid_json")

    try:
        chat_req = ChatCompletionRequest.model_validate(raw_body)
    except Exception as exc:  # pydantic ValidationError etc.
        return _error_json(400, f"Invalid request: {exc}", code="invalid_request")

    model_requested = chat_req.model
    model_kind = CONFIG["model_map"].get(model_requested)
    if model_kind is None:
        return _error_json(
            400,
            f"Model not found. Available models: {', '.join(PUBLIC_MODELS)}",
            code="model_not_found",
        )

    # 2) Auth
    api_key = _extract_bearer(authorization)
    key_info = await auth_service.validate(api_key)

    # 3) Enforce per-tier max context (count on prompt chars / 4 heuristic +
    #    the client-requested `max_tokens`).
    tier_cfg = CONFIG["tiers"][key_info.tier]
    est_in_tokens = _estimate_input_tokens(raw_body.get("messages") or [])
    requested_out = int(raw_body.get("max_tokens") or 512)
    total_ctx = est_in_tokens + requested_out
    if total_ctx > tier_cfg["max_context"]:
        return _error_json(
            400,
            f"Context window exceeded for tier '{key_info.tier}': "
            f"{total_ctx} > {tier_cfg['max_context']} tokens",
            code="context_too_large",
        )

    # 4) Reserve rate-limit budget
    try:
        reservation: ReservationReceipt = await rate_limiter.check_and_reserve(
            key=api_key or "",
            tier=key_info.tier,
            estimated_tokens=total_ctx,
        )
    except RateLimitExceeded as exc:
        # Logged by exception handler; also record request as 429
        log_request(
            request_id=request_id,
            api_key_owner=key_info.owner,
            tier=key_info.tier,
            model_requested=model_requested,
            model_served=None,
            route_decision=None,
            route_reason=None,
            route_time_ms=None,
            generation_time_ms=None,
            input_tokens=est_in_tokens,
            output_tokens=None,
            status=429,
            error=exc.reason,
        )
        raise

    # 5) Dispatch to pipeline
    stream = bool(raw_body.get("stream") or False)
    try:
        if model_kind == "mom":
            result = await mom(raw_body, stream=stream)
        elif model_kind == "direct_fast":
            result = await direct(
                upstream=model_manager.NPC_FAST,
                request_body=raw_body,
                public_model="npc-fast",
                stream=stream,
                fallback_on_error=False,
            )
        elif model_kind == "direct_fin":
            # BACKWARD COMPAT: identical semantics to legacy npc-fin-32b API.
            # No routing, no fallback — clients expect the legacy behavior.
            result = await direct(
                upstream=model_manager.NPC_FIN,
                request_body=raw_body,
                public_model="npc-fin-32b",
                stream=stream,
                fallback_on_error=False,
            )
        else:
            # Should be unreachable (model_map validated above), but be defensive
            await reservation.refund()
            return _error_json(500, f"Unhandled model kind: {model_kind}",
                               type_="internal_error", code="internal")

    except Exception as exc:  # pragma: no cover — defensive
        await reservation.refund()
        log.exception("pipeline-unhandled-error")
        log_request(
            request_id=request_id, api_key_owner=key_info.owner,
            tier=key_info.tier, model_requested=model_requested,
            model_served=None, route_decision=None, route_reason=None,
            route_time_ms=None, generation_time_ms=None,
            input_tokens=est_in_tokens, output_tokens=None,
            status=500, error=str(exc),
        )
        return _error_json(500, "Internal server error",
                           type_="internal_error", code="internal")

    # 6) Stats + response
    await model_manager.stats.incr_model(model_requested)

    # Merge upstream headers + rate-limit headers + request id
    extra_headers = dict(result.headers)
    extra_headers.update(reservation.headers())
    extra_headers["X-Request-ID"] = request_id

    if isinstance(result, PipelineStream):
        # For streams we can't know final token usage before the stream ends;
        # finalize with the upfront reservation.
        # Log a synthetic request entry immediately for observability.
        log_request(
            request_id=request_id,
            api_key_owner=key_info.owner,
            tier=key_info.tier,
            model_requested=model_requested,
            model_served=result.model_served,
            route_decision=result.route_decision,
            route_reason=result.route_reason,
            route_time_ms=result.route_time_ms,
            generation_time_ms=round((time.monotonic() - t0) * 1000, 2),
            input_tokens=est_in_tokens,
            output_tokens=None,
            status=result.status_code,
        )
        return StreamingResponse(
            result.stream,
            status_code=result.status_code,
            headers=extra_headers,
            media_type="text/event-stream",
        )

    # Non-streaming: finalize reservation with actual output tokens (if known)
    if result.output_tokens is not None:
        await reservation.finalize(
            actual_tokens=(result.input_tokens or 0) + (result.output_tokens or 0)
        )

    log_request(
        request_id=request_id,
        api_key_owner=key_info.owner,
        tier=key_info.tier,
        model_requested=model_requested,
        model_served=result.model_served,
        route_decision=result.route_decision,
        route_reason=result.route_reason,
        route_time_ms=result.route_time_ms,
        generation_time_ms=result.generation_time_ms,
        input_tokens=result.input_tokens if result.input_tokens is not None else est_in_tokens,
        output_tokens=result.output_tokens,
        status=result.status_code,
    )

    return JSONResponse(
        status_code=result.status_code,
        content=result.body,
        headers=extra_headers,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _extract_bearer(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    parts = header_value.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return header_value.strip() or None


def _estimate_input_tokens(messages: list) -> int:
    """
    Cheap heuristic: ~4 chars/token. Only used for tier gating and rate
    reservation; actual usage is re-accounted from upstream `usage` when
    available (non-streaming only).
    """
    total_chars = 0
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_chars += len(str(part.get("text", "")))
    return max(1, total_chars // 4)


# ----------------------------------------------------------------------
# Entrypoint (for python main.py; in prod use uvicorn)
# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(
        "main:app",
        host=CONFIG["api_host"],
        port=CONFIG["api_port"],
        workers=1,
        log_config=None,  # use our JSON logger
    )
