"""
vLLM model manager.

Responsibilities:
  - Own the shared httpx.AsyncClient used to talk to the two vLLM instances.
  - Track per-model stats (requests count, route decisions, route latency).
  - Answer health questions for /health.
  - Forward `/v1/chat/completions` requests to either the NPC Fast (port 8001)
    or NPC Fin (port 8002) upstream.

The two model servers themselves are launched as separate processes (either
`vllm/vllm-openai` Docker images in compose, or a transformers fallback worker
for NPC Fin — see scripts/transformers_worker.py).

This module does not load any model weights into its own process.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import httpx

from config import CONFIG

log = logging.getLogger("npc-mom")


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------
@dataclass
class _RouteStats:
    self_: int = 0
    npc_fin: int = 0


@dataclass
class Stats:
    started_at: float = field(default_factory=time.time)
    requests_total: int = 0
    requests_by_model: Dict[str, int] = field(default_factory=dict)
    route_stats: _RouteStats = field(default_factory=_RouteStats)
    _route_time_sum_ms: float = 0.0
    _route_time_n: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def incr_model(self, model: str) -> None:
        async with self._lock:
            self.requests_total += 1
            self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1

    async def incr_route(self, decision: str, route_time_ms: float) -> None:
        async with self._lock:
            if decision == "self":
                self.route_stats.self_ += 1
            elif decision == "npc_fin":
                self.route_stats.npc_fin += 1
            self._route_time_sum_ms += route_time_ms
            self._route_time_n += 1

    def avg_route_time_ms(self) -> float:
        if self._route_time_n == 0:
            return 0.0
        return round(self._route_time_sum_ms / self._route_time_n, 2)


# ----------------------------------------------------------------------
# Model Manager
# ----------------------------------------------------------------------
class ModelManager:
    """
    Thin HTTP gateway to the two vLLM upstreams. Reuses a single
    httpx.AsyncClient across the whole app for connection pooling.
    """

    NPC_FAST = "npc-fast"
    NPC_FIN = "npc-fin-32b"

    def __init__(self):
        self.stats = Stats()
        self._client: Optional[httpx.AsyncClient] = None

        # Published vLLM model IDs (what each upstream reports as `model`).
        # We need this because `model` sent to the upstream must match what
        # that upstream has loaded; the public name is rewritten accordingly.
        self._upstream_model_id: Dict[str, str] = {
            self.NPC_FAST: CONFIG["npc_fast_model"],
            self.NPC_FIN:  CONFIG["npc_fin_model"],
        }
        self._ports: Dict[str, int] = {self.NPC_FAST: 8001, self.NPC_FIN: 8002}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        # Generous pool: we run streaming requests that can stay open for a while.
        limits = httpx.Limits(
            max_connections=200,
            max_keepalive_connections=50,
            keepalive_expiry=30.0,
        )
        # Per-call timeout overridden via `timeout=` kw at call sites
        self._client = httpx.AsyncClient(limits=limits, timeout=None)
        log.info("model-manager-started")

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("model-manager-stopped")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def upstream_url(self, name: str) -> str:
        if name == self.NPC_FAST:
            return CONFIG["npc_fast_url"]
        if name == self.NPC_FIN:
            return CONFIG["npc_fin_url"]
        raise ValueError(f"unknown upstream: {name}")

    def upstream_auth_headers(self, name: str) -> Dict[str, str]:
        """Bearer headers for authenticated upstream vLLM (--api-key)."""
        if name == self.NPC_FAST:
            key = CONFIG.get("npc_fast_upstream_key", "")
        elif name == self.NPC_FIN:
            key = CONFIG.get("npc_fin_upstream_key", "")
        else:
            return {}
        return {"Authorization": f"Bearer {key}"} if key else {}

    def upstream_model_id(self, name: str) -> str:
        return self._upstream_model_id[name]

    def port(self, name: str) -> int:
        return self._ports[name]

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ModelManager not started")
        return self._client

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def health(self, name: str) -> Tuple[str, Optional[int]]:
        """Returns (status, vram_mb). vram_mb is best-effort (None if unknown)."""
        url = f"{self.upstream_url(name)}/health"
        try:
            r = await self.client.get(url, timeout=CONFIG["healthcheck_timeout_s"])
            if r.status_code == 200:
                return "loaded", None
            return f"unhealthy:{r.status_code}", None
        except httpx.RequestError as exc:
            return f"unreachable:{type(exc).__name__}", None

    async def list_upstream_models(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetches /v1/models from an upstream (not cached — used by admin tools only)."""
        url = f"{self.upstream_url(name)}/v1/models"
        try:
            r = await self.client.get(url, timeout=CONFIG["healthcheck_timeout_s"])
            if r.status_code == 200:
                return r.json()
        except httpx.RequestError:
            return None
        return None

    # ------------------------------------------------------------------
    # Forwarding (non-streaming)
    # ------------------------------------------------------------------
    async def chat_completion(
        self,
        upstream_name: str,
        payload: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> httpx.Response:
        """
        POST to `<upstream>/v1/chat/completions`. Caller handles `.json()` and
        status-code translation.

        `payload["model"]` is rewritten to the upstream's actual model id.
        """
        payload = dict(payload)
        payload["model"] = self.upstream_model_id(upstream_name)
        payload["stream"] = False  # force non-stream at this entrypoint

        url = f"{self.upstream_url(upstream_name)}/v1/chat/completions"
        timeout = timeout_s or CONFIG["generation_timeout_s"]
        headers = self.upstream_auth_headers(upstream_name)

        r = await self.client.post(url, json=payload, timeout=timeout, headers=headers)
        return r

    # ------------------------------------------------------------------
    # Forwarding (streaming SSE)
    # ------------------------------------------------------------------
    async def chat_completion_stream(
        self,
        upstream_name: str,
        payload: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Async generator yielding raw SSE bytes from upstream. Caller is
        responsible for relaying to the client.

        We pass through everything verbatim — including the terminal
        `data: [DONE]\\n\\n`.
        """
        payload = dict(payload)
        payload["model"] = self.upstream_model_id(upstream_name)
        payload["stream"] = True

        url = f"{self.upstream_url(upstream_name)}/v1/chat/completions"
        timeout = timeout_s or CONFIG["generation_timeout_s"]
        headers = self.upstream_auth_headers(upstream_name)

        async with self.client.stream(
            "POST",
            url,
            json=payload,
            timeout=httpx.Timeout(timeout, read=timeout),
            headers=headers,
        ) as resp:
            if resp.status_code >= 400:
                # Surface upstream error as a single SSE frame
                body = (await resp.aread()).decode("utf-8", errors="replace")
                import json as _json
                err_obj = {
                    "error": {
                        "message": body[:2000],
                        "type": "upstream_error",
                        "code": str(resp.status_code),
                    }
                }
                yield f"data: {_json.dumps(err_obj)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

            async for chunk in resp.aiter_raw():
                if chunk:
                    yield chunk


# Module-level singleton
model_manager = ModelManager()
