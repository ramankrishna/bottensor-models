"""
OpenAI-compatible request/response Pydantic schemas.

Intentionally permissive on unknown fields so we don't break clients that pass
vendor-specific extras (e.g. `tools`, `tool_choice`, `response_format`).
These extras are forwarded verbatim to the upstream vLLM instance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------------------
# Request
# ----------------------------------------------------------------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """
    We accept any extra OpenAI / vLLM-supported field and pass it through
    to the upstream. Only the fields we actually consume at the MoM layer
    are enumerated here.
    """
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    seed: Optional[int] = None


# ----------------------------------------------------------------------
# Response (we re-emit upstream responses mostly untouched, but define the
# shape for /v1/models and error envelopes)
# ----------------------------------------------------------------------
class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "bottensor"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelInfo]


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
class ModelHealth(BaseModel):
    status: str
    vram_mb: Optional[int] = None
    port: int


class HealthResponse(BaseModel):
    status: str
    models: Dict[str, ModelHealth]
    uptime_seconds: int
    requests_total: int
    requests_by_model: Dict[str, int]
    route_stats: Dict[str, int]
    avg_route_time_ms: float
