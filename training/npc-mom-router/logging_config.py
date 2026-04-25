"""
Structured JSON logging.

Two loggers are configured:
  - "npc-mom"          : general app logs (stdout only)
  - "npc-mom.requests" : one JSON object per request (stdout + rotating file)
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter. Keeps extras flat for easy indexing in Loki/ES."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge explicit "extra" fields passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
                "taskName",
            ):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    request_log_file: Optional[str] = "/var/log/npc-mom/requests.log",
) -> None:
    """
    Idempotent: safe to call multiple times (checks existing handlers).
    """
    level_num = getattr(logging, level.upper(), logging.INFO)

    # -------------------- root / app logger --------------------
    root = logging.getLogger()
    root.setLevel(level_num)
    # Strip default handlers uvicorn may have attached
    for h in list(root.handlers):
        root.removeHandler(h)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(JsonFormatter())
    stdout.setLevel(level_num)
    root.addHandler(stdout)

    # Silence overly chatty libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)

    # -------------------- request logger --------------------
    req = logging.getLogger("npc-mom.requests")
    req.setLevel(logging.INFO)
    req.propagate = False
    for h in list(req.handlers):
        req.removeHandler(h)

    req_stdout = logging.StreamHandler(sys.stdout)
    req_stdout.setFormatter(JsonFormatter())
    req.addHandler(req_stdout)

    if request_log_file:
        try:
            os.makedirs(os.path.dirname(request_log_file), exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                request_log_file,
                maxBytes=50 * 1024 * 1024,  # 50 MB
                backupCount=5,
                encoding="utf-8",
            )
            fh.setFormatter(JsonFormatter())
            req.addHandler(fh)
        except (OSError, PermissionError) as exc:
            # Don't crash startup if /var/log isn't writable — stdout is enough
            logging.getLogger("npc-mom").warning(
                "request-log-file-unavailable",
                extra={"path": request_log_file, "error": str(exc)},
            )


def log_request(
    *,
    request_id: str,
    api_key_owner: Optional[str],
    tier: Optional[str],
    model_requested: str,
    model_served: Optional[str],
    route_decision: Optional[str],
    route_reason: Optional[str],
    route_time_ms: Optional[float],
    generation_time_ms: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    status: int,
    error: Optional[str] = None,
) -> None:
    logging.getLogger("npc-mom.requests").info(
        "request",
        extra={
            "request_id": request_id,
            "api_key_owner": api_key_owner,
            "tier": tier,
            "model_requested": model_requested,
            "model_served": model_served,
            "route_decision": route_decision,
            "route_reason": route_reason,
            "route_time_ms": route_time_ms,
            "generation_time_ms": generation_time_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "status": status,
            "error": error,
        },
    )
