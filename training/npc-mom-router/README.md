# NPC MoM Router

> **Status: retired.** Replaced by direct vLLM serving with per-model
> tool-call parsing. Code preserved here for reference.

A FastAPI gateway that fronted **NPC Fast** (router) and **NPC Fin**
(heavy) on the same A40 pod. Implemented OpenAI-compatible
`/v1/chat/completions`, tenant API-key auth via MongoDB, request
rate-limiting, and Hermes-style tool-call parsing.

## Architecture

```
client ──► /v1/chat/completions ──► auth (Mongo) ──► rate limiter ──┐
                                                                     │
                       ┌─────────────────────────────────────────────┘
                       ▼
              ┌──── pipeline.py ────┐
              │  1. NPC Fast (vLLM) │  decides: self-answer or escalate
              │  2. NPC Fin (vLLM)  │  invoked only when escalated
              └─────────────────────┘
                       │
                       ▼
                  router.py — assembles streaming response
```

## Files

| File              | Role                                                      |
|-------------------|-----------------------------------------------------------|
| `main.py`         | FastAPI app, CORS, middleware, lifespan                   |
| `auth.py`         | Bearer-token validation, Mongo lookup, dev-key bypass     |
| `config.py`       | All env-driven config; loads `os.getenv(...)` only        |
| `models.py`       | Pydantic models for request/response/usage                |
| `schemas.py`      | OpenAI-compatible chat completion schemas                 |
| `pipeline.py`     | Two-stage routing: Fast → (optional) Fin                  |
| `rate_limiter.py` | Per-API-key sliding-window limiter                        |
| `router.py`       | Streams SSE chunks, manages tool-call markers             |
| `logging_config.py` | Structured JSON logging                                 |
| `Dockerfile`      | Slim base, `uvicorn` entrypoint                           |
| `docker-compose.yml`     | Full stack: router + 2× vLLM workers           |
| `docker-compose.fallback.yml` | Single-vLLM fallback compose             |
| `scripts/start.sh`, `stop.sh`, `health_check.sh` | Ops helpers       |
| `scripts/transformers_worker.py` | CPU fallback worker for dev mode      |

## Setup (dev)

```bash
cp ../.env.example .env              # set MONGO_URI, HF_TOKEN, ALLOW_DEV_KEY
docker compose up                    # router + 2× vLLM
# or for CPU dev:
docker compose -f docker-compose.fallback.yml up
```

## Why retired

- vLLM 0.18+ added native Hermes tool-parsing, removing the need for a
  custom router-side parser.
- The two-stage Fast→Fin handoff added latency without quality wins once
  the generalist NPC Agentic 7B replaced the split.

## Lessons baked into successors

- Per-tenant API keys → kept (now lives at vLLM ingress)
- Rate limiter sliding-window logic → reused
- Tool-call streaming markers → upstreamed into vLLM config
