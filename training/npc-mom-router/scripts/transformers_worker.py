#!/usr/bin/env python3
"""
Transformers + bitsandbytes fallback worker for NPC Fin 32B.

Exposes an OpenAI-compatible `/v1/chat/completions` endpoint (plus `/health`
and `/v1/models`) on a chosen port, so the MoM router in pipeline.py can talk
to it exactly as if it were a vLLM instance.

Used only when vLLM + bitsandbytes refuses to load the model. Slower than
vLLM but always works.

Key choices:
  * nf4 quantization, double_quant, compute dtype bfloat16 (A40-friendly)
  * `device_map="auto"` so accelerate places layers on GPU 0
  * Streaming implemented via `TextIteratorStreamer` in a background thread
  * Chat templating via the tokenizer's built-in `apply_chat_template`
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer


log = logging.getLogger("npc-fin-fallback")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ----------------------------------------------------------------------
# Request schema (mirror of main schemas.py but local to avoid sys.path deps)
# ----------------------------------------------------------------------
class _Msg(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any | None = None


class _Req(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: List[_Msg]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[Any] = None
    seed: Optional[int] = None


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------
class FinWorker:
    def __init__(self, model_name: str, max_model_len: int, served_name: str):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.served_name = served_name
        self.tokenizer = None
        self.model = None
        self._ready = False
        self._load_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            if self._ready:
                return
            log.info("loading tokenizer: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            log.info("loading model (4-bit NF4): %s", self.model_name)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self.model.eval()
            self._ready = True
            log.info("model ready")

    # ------------------------------------------------------------------
    def _render_prompt(self, messages: List[Dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _gen_args(self, req: _Req) -> Dict[str, Any]:
        temperature = max(1e-5, float(req.temperature))
        do_sample = temperature > 0.0 and (req.top_p or 1.0) < 1.0 or temperature != 1.0
        # Be generous: if either temp!=0 or top_p<1, sample; else greedy.
        if temperature <= 1e-5 and (req.top_p or 1.0) >= 1.0:
            do_sample = False
        return {
            "max_new_tokens": int(req.max_tokens or 512),
            "temperature": temperature,
            "top_p": float(req.top_p or 1.0),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    # ------------------------------------------------------------------
    def complete(self, req: _Req) -> Dict[str, Any]:
        """Non-streaming completion. Returns OpenAI-shaped response."""
        prompt = self._render_prompt([m.model_dump() for m in req.messages])
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len,
        ).to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **self._gen_args(req))

        in_len = inputs["input_ids"].shape[1]
        gen_tokens = out[0][in_len:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(in_len),
                "completion_tokens": int(gen_tokens.shape[0]),
                "total_tokens": int(in_len + gen_tokens.shape[0]),
            },
        }

    # ------------------------------------------------------------------
    async def stream(self, req: _Req) -> AsyncGenerator[bytes, None]:
        """Streaming completion via TextIteratorStreamer in a thread."""
        prompt = self._render_prompt([m.model_dump() for m in req.messages])
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {**self._gen_args(req), **inputs, "streamer": streamer}
        # Run generation in a separate thread so we can iterate tokens
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Role-only first chunk (OpenAI convention)
        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.served_name,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first)}\n\n".encode("utf-8")

        loop = asyncio.get_running_loop()
        # Pump the iterator off-loop
        while True:
            text_chunk = await loop.run_in_executor(None, _streamer_next, streamer)
            if text_chunk is _SENTINEL:
                break
            if not text_chunk:
                continue
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.served_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.served_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


_SENTINEL = object()


def _streamer_next(streamer: TextIteratorStreamer):
    """TextIteratorStreamer is itself an iterator — call next() on it directly."""
    try:
        return next(streamer)
    except StopIteration:
        return _SENTINEL


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
def make_app(worker: FinWorker) -> FastAPI:
    app = FastAPI(title="NPC Fin 32B — transformers fallback worker")

    @app.get("/health")
    async def health():
        return {"status": "ok" if worker._ready else "loading"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": worker.served_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "bottensor",
                },
                {
                    "id": worker.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "bottensor",
                },
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status_code=400,
            )

        try:
            req = _Req.model_validate(body)
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status_code=400,
            )

        if not worker._ready:
            # Load lazily on first request (also triggered by --preload)
            await asyncio.to_thread(worker.load)

        if req.stream:
            return StreamingResponse(
                worker.stream(req),
                media_type="text/event-stream",
            )

        result = await asyncio.to_thread(worker.complete, req)
        return JSONResponse(result)

    return app


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--served-model-name", default="npc-fin-32b")
    ap.add_argument("--preload", action="store_true",
                    help="Load model at startup instead of lazily")
    args = ap.parse_args()

    worker = FinWorker(
        model_name=args.model,
        max_model_len=args.max_model_len,
        served_name=args.served_model_name,
    )

    if args.preload:
        worker.load()

    app = make_app(worker)

    # Bind and go
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
