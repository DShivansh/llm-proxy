from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


MODEL = "llama3.2:1b"

app = FastAPI()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    prompt = first_prompt_text(body)
    model = body.get("model") or MODEL

    if not body.get("stream"):
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "fake response"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    if "quota-killer" in prompt:
        return StreamingResponse(quota_killer_events(model), media_type="text/event-stream")
    if "sentinel" in prompt:
        return StreamingResponse(sentinel_events(model), media_type="text/event-stream")
    return StreamingResponse(normal_events(model), media_type="text/event-stream")


def first_prompt_text(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    if not messages:
        return ""
    return str(messages[0].get("content") or "")


async def sentinel_events(model: str) -> AsyncIterator[bytes]:
    for index in range(1, 1000):
        yield sse(
            {
                "id": "chatcmpl-fake-sentinel",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": f"sentinel-{index} "}}],
            }
        )
        await asyncio.sleep(0.05)

    yield sse(
        {
            "id": "chatcmpl-fake-sentinel",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    yield b"data: [DONE]\n\n"


async def quota_killer_events(model: str) -> AsyncIterator[bytes]:
    yield sse(
        {
            "id": "chatcmpl-fake-killer",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "quota-killer-complete"}}],
        }
    )
    await asyncio.sleep(0.05)
    yield sse(
        {
            "id": "chatcmpl-fake-killer",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 120, "total_tokens": 121},
        }
    )
    yield b"data: [DONE]\n\n"


async def normal_events(model: str) -> AsyncIterator[bytes]:
    yield sse(
        {
            "id": "chatcmpl-fake-normal",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "normal"}}],
        }
    )
    yield sse(
        {
            "id": "chatcmpl-fake-normal",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    yield b"data: [DONE]\n\n"


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
