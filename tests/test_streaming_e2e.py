from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.app import create_app
from llm_proxy.config import load_config


MODEL = "llama3.2:3b"


class ServerThread:
    def __init__(self, app: FastAPI, port: int) -> None:
        self.port = port
        self.config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="on",
        )
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ServerThread":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _run(self) -> None:
        asyncio.run(self.server.serve())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fake_upstream_app() -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        prompt = str(body["messages"][0]["content"])
        if "upstream-429" in prompt:
            return JSONResponse(
                {"error": {"message": "upstream quota", "type": "rate_limit", "code": "rate_limit"}},
                status_code=429,
            )

        async def events() -> AsyncIterator[bytes]:
            slow = "slow-stream" in prompt
            missing_usage = "missing-usage" in prompt
            chunks = ["alpha ", "beta ", "gamma "] if not slow else [f"slow-{index} " for index in range(20)]
            for chunk in chunks:
                yield sse(
                    {
                        "id": "chatcmpl-upstream",
                        "object": "chat.completion.chunk",
                        "model": body.get("model", MODEL),
                        "choices": [{"index": 0, "delta": {"content": chunk}}],
                    }
                )
                if slow:
                    await asyncio.sleep(0.05)

            if not missing_usage:
                completion_tokens = 30 if "small-success" in prompt else 120
                yield sse(
                    {
                        "id": "chatcmpl-upstream",
                        "object": "chat.completion.chunk",
                        "model": body.get("model", MODEL),
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": completion_tokens,
                            "total_tokens": completion_tokens + 10,
                        },
                    }
                )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def sse(payload: dict[str, Any]) -> bytes:
    import json

    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def write_proxy_config(tmp_path: Path, upstream_url: str, quota: int = 500) -> Path:
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "llm_proxy.db"
    config_path.write_text(
        f"""
server:
  host: 127.0.0.1
  port: 8000
database:
  url: sqlite:///{db_path}
tracing:
  log_full_body: true
quota:
  period: manual
providers:
  default: local
  local:
    type: openai_compatible
    default_model: {MODEL}
    base_url: {upstream_url}/v1
models:
  {MODEL}:
    output_token_limit: {quota}
""",
        encoding="utf-8",
    )
    return config_path


def streaming_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }


async def read_stream(client: httpx.AsyncClient, prompt: str, trace_id: str, span_name: str) -> str:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Trace-ID": trace_id, "X-Span-Name": span_name, "X-LLM-Provider": "local"},
        json=streaming_payload(prompt),
    ) as response:
        assert response.status_code == 200
        parts = []
        async for text in response.aiter_text():
            parts.append(text)
        return "".join(parts)


def test_streaming_success_records_usage_and_trace(tmp_path: Path):
    asyncio.run(_streaming_success_records_usage_and_trace(tmp_path))


async def _streaming_success_records_usage_and_trace(tmp_path: Path) -> None:
    with ServerThread(fake_upstream_app(), free_port()) as upstream:
        proxy = create_app(load_config(write_proxy_config(tmp_path, upstream.url)))
        with ServerThread(proxy, free_port()) as proxy_server:
            await wait_until_healthy(proxy_server.url)
            async with httpx.AsyncClient(base_url=proxy_server.url, timeout=10) as client:
                text = await read_stream(client, "small-success", "stream-success", "single_stream")

                assert "alpha " in text
                assert "data: [DONE]" in text
                usage = await usage_for_model(client)
                assert usage["output_tokens_used"] == 30
                assert usage["circuit_open"] is False

                trace = (await client.get("/internal/traces/stream-success")).json()
                span = trace["spans"][0]
                assert span["status"] == "success"
                assert span["usage_source"] == "provider"
                assert span["output_tokens"] == 30
                assert span["response_body"]["choices"][0]["message"]["content"] == "alpha beta gamma "


def test_open_circuit_blocks_stream_before_upstream(tmp_path: Path):
    asyncio.run(_open_circuit_blocks_stream_before_upstream(tmp_path))


async def _open_circuit_blocks_stream_before_upstream(tmp_path: Path) -> None:
    with ServerThread(fake_upstream_app(), free_port()) as upstream:
        proxy = create_app(load_config(write_proxy_config(tmp_path, upstream.url, quota=20)))
        with ServerThread(proxy, free_port()) as proxy_server:
            await wait_until_healthy(proxy_server.url)
            async with httpx.AsyncClient(base_url=proxy_server.url, timeout=10) as client:
                await read_stream(client, "small-success", "first", "opens_circuit")
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Trace-ID": "blocked", "X-Span-Name": "blocked_stream", "X-LLM-Provider": "local"},
                    json=streaming_payload("small-success"),
                )

                assert response.status_code == 429
                assert response.json()["error"]["code"] == "output_quota_exceeded"
                trace = (await client.get("/internal/traces/blocked")).json()
                assert trace["spans"][0]["status"] == "blocked"


def test_parallel_stream_is_cancelled_when_another_stream_opens_quota(tmp_path: Path):
    asyncio.run(_parallel_stream_is_cancelled_when_another_stream_opens_quota(tmp_path))


async def _parallel_stream_is_cancelled_when_another_stream_opens_quota(tmp_path: Path) -> None:
    with ServerThread(fake_upstream_app(), free_port()) as upstream:
        proxy = create_app(load_config(write_proxy_config(tmp_path, upstream.url, quota=100)))
        with ServerThread(proxy, free_port()) as proxy_server:
            await wait_until_healthy(proxy_server.url)
            async with httpx.AsyncClient(base_url=proxy_server.url, timeout=10) as client:
                slow_started = asyncio.Event()

                async def slow_reader() -> str:
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        headers={"X-Trace-ID": "slow-trace", "X-Span-Name": "slow_stream", "X-LLM-Provider": "local"},
                        json=streaming_payload("slow-stream"),
                    ) as response:
                        assert response.status_code == 200
                        parts = []
                        async for text in response.aiter_text():
                            parts.append(text)
                            slow_started.set()
                        return "".join(parts)

                slow_task = asyncio.create_task(slow_reader())
                await asyncio.wait_for(slow_started.wait(), timeout=5)

                fast_text = await read_stream(client, "fast-opens-quota", "fast-trace", "fast_stream")
                slow_text = await slow_task

                assert "data: [DONE]" in fast_text
                assert "request_cancelled" in slow_text
                assert "data: [DONE]" in slow_text

                usage = await usage_for_model(client)
                assert usage["output_tokens_used"] == 120
                assert usage["circuit_open"] is True

                fast_trace = (await client.get("/internal/traces/fast-trace")).json()
                slow_trace = (await client.get("/internal/traces/slow-trace")).json()
                assert fast_trace["spans"][0]["status"] == "success"
                assert fast_trace["spans"][0]["circuit_opened_by_this_span"] is True
                assert slow_trace["spans"][0]["status"] == "cancelled"
                assert slow_trace["spans"][0]["output_tokens"] == 0


def test_upstream_429_before_streaming_returns_http_429(tmp_path: Path):
    asyncio.run(_upstream_429_before_streaming_returns_http_429(tmp_path))


async def _upstream_429_before_streaming_returns_http_429(tmp_path: Path) -> None:
    with ServerThread(fake_upstream_app(), free_port()) as upstream:
        proxy = create_app(load_config(write_proxy_config(tmp_path, upstream.url)))
        with ServerThread(proxy, free_port()) as proxy_server:
            await wait_until_healthy(proxy_server.url)
            async with httpx.AsyncClient(base_url=proxy_server.url, timeout=10) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Trace-ID": "upstream-429", "X-Span-Name": "rate_limited", "X-LLM-Provider": "local"},
                    json=streaming_payload("upstream-429"),
                )

                assert response.status_code == 429
                assert response.json()["error"]["code"] == "rate_limit"
                usage_rows = (await client.get("/internal/usage/models")).json()
                assert usage_rows == []
                trace = (await client.get("/internal/traces/upstream-429")).json()
                assert trace["spans"][0]["status"] == "error"
                assert trace["spans"][0]["http_status_code"] == 429


def test_missing_stream_usage_warns_without_accounting(tmp_path: Path):
    asyncio.run(_missing_stream_usage_warns_without_accounting(tmp_path))


async def _missing_stream_usage_warns_without_accounting(tmp_path: Path) -> None:
    with ServerThread(fake_upstream_app(), free_port()) as upstream:
        proxy = create_app(load_config(write_proxy_config(tmp_path, upstream.url)))
        with ServerThread(proxy, free_port()) as proxy_server:
            await wait_until_healthy(proxy_server.url)
            async with httpx.AsyncClient(base_url=proxy_server.url, timeout=10) as client:
                text = await read_stream(client, "missing-usage", "missing-usage", "missing_usage_stream")

                assert "stream_usage_missing" in text
                assert "data: [DONE]" in text
                usage_rows = (await client.get("/internal/usage/models")).json()
                assert usage_rows == []

                trace = (await client.get("/internal/traces/missing-usage")).json()
                span = trace["spans"][0]
                assert span["status"] == "success"
                assert span["usage_source"] == "missing"
                assert span["output_tokens"] == 0
                assert span["response_body"]["warning"]["code"] == "stream_usage_missing"


async def wait_until_healthy(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=1) as client:
        for _ in range(100):
            try:
                response = await client.get("/healthz")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)
    raise RuntimeError(f"Server did not become healthy: {base_url}")


async def usage_for_model(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/internal/usage/models")
    response.raise_for_status()
    for row in response.json():
        if row["model"] == MODEL:
            return row
    raise AssertionError(f"No usage row for {MODEL}")
