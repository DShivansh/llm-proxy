from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import yaml


MODEL = "llama3.2:1b"
PROVIDER = "local"
QUOTA_LIMIT = 100
QUOTA_ERROR_CODE = "output_quota_exceeded"
CANCELLED_CODE = "request_cancelled"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "reports"


@dataclass
class Artifacts:
    config_path: Path
    report_path: Path
    db_path: Path
    trace_path: Path


@dataclass
class StreamResult:
    phase: str
    trace_id: str
    span_name: str
    status_code: int | None = None
    terminal_code: str | None = None
    cancelled: bool = False
    usage_delta: int | None = None
    chunks: int = 0

    def row(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "trace_id": self.trace_id,
            "span_name": self.span_name,
            "status_code": self.status_code,
            "terminal_code": self.terminal_code,
            "cancelled": self.cancelled,
            "usage_delta": self.usage_delta,
            "chunks": self.chunks,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic fake-upstream proof of mid-stream quota cancellation."
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")

    artifacts = create_artifacts(args.report_dir)
    upstream_port = free_port()
    proxy_port = free_port()
    write_proxy_config(artifacts.config_path, artifacts.db_path, proxy_port, upstream_port)

    upstream = await start_uvicorn("examples.fake_streaming_server:app", upstream_port)
    proxy = await start_proxy(artifacts.config_path, proxy_port)
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    upstream_url = f"http://127.0.0.1:{upstream_port}"

    try:
        await wait_until_healthy(upstream_url)
        await wait_until_healthy(proxy_url)
        result = await asyncio.wait_for(run_probe(proxy_url, artifacts.report_path), timeout=args.timeout)
        print_final(proxy_url, upstream_url, artifacts, result)
        return 0 if result["passed"] else 1
    finally:
        await terminate_process(proxy)
        await terminate_process(upstream)


def create_artifacts(report_dir: Path) -> Artifacts:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = unique_stem(report_dir, timestamp)
    artifacts = Artifacts(
        config_path=report_dir / f"{stem}_fake_cancel_proxy.yaml",
        report_path=report_dir / f"{stem}_fake_cancel_streams.jsonl",
        db_path=report_dir / f"{stem}_fake_cancel.sqlite3",
        trace_path=report_dir / f"{stem}_fake_cancel_trace.json",
    )
    artifacts.report_path.touch()
    return artifacts


def unique_stem(report_dir: Path, timestamp: str) -> str:
    suffix = 0
    while True:
        stem = timestamp if suffix == 0 else f"{timestamp}_{suffix}"
        if not any(report_dir.glob(f"{stem}_fake_cancel*")):
            return stem
        suffix += 1


def write_proxy_config(config_path: Path, db_path: Path, proxy_port: int, upstream_port: int) -> None:
    config = {
        "server": {"host": "127.0.0.1", "port": proxy_port},
        "database": {"url": f"sqlite:///{db_path}"},
        "tracing": {"log_full_body": True},
        "quota": {"period": "manual"},
        "providers": {
            "default": PROVIDER,
            PROVIDER: {
                "type": "openai_compatible",
                "default_model": MODEL,
                "base_url": f"http://127.0.0.1:{upstream_port}/v1",
            },
        },
        "models": {MODEL: {"output_token_limit": QUOTA_LIMIT}},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def start_uvicorn(app_path: str, port: int) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "uvicorn",
        app_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def start_proxy(config_path: Path, port: int) -> asyncio.subprocess.Process:
    env = {**os.environ, "LLM_PROXY_CONFIG": str(config_path)}
    return await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


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


async def run_probe(proxy_url: str, report_path: Path) -> dict[str, Any]:
    trace_id = str(uuid4())
    sentinel_started = asyncio.Event()
    async with httpx.AsyncClient(base_url=proxy_url, timeout=None) as client:
        reset_row = await reset_usage(client)
        sentinel_task = asyncio.create_task(read_sentinel(client, trace_id, sentinel_started))
        await asyncio.wait_for(sentinel_started.wait(), timeout=5)

        quota_result = await read_stream(
            client=client,
            trace_id=trace_id,
            span_name="quota_killer",
            phase="quota_killer",
            prompt="quota-killer",
        )
        sentinel_result = await asyncio.wait_for(sentinel_task, timeout=5)
        final_result = await final_429(client, trace_id)
        usage = await usage_for_model(client)
        trace = await trace_for_id(client, trace_id)

    write_report(report_path, sentinel_result)
    write_report(report_path, quota_result)
    write_report(report_path, final_result)

    passed = (
        reset_row.get("output_tokens_used") == 0
        and quota_result.status_code == 200
        and quota_result.usage_delta == 120
        and sentinel_result.status_code == 200
        and sentinel_result.cancelled
        and final_result.status_code == 429
        and final_result.terminal_code == QUOTA_ERROR_CODE
        and usage.get("circuit_open") is True
    )
    return {
        "trace_id": trace_id,
        "reset": reset_row,
        "usage": usage,
        "sentinel": sentinel_result.row(),
        "quota_killer": quota_result.row(),
        "final_429": final_result.row(),
        "passed": passed,
        "trace": trace,
    }


async def reset_usage(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.post(f"/internal/usage/models/{MODEL}/reset")
    response.raise_for_status()
    return response.json()


async def usage_for_model(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/internal/usage/models")
    response.raise_for_status()
    for row in response.json():
        if row.get("model") == MODEL:
            return row
    raise RuntimeError(f"No usage row found for model {MODEL}")


async def trace_for_id(client: httpx.AsyncClient, trace_id: str) -> dict[str, Any]:
    response = await client.get(f"/internal/traces/{trace_id}")
    response.raise_for_status()
    return response.json()


async def read_sentinel(client: httpx.AsyncClient, trace_id: str, started: asyncio.Event) -> StreamResult:
    return await read_stream(
        client=client,
        trace_id=trace_id,
        span_name="sentinel_slow_stream",
        phase="sentinel",
        prompt="sentinel slow stream",
        first_chunk_event=started,
    )


async def final_429(client: httpx.AsyncClient, trace_id: str) -> StreamResult:
    return await read_stream(
        client=client,
        trace_id=trace_id,
        span_name="final_429_verification",
        phase="final_429_verification",
        prompt="normal after circuit open",
    )


async def read_stream(
    *,
    client: httpx.AsyncClient,
    trace_id: str,
    span_name: str,
    phase: str,
    prompt: str,
    first_chunk_event: asyncio.Event | None = None,
) -> StreamResult:
    result = StreamResult(phase=phase, trace_id=trace_id, span_name=span_name)
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Trace-ID": trace_id, "X-Span-Name": span_name, "X-LLM-Provider": PROVIDER},
        json={
            "model": MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
    ) as response:
        result.status_code = response.status_code
        if response.status_code != 200:
            result.terminal_code = terminal_code_from_body(await response.aread())
            return result

        async for line in response.aiter_lines():
            parse_sse_line(result, line, first_chunk_event)
            if result.cancelled:
                continue
        if result.terminal_code is None:
            result.terminal_code = "done"
        return result


def parse_sse_line(result: StreamResult, line: str, first_chunk_event: asyncio.Event | None = None) -> None:
    if not line.startswith("data:"):
        return
    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        return
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        result.terminal_code = "invalid_sse_json"
        return

    error = payload.get("error")
    if isinstance(error, dict):
        result.terminal_code = str(error.get("code") or "stream_error")
        result.cancelled = result.terminal_code == CANCELLED_CODE
        return

    usage = payload.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        result.usage_delta = int(usage["completion_tokens"])
        return

    for choice in payload.get("choices") or []:
        content = (choice.get("delta") or {}).get("content")
        if content:
            result.chunks += 1
            if first_chunk_event is not None:
                first_chunk_event.set()


def terminal_code_from_body(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "http_error"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("code") or "http_error")
    return "http_error"


def write_report(report_path: Path, result: StreamResult) -> None:
    with report_path.open("a", encoding="utf-8") as report:
        report.write(json.dumps(result.row(), separators=(",", ":")))
        report.write("\n")


def print_final(proxy_url: str, upstream_url: str, artifacts: Artifacts, result: dict[str, Any]) -> None:
    artifacts.trace_path.write_text(json.dumps(result["trace"], indent=2, sort_keys=True), encoding="utf-8")
    print(f"fake_upstream_url={upstream_url}")
    print(f"proxy_url={proxy_url}")
    print(f"config_path={artifacts.config_path}")
    print(f"report_path={artifacts.report_path}")
    print(f"db_path={artifacts.db_path}")
    print(f"trace_path={artifacts.trace_path}")
    print(f"trace_url={proxy_url.rstrip('/')}/internal/traces/{result['trace_id']}")
    print(f"sentinel={json.dumps(result['sentinel'], sort_keys=True)}")
    print(f"quota_killer={json.dumps(result['quota_killer'], sort_keys=True)}")
    print(f"final_429={json.dumps(result['final_429'], sort_keys=True)}")
    print(f"final_usage={json.dumps(result['usage'], sort_keys=True)}")
    print(f"verdict={'PASS' if result['passed'] else 'FAIL'}")


def _handle_sigterm(signum, frame) -> None:  # pragma: no cover - exercised manually
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Interrupted; servers shutdown requested.", file=sys.stderr)
        raise SystemExit(130)
