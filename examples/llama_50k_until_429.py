from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import yaml


MODEL = "llama3.2:1b"
PROVIDER = "local"
OLLAMA_BASE_URL = "http://172.23.240.1:11434/v1"
QUOTA_LIMIT = 50000
QUOTA_ERROR_CODE = "output_quota_exceeded"
CANCELLED_CODE = "request_cancelled"
MISSING_USAGE_CODE = "stream_usage_missing"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "reports"
HEARTBEAT_SECONDS = 30
PROMPT = (
    "Write a long, detailed technical guide for evaluating a streaming LLM proxy under quota pressure. "
    "Cover request tracing, output-token accounting, stream cancellation, circuit breakers, usage polling, "
    "operator reporting, failure handling, and how to interpret concurrent worker behavior. Include many "
    "sections and enough detail to produce thousands of output tokens."
)


@dataclass
class ArtifactPaths:
    config_path: Path
    report_path: Path
    db_path: Path


@dataclass
class StreamAttempt:
    phase: str
    trace_id: str
    span_name: str
    status_code: int | None = None
    terminal_code: str | None = None
    cancelled: bool = False
    quota_warning: str | None = None
    usage_delta: int | None = None
    circuit_open: bool | None = None

    def report_row(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "trace_id": self.trace_id,
            "span_name": self.span_name,
            "status_code": self.status_code,
            "terminal_code": self.terminal_code,
            "cancelled": self.cancelled,
            "quota_warning": self.quota_warning,
            "usage_delta": self.usage_delta,
            "circuit_open": self.circuit_open,
        }


@dataclass
class ProbeState:
    trace_id: str
    started_at: float
    timeout: float
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    failure: str | None = None
    observed_circuit_open: bool = False
    observed_stream_cancel: bool = False
    observed_final_429: bool = False
    completed_streams: int = 0
    active_streams: int = 0
    attempts: int = 0

    def mark_failure(self, message: str) -> None:
        if self.failure is None:
            self.failure = message
        self.stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the proxy and stream Llama requests until the 50k output quota returns 429."
    )
    parser.add_argument("--stream-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--cancel-grace-seconds", type=float, default=10)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    validate_args(args)

    artifacts = create_artifacts(args.report_dir)
    port = read_config_port(artifacts.config_path)
    proxy_url = f"http://127.0.0.1:{port}"
    proxy = await start_proxy(artifacts.config_path, port)

    try:
        await wait_until_healthy(proxy_url)
        timeout = httpx.Timeout(args.timeout, connect=10)
        async with httpx.AsyncClient(base_url=proxy_url, timeout=timeout) as client:
            reset_row = await reset_usage(client, MODEL)
            validate_reset_row(reset_row)
            result = await run_probe(client, args, artifacts.report_path, proxy_url)
            final_usage = await usage_for_model(client, MODEL)
            print_final_report(
                proxy_url=proxy_url,
                artifacts=artifacts,
                trace_id=result.trace_id,
                final_usage=final_usage,
                state=result,
            )
            return 0 if probe_passed(result) else 1
    finally:
        await terminate_proxy(proxy)


def validate_args(args: argparse.Namespace) -> None:
    if args.stream_workers < 1:
        raise ValueError("--stream-workers must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if args.cancel_grace_seconds < 0:
        raise ValueError("--cancel-grace-seconds must be non-negative")


def create_artifacts(report_dir: Path) -> ArtifactPaths:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = unique_stem(report_dir, timestamp)
    paths = ArtifactPaths(
        config_path=report_dir / f"{stem}_llama_50k_proxy.yaml",
        report_path=report_dir / f"{stem}_llama_50k_streams.jsonl",
        db_path=report_dir / f"{stem}_llama_50k.sqlite3",
    )
    write_config(paths.config_path, paths.db_path, free_port())
    paths.report_path.touch()
    return paths


def unique_stem(report_dir: Path, timestamp: str) -> str:
    suffix = 0
    while True:
        stem = f"{timestamp}" if suffix == 0 else f"{timestamp}_{suffix}"
        if not any(report_dir.glob(f"{stem}_llama_50k*")):
            return stem
        suffix += 1


def write_config(config_path: Path, db_path: Path, port: int) -> None:
    config = build_config(db_path, port)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def build_config(db_path: Path, port: int) -> dict[str, Any]:
    return {
        "server": {"host": "127.0.0.1", "port": port},
        "database": {"url": f"sqlite:///{db_path}"},
        "tracing": {"log_full_body": True},
        "quota": {"period": "manual"},
        "providers": {
            "default": PROVIDER,
            PROVIDER: {
                "type": "openai_compatible",
                "default_model": MODEL,
                "base_url": OLLAMA_BASE_URL,
            },
        },
        "models": {MODEL: {"output_token_limit": QUOTA_LIMIT}},
    }


def read_config_port(config_path: Path) -> int:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(data["server"]["port"])


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def terminate_proxy(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def wait_until_healthy(proxy_url: str) -> None:
    async with httpx.AsyncClient(base_url=proxy_url, timeout=1) as client:
        for _ in range(200):
            try:
                response = await client.get("/healthz")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError(f"Proxy did not become healthy: {proxy_url}")


async def reset_usage(client: httpx.AsyncClient, model: str) -> dict[str, Any]:
    response = await client.post(f"/internal/usage/models/{model}/reset")
    response.raise_for_status()
    return response.json()


def validate_reset_row(row: dict[str, Any]) -> None:
    if row.get("model") != MODEL:
        raise RuntimeError(f"Reset returned model {row.get('model')!r}, expected {MODEL!r}")
    if row.get("output_tokens_used") != 0:
        raise RuntimeError(f"Reset did not zero usage: {row}")
    if row.get("circuit_open") is not False:
        raise RuntimeError(f"Reset did not close circuit: {row}")


async def run_probe(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    report_path: Path,
    proxy_url: str,
) -> ProbeState:
    state = ProbeState(trace_id=str(uuid4()), started_at=asyncio.get_running_loop().time(), timeout=args.timeout)
    report_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()

    async def append_report(attempt: StreamAttempt) -> None:
        async with report_lock:
            with report_path.open("a", encoding="utf-8") as report:
                report.write(json.dumps(attempt.report_row(), separators=(",", ":")))
                report.write("\n")

    workers = [
        asyncio.create_task(
            stream_worker(
                client=client,
                worker_index=index,
                max_tokens=args.max_tokens,
                state=state,
                append_report=append_report,
                progress_lock=progress_lock,
            )
        )
        for index in range(1, args.stream_workers + 1)
    ]
    heartbeat = asyncio.create_task(heartbeat_loop(client, state, progress_lock))

    try:
        while not state.stop_event.is_set():
            if asyncio.get_running_loop().time() - state.started_at >= args.timeout:
                state.mark_failure(f"Timed out after {args.timeout:g} seconds before circuit opened")
                break
            await asyncio.sleep(0.2)

        if state.observed_circuit_open:
            await wait_for_worker_grace(workers, args.cancel_grace_seconds)
            final_attempt = await final_429_verification(client, state.trace_id, append_report)
            state.observed_final_429 = is_output_quota_exceeded_attempt(final_attempt)
        else:
            await asyncio.gather(*workers, return_exceptions=True)
    finally:
        state.stop_event.set()
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        await asyncio.gather(*workers, return_exceptions=True)

    return state


async def wait_for_worker_grace(tasks: list[asyncio.Task[Any]], grace_seconds: float) -> None:
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=grace_seconds)
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()


async def stream_worker(
    *,
    client: httpx.AsyncClient,
    worker_index: int,
    max_tokens: int,
    state: ProbeState,
    append_report,
    progress_lock: asyncio.Lock,
) -> None:
    attempt_number = 0
    while not state.stop_event.is_set():
        attempt_number += 1
        state.attempts += 1
        span_name = f"worker_{worker_index}_stream_{attempt_number}"
        state.active_streams += 1
        try:
            attempt = await stream_once(
                client=client,
                trace_id=state.trace_id,
                span_name=span_name,
                phase="probe",
                max_tokens=max_tokens,
            )
            usage = await usage_for_model(client, MODEL, required=False)
            if usage:
                attempt.circuit_open = bool(usage.get("circuit_open"))
            await append_report(attempt)
            state.completed_streams += 1
            update_state_from_attempt(state, attempt)
            await print_progress(progress_lock, state, attempt, usage)
        except Exception as exc:
            state.mark_failure(f"{span_name} failed: {exc}")
            await append_report(
                StreamAttempt(
                    phase="probe",
                    trace_id=state.trace_id,
                    span_name=span_name,
                    terminal_code="exception",
                )
            )
        finally:
            state.active_streams -= 1


async def stream_once(
    *,
    client: httpx.AsyncClient,
    trace_id: str,
    span_name: str,
    phase: str,
    max_tokens: int,
) -> StreamAttempt:
    attempt = StreamAttempt(phase=phase, trace_id=trace_id, span_name=span_name)
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "X-Trace-ID": trace_id,
            "X-Span-Name": span_name,
            "X-LLM-Provider": PROVIDER,
        },
        json={
            "model": MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        },
    ) as response:
        attempt.status_code = response.status_code
        if response.status_code != 200:
            body = await response.aread()
            attempt.terminal_code = terminal_code_from_http_body(body)
            return attempt

        async for line in response.aiter_lines():
            parse_sse_line(attempt, line)
            if attempt.terminal_code in {CANCELLED_CODE, MISSING_USAGE_CODE}:
                continue
        if attempt.terminal_code is None:
            attempt.terminal_code = "done"
    return attempt


def parse_sse_line(attempt: StreamAttempt, line: str) -> None:
    if not line.startswith("data:"):
        return
    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        return
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        attempt.terminal_code = "invalid_sse_json"
        return

    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        attempt.terminal_code = code or "stream_error"
        attempt.cancelled = code == CANCELLED_CODE
        return

    warning = payload.get("warning")
    if isinstance(warning, dict):
        code = warning.get("code")
        attempt.quota_warning = code
        if code == MISSING_USAGE_CODE:
            attempt.terminal_code = MISSING_USAGE_CODE
        return

    usage = payload.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        attempt.usage_delta = int(usage["completion_tokens"])


def terminal_code_from_http_body(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "http_error"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("code") or "http_error")
    return "http_error"


def update_state_from_attempt(state: ProbeState, attempt: StreamAttempt) -> None:
    if attempt.terminal_code == MISSING_USAGE_CODE:
        state.mark_failure("A stream completed with stream_usage_missing")
        return
    if attempt.cancelled:
        state.observed_stream_cancel = True
        return
    if attempt.status_code == 200 and attempt.terminal_code != "done":
        state.mark_failure(f"Unexpected stream terminal_code={attempt.terminal_code}")
        return
    if attempt.status_code not in {200, None}:
        if not is_output_quota_exceeded_attempt(attempt):
            state.mark_failure(f"Unexpected HTTP {attempt.status_code} terminal_code={attempt.terminal_code}")
        return
    if attempt.circuit_open:
        state.observed_circuit_open = True
        state.stop_event.set()


def is_output_quota_exceeded_attempt(attempt: StreamAttempt) -> bool:
    return attempt.status_code == 429 and attempt.terminal_code == QUOTA_ERROR_CODE


async def final_429_verification(client: httpx.AsyncClient, trace_id: str, append_report) -> StreamAttempt:
    attempt = await stream_once(
        client=client,
        trace_id=trace_id,
        span_name="final_429_verification",
        phase="final_429_verification",
        max_tokens=1,
    )
    usage = await usage_for_model(client, MODEL, required=False)
    if usage:
        attempt.circuit_open = bool(usage.get("circuit_open"))
    await append_report(attempt)
    print(
        f"final_429_verification status_code={attempt.status_code} terminal_code={attempt.terminal_code}",
        flush=True,
    )
    return attempt


async def heartbeat_loop(client: httpx.AsyncClient, state: ProbeState, progress_lock: asyncio.Lock) -> None:
    while not state.stop_event.is_set():
        await asyncio.sleep(HEARTBEAT_SECONDS)
        usage = await usage_for_model(client, MODEL, required=False)
        async with progress_lock:
            print(
                "heartbeat "
                f"attempts={state.attempts} active={state.active_streams} "
                f"completed={state.completed_streams} usage={json.dumps(usage, sort_keys=True)}",
                flush=True,
            )


async def print_progress(
    progress_lock: asyncio.Lock,
    state: ProbeState,
    attempt: StreamAttempt,
    usage: dict[str, Any] | None,
) -> None:
    async with progress_lock:
        print(
            f"span={attempt.span_name} status={attempt.status_code} terminal={attempt.terminal_code} "
            f"delta={attempt.usage_delta} cancelled={attempt.cancelled} "
            f"circuit_open={attempt.circuit_open} usage={json.dumps(usage, sort_keys=True)}",
            flush=True,
        )


async def usage_for_model(client: httpx.AsyncClient, model: str, *, required: bool = True) -> dict[str, Any] | None:
    response = await client.get("/internal/usage/models")
    response.raise_for_status()
    for row in response.json():
        if row.get("model") == model:
            return row
    if required:
        raise RuntimeError(f"No usage row found for model {model}")
    return None


def probe_passed(state: ProbeState) -> bool:
    return (
        state.failure is None
        and state.observed_circuit_open
        and state.observed_stream_cancel
        and state.observed_final_429
    )


def print_final_report(
    *,
    proxy_url: str,
    artifacts: ArtifactPaths,
    trace_id: str,
    final_usage: dict[str, Any],
    state: ProbeState,
) -> None:
    print()
    print(f"proxy_url={proxy_url}")
    print(f"config_path={artifacts.config_path}")
    print(f"report_path={artifacts.report_path}")
    print(f"db_path={artifacts.db_path}")
    print(f"trace_url={proxy_url.rstrip('/')}/internal/traces/{trace_id}")
    print(f"final_usage={json.dumps(final_usage, sort_keys=True)}")
    print(f"observed_circuit_open={state.observed_circuit_open}")
    print(f"observed_stream_cancel={state.observed_stream_cancel}")
    print(f"observed_final_429={state.observed_final_429}")
    if state.failure:
        print(f"failure={state.failure}")
    print(f"verdict={'PASS' if probe_passed(state) else 'FAIL'}")


def _handle_sigterm(signum, frame) -> None:  # pragma: no cover - exercised manually
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Interrupted; proxy shutdown requested.", file=sys.stderr)
        raise SystemExit(130)
