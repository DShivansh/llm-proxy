from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx


REPORT_DIR = Path(__file__).resolve().parent / "reports"
QUOTA_ERROR_CODE = "output_quota_exceeded"
PROMPT = (
    "You are helping evaluate an LLM sidecar proxy under concurrent agentic load. "
    "Explain how a proxy should coordinate tracing, quota enforcement, cancellation, "
    "and usage reporting during a burst test. Respond with a moderately detailed "
    "6-8 bullet point response."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel agentic workflows against a running LLM proxy until quota opens."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--workflows", type=int, default=5)
    parser.add_argument("--calls-per-workflow", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    validate_args(args)

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        await require_healthy(client)
        reset_row = await reset_usage(client, args.model)
        validate_reset_row(reset_row, args.model)

        report_path = next_report_path()
        report_path.parent.mkdir(parents=True, exist_ok=True)

        stop_event = asyncio.Event()
        payload = build_payload(args.model)

        with report_path.open("w", encoding="utf-8") as report:
            tasks = [
                asyncio.create_task(
                    run_workflow(
                        client=client,
                        provider=args.provider,
                        payload=payload,
                        workflow_number=workflow_number,
                        calls_per_workflow=args.calls_per_workflow,
                        stop_event=stop_event,
                        report=report,
                    )
                )
                for workflow_number in range(1, args.workflows + 1)
            ]
            await asyncio.gather(*tasks)

        final_usage = await usage_for_model(client, args.model)
        print()
        print(f"report_file={report_path}")
        print(f"final_usage={json.dumps(final_usage, sort_keys=True)}")


def validate_args(args: argparse.Namespace) -> None:
    if args.workflows < 1:
        raise ValueError("--workflows must be at least 1")
    if args.calls_per_workflow < 1:
        raise ValueError("--calls-per-workflow must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if "/" in args.model:
        raise ValueError("--model must not contain '/' because it is used in the reset endpoint path")


async def require_healthy(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    if response.status_code != 200:
        raise RuntimeError(f"Health check failed: status_code={response.status_code} body={response.text}")


async def reset_usage(client: httpx.AsyncClient, model: str) -> dict[str, Any]:
    response = await client.post(f"/internal/usage/models/{model}/reset")
    response.raise_for_status()
    return response.json()


def validate_reset_row(row: dict[str, Any], model: str) -> None:
    if row.get("model") != model:
        raise RuntimeError(f"Reset returned model {row.get('model')!r}, expected {model!r}")
    if row.get("output_tokens_used") != 0:
        raise RuntimeError(f"Reset did not zero usage for {model}: {row}")
    if row.get("circuit_open") is not False:
        raise RuntimeError(f"Reset did not close circuit for {model}: {row}")


def next_report_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_path = REPORT_DIR / f"{timestamp}_burst_responses.jsonl"
    if not base_path.exists():
        return base_path

    suffix = 1
    while True:
        candidate = REPORT_DIR / f"{timestamp}_burst_responses_{suffix}.jsonl"
        if not candidate.exists():
            return candidate
        suffix += 1


def build_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
    }


async def run_workflow(
    *,
    client: httpx.AsyncClient,
    provider: str,
    payload: dict[str, Any],
    workflow_number: int,
    calls_per_workflow: int,
    stop_event: asyncio.Event,
    report,
) -> None:
    trace_id = str(uuid4())

    for call_number in range(1, calls_per_workflow + 1):
        if stop_event.is_set():
            return

        span_name = f"workflow_{workflow_number}_call_{call_number}"
        response = await call_proxy_call(
            client=client,
            provider=provider,
            payload=payload,
            trace_id=trace_id,
            span_name=span_name,
        )

        write_report_row(
            report=report,
            trace_id=trace_id,
            span_name=span_name,
            status_code=response.status_code,
        )
        print(f"trace_id={trace_id} span_name={span_name} status_code={response.status_code}", flush=True)

        if is_output_quota_exceeded(response):
            stop_event.set()


async def call_proxy_call(
    *,
    client: httpx.AsyncClient,
    provider: str,
    payload: dict[str, Any],
    trace_id: str,
    span_name: str,
) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        headers={
            "X-Trace-ID": trace_id,
            "X-Span-Name": span_name,
            "X-LLM-Provider": provider,
        },
        json=payload,
    )


def write_report_row(*, report, trace_id: str, span_name: str, status_code: int) -> None:
    report.write(
        json.dumps(
            {
                "trace_id": trace_id,
                "span_name": span_name,
                "status_code": status_code,
            },
            separators=(",", ":"),
        )
    )
    report.write("\n")
    report.flush()


def is_output_quota_exceeded(response: httpx.Response) -> bool:
    if response.status_code != 429:
        return False
    try:
        body = response.json()
    except json.JSONDecodeError:
        return False
    return body.get("error", {}).get("code") == QUOTA_ERROR_CODE


async def usage_for_model(client: httpx.AsyncClient, model: str) -> dict[str, Any]:
    response = await client.get("/internal/usage/models")
    response.raise_for_status()
    for row in response.json():
        if row.get("model") == model:
            return row
    raise RuntimeError(f"No usage row found for model {model}")


if __name__ == "__main__":
    asyncio.run(main())
