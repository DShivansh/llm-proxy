from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any
from uuid import uuid4

import httpx


PROMPT = (
    "Explain how a local LLM proxy should handle streaming responses, tracing, "
    "quota accounting, and cancellation during a parallel agent workflow. "
    "Respond with 6 concise bullet points."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming requests through a running LLM proxy.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--parallel-cancel-demo", action="store_true")
    parser.add_argument("--streams", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.streams < 1:
        raise ValueError("--streams must be at least 1")

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        await require_healthy(client)
        reset_row = await reset_usage(client, args.model)
        print_quota_note(reset_row, args.model)

        if args.parallel_cancel_demo:
            await run_parallel_demo(client, args)
        else:
            trace_id = str(uuid4())
            await stream_call(
                client=client,
                provider=args.provider,
                model=args.model,
                trace_id=trace_id,
                span_name="streaming_single",
                prefix="single",
            )
            print(f"\ntrace_url={args.base_url.rstrip('/')}/internal/traces/{trace_id}")


async def run_parallel_demo(client: httpx.AsyncClient, args: argparse.Namespace) -> None:
    trace_id = str(uuid4())
    tasks = [
        asyncio.create_task(
            stream_call(
                client=client,
                provider=args.provider,
                model=args.model,
                trace_id=trace_id,
                span_name=f"streaming_parallel_{index}",
                prefix=f"stream-{index}",
            )
        )
        for index in range(1, args.streams + 1)
    ]
    await asyncio.gather(*tasks)
    print(f"\ntrace_url={args.base_url.rstrip('/')}/internal/traces/{trace_id}")


async def stream_call(
    *,
    client: httpx.AsyncClient,
    provider: str,
    model: str,
    trace_id: str,
    span_name: str,
    prefix: str,
) -> None:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "X-Trace-ID": trace_id,
            "X-Span-Name": span_name,
            "X-LLM-Provider": provider,
        },
        json={
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.2,
        },
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            print(f"{prefix} status={response.status_code} body={body!r}")
            return

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                print(f"\n{prefix} done")
                return
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                print(f"\n{prefix} raw={data}")
                continue
            print_stream_payload(prefix, payload)


def print_stream_payload(prefix: str, payload: dict[str, Any]) -> None:
    if "error" in payload:
        print(f"\n{prefix} error={payload['error'].get('code')} message={payload['error'].get('message')}")
        return
    if "warning" in payload:
        print(f"\n{prefix} warning={payload['warning'].get('code')} message={payload['warning'].get('message')}")
        return

    for choice in payload.get("choices") or []:
        content = (choice.get("delta") or {}).get("content")
        if content:
            print(content, end="", flush=True)


async def require_healthy(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    if response.status_code != 200:
        raise RuntimeError(f"Health check failed: status_code={response.status_code} body={response.text}")


async def reset_usage(client: httpx.AsyncClient, model: str) -> dict[str, Any]:
    response = await client.post(f"/internal/usage/models/{model}/reset")
    response.raise_for_status()
    return response.json()


def print_quota_note(row: dict[str, Any], model: str) -> None:
    limit = row.get("output_token_limit")
    print(f"model={model} quota_limit={limit} usage_reset=true")
    if limit is None or limit > 1000:
        print(
            "For a cancellation demo, start the proxy with: "
            "LLM_PROXY_CONFIG=examples/config.streaming-local.yaml ./start-server.sh"
        )


if __name__ == "__main__":
    asyncio.run(main())
