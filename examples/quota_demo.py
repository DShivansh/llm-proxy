from __future__ import annotations

import argparse
from typing import Any

import httpx


SPAN_NAMES = [
    "planner_agent",
    "researcher_agent",
    "writer_agent",
    "reviewer_agent",
    "final_editor_agent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM proxy quota demo workflow.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--trace-id", default="launch-demo-001")
    parser.add_argument("--provider", default="mock", choices=["mock", "groq", "local"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = httpx.Client(base_url=args.base_url, timeout=30.0)
    reset_demo_model(client, args.provider)

    print(f"Trace: {args.trace_id}")
    print()

    for span_name in SPAN_NAMES:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "X-Trace-ID": args.trace_id,
                "X-Span-Name": span_name,
                "X-LLM-Provider": args.provider,
            },
            json={
                "model": model_for_provider(args.provider),
                "messages": [{"role": "user", "content": "Plan the launch for the LLM sidecar proxy."}],
            },
        )
        print(format_row(client, span_name, response))

    print()
    print("Open full trace:")
    print(f"{args.base_url.rstrip('/')}/internal/traces/{args.trace_id}")


def reset_demo_model(client: httpx.Client, provider: str) -> None:
    client.post(f"/internal/usage/models/{model_for_provider(provider)}/reset")


def model_for_provider(provider: str) -> str:
    if provider == "groq":
        return "llama-3.1-8b-instant"
    if provider == "local":
        return "llama3.2:1b"
    return "gpt-4.1-mini"


def format_row(client: httpx.Client, span_name: str, response: httpx.Response) -> str:
    if response.status_code == 429:
        return f"{span_name:<19} blocked   circuit already open"

    response.raise_for_status()
    body = response.json()
    output_tokens = body["usage"]["completion_tokens"]
    usage = usage_for_model(client, body["model"])
    suffix = " circuit opened" if usage["circuit_open"] else ""
    return (
        f"{span_name:<19} success   output={output_tokens:<5} "
        f"used={usage['output_tokens_used']}/{usage['output_token_limit']}{suffix}"
    )


def usage_for_model(client: httpx.Client, model: str) -> dict[str, Any]:
    response = client.get("/internal/usage/models")
    response.raise_for_status()
    for usage in response.json():
        if usage["model"] == model:
            return usage
    raise RuntimeError(f"No usage row found for model {model}")


if __name__ == "__main__":
    main()
