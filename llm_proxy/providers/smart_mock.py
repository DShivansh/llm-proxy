from __future__ import annotations

from time import time_ns
from typing import Any

from llm_proxy.providers.base import ProviderResult


MOCK_OUTPUT_TOKENS = {
    "planner_agent": 120,
    "researcher_agent": 130,
    "writer_agent": 160,
    "reviewer_agent": 120,
    "final_editor_agent": 90,
}

MOCK_RESPONSES = {
    "planner_agent": "Launch plan: define the proxy story, show the quota guardrail, and keep the demo local-first.",
    "researcher_agent": "Research notes: developers want OpenAI-compatible proxies, durable traces, and clear quota behavior.",
    "writer_agent": "Draft: this sidecar records each agent call under one trace and opens a model circuit when output budget is spent.",
    "reviewer_agent": "Review: the demo is credible because the successful response that crosses quota is returned before future calls are blocked.",
    "final_editor_agent": "Final polish: tighten the summary and include the trace URL.",
}


class SmartMockProvider:
    def __init__(self, default_model: str) -> None:
        self.default_model = default_model

    async def chat_completion(
        self,
        *,
        body: dict[str, Any],
        span_name: str,
        trace_id: str,
    ) -> ProviderResult:
        model = body.get("model") or self.default_model
        output_tokens = MOCK_OUTPUT_TOKENS.get(span_name, 80)
        input_tokens = 35
        total_tokens = input_tokens + output_tokens
        content = MOCK_RESPONSES.get(
            span_name,
            "Mock response: the proxy accepted this OpenAI-compatible chat completion request.",
        )
        response_body = {
            "id": f"chatcmpl-mock-{time_ns()}",
            "object": "chat.completion",
            "created": int(time_ns() / 1_000_000_000),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
        return ProviderResult(
            body=response_body,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_source="mock",
        )

