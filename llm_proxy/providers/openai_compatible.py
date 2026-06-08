from __future__ import annotations

import os
from typing import Any

import httpx

from llm_proxy.providers.base import ProviderResult
from llm_proxy.tokens import estimate_messages_tokens, estimate_text_tokens


class OpenAICompatibleProvider:
    def __init__(self, *, base_url: str, api_key_env: str | None, default_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.default_model = default_model

    async def chat_completion(
        self,
        *,
        body: dict[str, Any],
        span_name: str,
        trace_id: str,
    ) -> ProviderResult:
        api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        if self.api_key_env and not api_key:
            raise RuntimeError(f"Missing provider API key environment variable: {self.api_key_env}")

        request_body = {**body}
        request_body.setdefault("model", self.default_model)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()

        response_body = response.json()
        model = response_body.get("model") or request_body["model"]
        usage = response_body.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        usage_source = "provider"

        if output_tokens is None:
            choices_text = " ".join(
                str(choice.get("message", {}).get("content", ""))
                for choice in response_body.get("choices", [])
            )
            output_tokens = estimate_text_tokens(choices_text, model)
            input_tokens = input_tokens or estimate_messages_tokens(request_body.get("messages", []), model)
            total_tokens = input_tokens + output_tokens
            usage_source = "estimated"

        return ProviderResult(
            body=response_body,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_source=usage_source,
            upstream_base_url=self.base_url,
        )
