from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderResult:
    body: dict[str, Any]
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_source: str
    upstream_base_url: str | None = None


class Provider(Protocol):
    async def chat_completion(
        self,
        *,
        body: dict[str, Any],
        span_name: str,
        trace_id: str,
    ) -> ProviderResult:
        ...

