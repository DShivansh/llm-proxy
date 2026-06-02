from __future__ import annotations

from math import ceil
from typing import Any

import tiktoken


def estimate_text_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        return max(1, ceil(len(text) / 4))
    return len(encoding.encode(text))


def estimate_messages_tokens(messages: list[dict[str, Any]], model: str) -> int:
    total = 0
    for message in messages:
        total += estimate_text_tokens(str(message.get("role", "")), model)
        content = message.get("content", "")
        if isinstance(content, str):
            total += estimate_text_tokens(content, model)
        else:
            total += estimate_text_tokens(str(content), model)
    return total

