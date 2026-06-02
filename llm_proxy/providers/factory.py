from __future__ import annotations

from llm_proxy.config import ProviderConfig
from llm_proxy.providers.base import Provider
from llm_proxy.providers.openai_compatible import OpenAICompatibleProvider
from llm_proxy.providers.smart_mock import SmartMockProvider


def build_provider(config: ProviderConfig) -> Provider:
    if config.type == "smart_mock":
        return SmartMockProvider(default_model=config.default_model)
    if config.type == "openai_compatible":
        if not config.base_url or not config.api_key_env:
            raise ValueError("openai_compatible providers require base_url and api_key_env")
        return OpenAICompatibleProvider(
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            default_model=config.default_model,
        )
    raise ValueError(f"Unsupported provider type: {config.type}")
