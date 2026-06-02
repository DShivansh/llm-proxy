from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./llm_proxy.db"


class TracingConfig(BaseModel):
    log_full_body: bool = True


class QuotaConfig(BaseModel):
    period: Literal["manual", "daily", "monthly"] = "manual"


class ProviderConfig(BaseModel):
    type: Literal["smart_mock", "openai_compatible"]
    default_model: str
    base_url: str | None = None
    api_key_env: str | None = None


class ProvidersConfig(BaseModel):
    default: str = "mock"
    entries: dict[str, ProviderConfig] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    output_token_limit: int | None = None


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    providers: ProvidersConfig
    models: dict[str, ModelConfig] = Field(default_factory=dict)


def _normalize_providers(raw: dict) -> dict:
    providers = raw.get("providers", {})
    default = providers.get("default", "mock")
    entries = {name: value for name, value in providers.items() if name != "default"}
    return {**raw, "providers": {"default": default, "entries": entries}}


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("LLM_PROXY_CONFIG", "config.yaml"))
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(_normalize_providers(raw))

