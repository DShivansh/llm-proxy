from __future__ import annotations

from dataclasses import dataclass

from llm_proxy.config import AppConfig
from llm_proxy.storage import Store
from llm_proxy.time import utc_now


@dataclass(frozen=True)
class QuotaState:
    limit: int | None
    used: int
    circuit_open: bool
    period_key: str


class QuotaService:
    def __init__(self, config: AppConfig, store: Store) -> None:
        self.config = config
        self.store = store

    def period_key(self) -> str:
        now = utc_now()
        if self.config.quota.period == "manual":
            return "manual"
        if self.config.quota.period == "daily":
            return now.strftime("%Y-%m-%d")
        return now.strftime("%Y-%m")

    def model_limit(self, model: str) -> int | None:
        model_config = self.config.models.get(model)
        return model_config.output_token_limit if model_config else None

    def get_state(self, model: str) -> QuotaState:
        period_key = self.period_key()
        usage = self.store.get_usage(model, period_key)
        return QuotaState(
            limit=self.model_limit(model),
            used=usage["output_tokens_used"],
            circuit_open=usage["circuit_open"],
            period_key=period_key,
        )

    def record_output(self, model: str, output_tokens: int) -> dict:
        return self.store.add_usage(
            model=model,
            period_key=self.period_key(),
            output_tokens=output_tokens,
            limit=self.model_limit(model),
        )

    def reset(self, model: str) -> dict:
        return self.store.reset_usage(model, self.period_key())
