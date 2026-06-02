from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

import asyncio
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from llm_proxy.active import ActiveRequestRegistry
from llm_proxy.config import AppConfig
from llm_proxy.migrations import run_migrations
from llm_proxy.providers import ProviderResult, build_provider
from llm_proxy.quota import QuotaService
from llm_proxy.storage import Store
from llm_proxy.time import utc_iso, utc_now


STREAMING_ERROR = {
    "error": {
        "message": "Streaming is not supported by this proxy MVP",
        "type": "unsupported_feature",
        "code": "streaming_not_supported",
    }
}


def create_app(config: AppConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        run_migrations(config.database.url)
        print("migrations ran")
        yield

    print("starting a lifespan")
    app = FastAPI(title="LLM Sidecar Proxy", lifespan=lifespan)
    print("initializing store")
    store = Store(config.database.url)
    quota = QuotaService(config, store)
    active_requests = ActiveRequestRegistry()
    providers = {
        name: build_provider(provider_config)
        for name, provider_config in config.providers.entries.items()
    }

    app.state.config = config
    app.state.store = store
    app.state.quota = quota
    app.state.active_requests = active_requests
    app.state.providers = providers
    print(f"everything is done and the config is {config}")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
        x_span_name: str | None = Header(default=None, alias="X-Span-Name"),
        x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    ) -> JSONResponse:
        request_body = await request.json()
        trace_id = x_trace_id or str(uuid4())
        span_name = x_span_name or "chat_completion"
        provider_name = x_llm_provider or config.providers.default
        started_at = utc_now()
        started = perf_counter()

        provider_config = config.providers.entries.get(provider_name)
        if provider_config is None:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")

        model = request_body.get("model") or provider_config.default_model

        if request_body.get("stream") is True:
            _record_span(
                store,
                trace_id=trace_id,
                span_name=span_name,
                started_at=started_at,
                started=started,
                provider=provider_name,
                model=model,
                status="error",
                http_status_code=400,
                request_body=request_body,
                response_body=STREAMING_ERROR,
                error_message=STREAMING_ERROR["error"]["message"],
                log_body=config.tracing.log_full_body,
            )
            return JSONResponse(STREAMING_ERROR, status_code=400)

        quota_state = quota.get_state(model)
        if quota_state.circuit_open:
            response_body = _quota_error(model)
            _record_span(
                store,
                trace_id=trace_id,
                span_name=span_name,
                started_at=started_at,
                started=started,
                provider=provider_name,
                model=model,
                status="blocked",
                http_status_code=429,
                request_body=request_body,
                response_body=response_body,
                error_message=response_body["error"]["message"],
                quota_limit=quota_state.limit,
                quota_used_before=quota_state.used,
                quota_used_after=quota_state.used,
                log_body=config.tracing.log_full_body,
            )
            return JSONResponse(response_body, status_code=429)

        try:
            with active_requests.track(model) as current_task:
                result = await providers[provider_name].chat_completion(
                    body=request_body,
                    span_name=span_name,
                    trace_id=trace_id,
                )
        except httpx.HTTPStatusError as exc:
            response_body = exc.response.json()
            _record_span(
                store,
                trace_id=trace_id,
                span_name=span_name,
                started_at=started_at,
                started=started,
                provider=provider_name,
                model=model,
                status="error",
                http_status_code=exc.response.status_code,
                request_body=request_body,
                response_body=response_body,
                error_message=str(exc),
                log_body=config.tracing.log_full_body,
            )
            return JSONResponse(response_body, status_code=exc.response.status_code)
        except asyncio.CancelledError:
            response_body = {
                "error": {
                    "message": f"Request cancelled because output token circuit opened for model {model}",
                    "type": "cancelled",
                    "code": "request_cancelled",
                }
            }
            _record_span(
                store,
                trace_id=trace_id,
                span_name=span_name,
                started_at=started_at,
                started=started,
                provider=provider_name,
                model=model,
                status="cancelled",
                http_status_code=499,
                request_body=request_body,
                response_body=response_body,
                error_message=response_body["error"]["message"],
                log_body=config.tracing.log_full_body,
            )
            return JSONResponse(response_body, status_code=499)
        except Exception as exc:
            response_body = {"error": {"message": str(exc), "type": "provider_error", "code": "provider_error"}}
            _record_span(
                store,
                trace_id=trace_id,
                span_name=span_name,
                started_at=started_at,
                started=started,
                provider=provider_name,
                model=model,
                status="error",
                http_status_code=502,
                request_body=request_body,
                response_body=response_body,
                error_message=str(exc),
                log_body=config.tracing.log_full_body,
            )
            return JSONResponse(response_body, status_code=502)

        accounting = quota.record_output(result.model, result.output_tokens or 0)
        if accounting["opened_by_this_span"]:
            active_requests.cancel_others(result.model, current_task)
        _record_success_span(
            store,
            trace_id=trace_id,
            span_name=span_name,
            started_at=started_at,
            started=started,
            provider=provider_name,
            request_body=request_body,
            result=result,
            quota_limit=quota.model_limit(result.model),
            accounting=accounting,
            log_body=config.tracing.log_full_body,
        )
        return JSONResponse(result.body, status_code=200)

    @app.get("/internal/traces")
    async def list_traces() -> list[dict[str, Any]]:
        return store.list_trace_summaries()

    @app.get("/internal/traces/{trace_id}")
    async def get_trace(trace_id: str) -> dict[str, Any]:
        trace = store.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace

    @app.get("/internal/usage/models")
    async def list_model_usage() -> list[dict[str, Any]]:
        return [
            {
                **usage,
                "output_token_limit": quota.model_limit(usage["model"]),
            }
            for usage in store.list_usage()
        ]

    @app.post("/internal/usage/models/{model}/reset")
    async def reset_model_usage(model: str) -> dict[str, Any]:
        usage = quota.reset(model)
        return {
            **usage,
            "output_token_limit": quota.model_limit(model),
        }

    return app


def _record_success_span(
    store: Store,
    *,
    trace_id: str,
    span_name: str,
    started_at,
    started: float,
    provider: str,
    request_body: dict[str, Any],
    result: ProviderResult,
    quota_limit: int | None,
    accounting: dict[str, Any],
    log_body: bool,
) -> None:
    _record_span(
        store,
        trace_id=trace_id,
        span_name=span_name,
        started_at=started_at,
        started=started,
        provider=provider,
        upstream_base_url=result.upstream_base_url,
        model=result.model,
        status="success",
        http_status_code=200,
        input_tokens=result.input_tokens or 0,
        output_tokens=result.output_tokens or 0,
        total_tokens=result.total_tokens or 0,
        usage_source=result.usage_source,
        quota_limit=quota_limit,
        quota_used_before=accounting["used_before"],
        quota_used_after=accounting["used_after"],
        circuit_opened_by_this_span=accounting["opened_by_this_span"],
        request_body=request_body,
        response_body=result.body,
        log_body=log_body,
    )


def _record_span(
    store: Store,
    *,
    trace_id: str,
    span_name: str,
    started_at,
    started: float,
    provider: str,
    model: str,
    status: str,
    http_status_code: int,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    log_body: bool,
    upstream_base_url: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    usage_source: str | None = None,
    quota_limit: int | None = None,
    quota_used_before: int = 0,
    quota_used_after: int = 0,
    circuit_opened_by_this_span: bool = False,
    error_message: str | None = None,
) -> None:
    ended_at = utc_now()
    store.insert_span(
        {
            "span_id": str(uuid4()),
            "trace_id": trace_id,
            "span_name": span_name,
            "started_at": utc_iso(started_at),
            "ended_at": utc_iso(ended_at),
            "latency_ms": int((perf_counter() - started) * 1000),
            "provider": provider,
            "upstream_base_url": upstream_base_url,
            "model": model,
            "status": status,
            "http_status_code": http_status_code,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "usage_source": usage_source,
            "quota_limit": quota_limit,
            "quota_used_before": quota_used_before,
            "quota_used_after": quota_used_after,
            "circuit_opened_by_this_span": circuit_opened_by_this_span,
            "request_body": request_body if log_body else None,
            "response_body": response_body if log_body else None,
            "error_message": error_message,
        }
    )


def _quota_error(model: str) -> dict[str, Any]:
    return {
        "error": {
            "message": f"Output token circuit is open for model {model}",
            "type": "quota_exceeded",
            "code": "output_quota_exceeded",
        }
    }
