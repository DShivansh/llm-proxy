from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from llm_proxy.models import ModelUsage, Span, Trace
from llm_proxy.time import utc_iso, utc_now


def sqlite_path(database_url: str) -> str:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Only sqlite:/// database URLs are supported")
    return database_url.removeprefix(prefix)


class Store:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        path = sqlite_path(database_url)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            future=True,
        )
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def ensure_trace(self, trace_id: str) -> None:
        now = utc_iso(utc_now())
        with self.session_factory.begin() as session:
            trace = session.get(Trace, trace_id)
            if trace is None:
                session.add(Trace(trace_id=trace_id, created_at=now, updated_at=now))
            else:
                trace.updated_at = now

    def insert_span(self, span: dict[str, Any]) -> None:
        now = utc_iso(utc_now())
        with self.session_factory.begin() as session:
            trace = session.get(Trace, span["trace_id"])
            if trace is None:
                trace = Trace(trace_id=span["trace_id"], created_at=now, updated_at=now)
                session.add(trace)
            else:
                trace.updated_at = now

            session.add(
                Span(
                    span_id=span["span_id"],
                    trace_id=span["trace_id"],
                    span_name=span["span_name"],
                    started_at=self._serialize(span.get("started_at")),
                    ended_at=self._serialize(span.get("ended_at")),
                    latency_ms=span.get("latency_ms"),
                    provider=span.get("provider"),
                    upstream_base_url=span.get("upstream_base_url"),
                    model=span.get("model"),
                    status=span["status"],
                    http_status_code=span.get("http_status_code"),
                    input_tokens=span.get("input_tokens") or 0,
                    output_tokens=span.get("output_tokens") or 0,
                    total_tokens=span.get("total_tokens") or 0,
                    usage_source=span.get("usage_source"),
                    quota_limit=span.get("quota_limit"),
                    quota_used_before=span.get("quota_used_before") or 0,
                    quota_used_after=span.get("quota_used_after") or 0,
                    circuit_opened_by_this_span=bool(span.get("circuit_opened_by_this_span")),
                    request_body=self._serialize(span.get("request_body")),
                    response_body=self._serialize(span.get("response_body")),
                    error_message=span.get("error_message"),
                )
            )

    def get_usage(self, model: str, period_key: str) -> dict[str, Any]:
        with self.session_factory() as session:
            usage = session.get(ModelUsage, {"model": model, "period_key": period_key})
            if usage is None:
                return {
                    "model": model,
                    "period_key": period_key,
                    "output_tokens_used": 0,
                    "circuit_open": False,
                }
            return self._usage_to_dict(usage)

    def add_usage(self, model: str, period_key: str, output_tokens: int, limit: int | None) -> dict[str, Any]:
        with self.session_factory.begin() as session:
            usage = session.get(ModelUsage, {"model": model, "period_key": period_key})
            if usage is None:
                usage = ModelUsage(
                    model=model,
                    period_key=period_key,
                    output_tokens_used=0,
                    circuit_open=False,
                    updated_at=utc_iso(utc_now()),
                )
                session.add(usage)
                session.flush()

            used_before = usage.output_tokens_used
            circuit_was_open = usage.circuit_open
            used_after = used_before + output_tokens
            circuit_is_open = circuit_was_open or (limit is not None and used_after >= limit)

            usage.output_tokens_used = used_after
            usage.circuit_open = circuit_is_open
            usage.updated_at = utc_iso(utc_now())

        return {
            "used_before": used_before,
            "used_after": used_after,
            "circuit_was_open": circuit_was_open,
            "circuit_is_open": circuit_is_open,
            "opened_by_this_span": not circuit_was_open and circuit_is_open,
        }

    def list_usage(self) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            usage_rows = session.scalars(
                select(ModelUsage).order_by(ModelUsage.model, ModelUsage.period_key)
            ).all()
            return [self._usage_to_dict(usage) for usage in usage_rows]

    def reset_usage(self, model: str, period_key: str) -> dict[str, Any]:
        with self.session_factory.begin() as session:
            usage = session.get(ModelUsage, {"model": model, "period_key": period_key})
            if usage is None:
                usage = ModelUsage(
                    model=model,
                    period_key=period_key,
                    output_tokens_used=0,
                    circuit_open=False,
                    updated_at=utc_iso(utc_now()),
                )
                session.add(usage)
            else:
                usage.output_tokens_used = 0
                usage.circuit_open = False
                usage.updated_at = utc_iso(utc_now())
            return self._usage_to_dict(usage)

    def list_trace_summaries(self) -> list[dict[str, Any]]:
        with self.session_factory() as session:
            rows = session.execute(
                select(
                    Trace.trace_id,
                    Trace.created_at,
                    Trace.updated_at,
                    func.count(Span.span_id).label("span_count"),
                    func.coalesce(func.sum(Span.input_tokens), 0).label("total_input_tokens"),
                    func.coalesce(func.sum(Span.output_tokens), 0).label("total_output_tokens"),
                )
                .outerjoin(Span)
                .group_by(Trace.trace_id)
                .order_by(desc(Trace.updated_at))
            ).all()
            return [self._trace_summary(session, row) for row in rows]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self.session_factory() as session:
            trace = session.get(Trace, trace_id)
            if trace is None:
                return None
            spans = session.scalars(
                select(Span).where(Span.trace_id == trace_id).order_by(Span.started_at)
            ).all()
            return {
                "trace_id": trace.trace_id,
                "created_at": trace.created_at,
                "updated_at": trace.updated_at,
                "spans": [self._span_to_dict(span) for span in spans],
            }

    def _trace_summary(self, session: Session, row: Any) -> dict[str, Any]:
        spans = session.scalars(
            select(Span).where(Span.trace_id == row.trace_id).order_by(Span.started_at)
        ).all()
        statuses = [span.status for span in spans]
        models = sorted({span.model for span in spans if span.model})
        status = "success"
        if "blocked" in statuses:
            status = "blocked"
        elif "error" in statuses:
            status = "error"
        elif "cancelled" in statuses:
            status = "cancelled"
        return {
            "trace_id": row.trace_id,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "span_count": row.span_count,
            "status": status,
            "models": models,
            "total_input_tokens": row.total_input_tokens,
            "total_output_tokens": row.total_output_tokens,
        }

    def _span_to_dict(self, span: Span) -> dict[str, Any]:
        return {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "span_name": span.span_name,
            "started_at": span.started_at,
            "ended_at": span.ended_at,
            "latency_ms": span.latency_ms,
            "provider": span.provider,
            "upstream_base_url": span.upstream_base_url,
            "model": span.model,
            "status": span.status,
            "http_status_code": span.http_status_code,
            "input_tokens": span.input_tokens,
            "output_tokens": span.output_tokens,
            "total_tokens": span.total_tokens,
            "usage_source": span.usage_source,
            "quota_limit": span.quota_limit,
            "quota_used_before": span.quota_used_before,
            "quota_used_after": span.quota_used_after,
            "circuit_opened_by_this_span": span.circuit_opened_by_this_span,
            "request_body": json.loads(span.request_body) if span.request_body else None,
            "response_body": json.loads(span.response_body) if span.response_body else None,
            "error_message": span.error_message,
        }

    def _usage_to_dict(self, usage: ModelUsage) -> dict[str, Any]:
        return {
            "model": usage.model,
            "period_key": usage.period_key,
            "output_tokens_used": usage.output_tokens_used,
            "circuit_open": usage.circuit_open,
        }

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, bool):
            return value
        if isinstance(value, datetime):
            return utc_iso(value)
        return value
