from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Trace(Base):
    __tablename__ = "traces"

    trace_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)

    spans: Mapped[list[Span]] = relationship(
        back_populates="trace",
        cascade="all, delete-orphan",
        order_by="Span.started_at",
    )


class Span(Base):
    __tablename__ = "spans"

    span_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.trace_id"), nullable=False, index=True)
    span_name: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[str] = mapped_column(String, nullable=False)
    ended_at: Mapped[str | None] = mapped_column(String)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    provider: Mapped[str | None] = mapped_column(String)
    upstream_base_url: Mapped[str | None] = mapped_column(String)
    model: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    http_status_code: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    usage_source: Mapped[str | None] = mapped_column(String)
    quota_limit: Mapped[int | None] = mapped_column(Integer)
    quota_used_before: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quota_used_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    circuit_opened_by_this_span: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    request_body: Mapped[str | None] = mapped_column(Text)
    response_body: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    trace: Mapped[Trace] = relationship(back_populates="spans")


class ModelUsage(Base):
    __tablename__ = "model_usage"
    __table_args__ = (UniqueConstraint("model", "period_key", name="uq_model_usage_model_period"),)

    model: Mapped[str] = mapped_column(String, primary_key=True)
    period_key: Mapped[str] = mapped_column(String, primary_key=True)
    output_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    circuit_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
