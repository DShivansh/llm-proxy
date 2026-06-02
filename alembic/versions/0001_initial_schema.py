from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "traces",
        sa.Column("trace_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("trace_id"),
    )
    op.create_table(
        "model_usage",
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("period_key", sa.String(), nullable=False),
        sa.Column("output_tokens_used", sa.Integer(), nullable=False),
        sa.Column("circuit_open", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("model", "period_key"),
        sa.UniqueConstraint("model", "period_key", name="uq_model_usage_model_period"),
    )
    op.create_table(
        "spans",
        sa.Column("span_id", sa.String(), nullable=False),
        sa.Column("trace_id", sa.String(), nullable=False),
        sa.Column("span_name", sa.String(), nullable=False),
        sa.Column("started_at", sa.String(), nullable=False),
        sa.Column("ended_at", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("upstream_base_url", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("http_status_code", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("usage_source", sa.String(), nullable=True),
        sa.Column("quota_limit", sa.Integer(), nullable=True),
        sa.Column("quota_used_before", sa.Integer(), nullable=False),
        sa.Column("quota_used_after", sa.Integer(), nullable=False),
        sa.Column("circuit_opened_by_this_span", sa.Boolean(), nullable=False),
        sa.Column("request_body", sa.Text(), nullable=True),
        sa.Column("response_body", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["trace_id"], ["traces.trace_id"]),
        sa.PrimaryKeyConstraint("span_id"),
    )
    op.create_index(op.f("ix_spans_trace_id"), "spans", ["trace_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_spans_trace_id"), table_name="spans")
    op.drop_table("spans")
    op.drop_table("model_usage")
    op.drop_table("traces")
