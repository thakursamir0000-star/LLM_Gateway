"""Create the original gateway request, call, feedback and deployment schema."""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0001_core_schema"
down_revision = None
branch_labels = None
depends_on = None

TIMESTAMP = sa.DateTime(timezone=True).with_variant(sa.Text(), "sqlite")
DAY = sa.Date().with_variant(sa.Text(), "sqlite")
MONEY = sa.Numeric(20, 12).with_variant(sa.Float(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "requests",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("ts", TIMESTAMP, nullable=False),
        sa.Column("day", DAY, nullable=False),
        sa.Column("app_id", sa.Text(), nullable=False),
        sa.Column("requested_model", sa.Text()),
        sa.Column("release", sa.Text()),
        sa.Column("prompt_version", sa.Text()),
        sa.Column("canary", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("mode", sa.Text()),
        sa.Column("router", sa.Text()),
        sa.Column("route_score", sa.Float()),
        sa.Column("route_reason", sa.Text()),
        sa.Column("threshold", sa.Float()),
        sa.Column("tier", sa.Text()),
        sa.Column("model", sa.Text()),
        sa.Column("provider_model", sa.Text()),
        sa.Column("escalated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fallback", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", MONEY, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("router_ms", sa.Float()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("http_status", sa.Integer()),
        sa.Column("error_type", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("stream", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("requests_ts", "requests", ["ts"])
    op.create_index("requests_day_app", "requests", ["day", "app_id"])
    op.create_table(
        "calls",
        sa.Column("request_id", sa.Text(), primary_key=True),
        sa.Column("seq", sa.Integer(), primary_key=True),
        sa.Column("ts", TIMESTAMP, nullable=False),
        sa.Column("tier", sa.Text()),
        sa.Column("model", sa.Text()),
        sa.Column("provider_model", sa.Text()),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", MONEY, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_type", sa.Text()),
        sa.Column("error_message", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column("check_reason", sa.Text()),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("ts", TIMESTAMP, nullable=False),
        sa.Column("app_id", sa.Text(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text()),
        sa.CheckConstraint("score IN (-1, 1)"),
        sqlite_autoincrement=True,
    )
    op.create_index("feedback_request", "feedback", ["request_id"])
    op.create_table(
        "deployments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ts", TIMESTAMP, nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("active", sa.Text(), nullable=False),
        sa.Column("canary_release", sa.Text()),
        sa.Column("canary_share", sa.Float(), nullable=False, server_default="0"),
        sa.Column("note", sa.Text()),
        sqlite_autoincrement=True,
    )


def downgrade() -> None:
    if os.environ.get("GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE") != "1":
        raise RuntimeError("destructive downgrade requires GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE=1")
    op.drop_table("deployments")
    op.drop_index("feedback_request", table_name="feedback")
    op.drop_table("feedback")
    op.drop_table("calls")
    op.drop_index("requests_day_app", table_name="requests")
    op.drop_index("requests_ts", table_name="requests")
    op.drop_table("requests")
