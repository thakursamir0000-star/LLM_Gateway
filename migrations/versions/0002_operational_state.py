"""Add latency classification, alerts and in-flight budget reservations."""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0002_operational_state"
down_revision = "0001_core_schema"
branch_labels = None
depends_on = None

TIMESTAMP = sa.DateTime(timezone=True).with_variant(sa.Text(), "sqlite")
DAY = sa.Date().with_variant(sa.Text(), "sqlite")
MONEY = sa.Numeric(20, 12).with_variant(sa.Float(), "sqlite")


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    tables = _tables()
    if "latency_class" not in _columns("requests"):
        op.add_column("requests", sa.Column("latency_class", sa.Text()))
    if "alerts" not in tables:
        op.create_table(
            "alerts",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("ts", TIMESTAMP, nullable=False),
            sa.Column("kind", sa.Text(), nullable=False),
            sa.Column("value", sa.Float()),
            sa.Column("threshold", sa.Float()),
            sa.Column("message", sa.Text(), nullable=False),
            sqlite_autoincrement=True,
        )
    if "budget_reservations" not in tables:
        op.create_table(
            "budget_reservations",
            sa.Column("request_id", sa.Text(), primary_key=True),
            sa.Column("app_id", sa.Text(), nullable=False),
            sa.Column("day", DAY, nullable=False),
            sa.Column("amount_usd", MONEY, nullable=False),
            sa.Column("created_at", TIMESTAMP, nullable=False),
            sa.Column("expires_at", TIMESTAMP, nullable=False),
        )
        op.create_index(
            "budget_reservations_app_day", "budget_reservations", ["app_id", "day"]
        )


def downgrade() -> None:
    if os.environ.get("GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE") != "1":
        raise RuntimeError("destructive downgrade requires GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE=1")
    tables = _tables()
    if "budget_reservations" in tables:
        op.drop_index("budget_reservations_app_day", table_name="budget_reservations")
        op.drop_table("budget_reservations")
    if "alerts" in tables:
        op.drop_table("alerts")
    if "latency_class" in _columns("requests"):
        with op.batch_alter_table("requests") as batch:
            batch.drop_column("latency_class")
