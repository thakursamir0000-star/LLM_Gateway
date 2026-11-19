"""Add alert deduplication and PostgreSQL request foreign keys."""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0003_distributed_integrity"
down_revision = "0002_operational_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    alert_columns = {column["name"] for column in inspector.get_columns("alerts")}
    if "dedup_key" not in alert_columns:
        op.add_column("alerts", sa.Column("dedup_key", sa.Text()))
    indexes = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("alerts")}
    if "alerts_dedup_key" not in indexes:
        op.create_index("alerts_dedup_key", "alerts", ["dedup_key"], unique=True)
    if op.get_bind().dialect.name == "postgresql":
        call_fks = {fk["name"] for fk in inspector.get_foreign_keys("calls")}
        feedback_fks = {fk["name"] for fk in inspector.get_foreign_keys("feedback")}
        if "calls_request_id_fkey" not in call_fks:
            op.create_foreign_key(
                "calls_request_id_fkey", "calls", "requests", ["request_id"], ["id"], ondelete="CASCADE"
            )
        if "feedback_request_id_fkey" not in feedback_fks:
            op.create_foreign_key(
                "feedback_request_id_fkey", "feedback", "requests", ["request_id"], ["id"], ondelete="CASCADE"
            )


def downgrade() -> None:
    if os.environ.get("GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE") != "1":
        raise RuntimeError("destructive downgrade requires GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE=1")
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint("feedback_request_id_fkey", "feedback", type_="foreignkey")
        op.drop_constraint("calls_request_id_fkey", "calls", type_="foreignkey")
    op.drop_index("alerts_dedup_key", table_name="alerts")
    with op.batch_alter_table("alerts") as batch:
        batch.drop_column("dedup_key")
