"""Record the shadow router's decision on each request (router shadow mode)."""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0004_shadow_routing"
down_revision = "0003_distributed_integrity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("requests")}
    if "shadow_score" not in columns:
        op.add_column("requests", sa.Column("shadow_score", sa.Float()))
    if "shadow_tier" not in columns:
        op.add_column("requests", sa.Column("shadow_tier", sa.Text()))


def downgrade() -> None:
    if os.environ.get("GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE") != "1":
        raise RuntimeError("destructive downgrade requires GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE=1")
    op.drop_column("requests", "shadow_tier")
    op.drop_column("requests", "shadow_score")
