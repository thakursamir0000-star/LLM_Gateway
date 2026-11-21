"""Safely copy an LLM Gateway SQLite database into an empty migrated PostgreSQL database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# Ensure project root is importable when executed directly as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import create_engine, func, select

from gateway.migrations import verify_database_revision
from gateway.monitoring.store import (
    ALERTS,
    BUDGET_RESERVATIONS,
    CALLS,
    DEPLOYMENTS,
    FEEDBACK,
    REQUESTS,
)

TABLES = (REQUESTS, CALLS, FEEDBACK, DEPLOYMENTS, ALERTS, BUDGET_RESERVATIONS)


class DataMigrationError(RuntimeError):
    pass


def backup_sqlite(source: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = source.with_name(f"{source.name}.backup-{timestamp}")
    if backup.exists():
        raise DataMigrationError(f"backup already exists: {backup}")
    source_connection = sqlite3.connect(source)
    backup_connection = sqlite3.connect(backup)
    try:
        source_connection.backup(backup_connection)
    finally:
        backup_connection.close()
        source_connection.close()
    return backup


def _counts(connection: Any) -> dict[str, int]:
    return {table.name: int(connection.scalar(select(func.count()).select_from(table)) or 0) for table in TABLES}


def _daily_totals(connection: Any) -> dict[tuple[str, str], Decimal]:
    rows = connection.execute(select(
        REQUESTS.c.day,
        REQUESTS.c.app_id,
        func.coalesce(func.sum(REQUESTS.c.cost_usd), 0),
    ).group_by(REQUESTS.c.day, REQUESTS.c.app_id))
    return {(str(day), app_id): Decimal(str(cost)) for day, app_id, cost in rows}


# SQLite sums float costs and PostgreSQL sums exact NUMERIC(20, 12) ones, so the same costs can differ in the
# last float digits (0.1 + 0.2 is 0.30000000000000004 in SQLite, 0.3 in PostgreSQL): compare within a bound.
COST_TOLERANCE = Decimal("0.000000001")


def _daily_mismatch(source: dict[tuple[str, str], Decimal],
                    target: dict[tuple[str, str], Decimal]) -> dict[str, Any]:
    """Day/app groups missing on one side or whose cost differs by more than COST_TOLERANCE."""
    mismatch: dict[str, Any] = {}
    for key in sorted(source.keys() | target.keys()):
        if key not in source or key not in target or abs(source[key] - target[key]) > COST_TOLERANCE:
            mismatch["/".join(key)] = {"source": str(source.get(key)), "target": str(target.get(key))}
    return mismatch


def _orphans(connection: Any) -> dict[str, int]:
    call_orphans = connection.scalar(select(func.count()).select_from(
        CALLS.outerjoin(REQUESTS, CALLS.c.request_id == REQUESTS.c.id)
    ).where(REQUESTS.c.id.is_(None))) or 0
    feedback_orphans = connection.scalar(select(func.count()).select_from(
        FEEDBACK.outerjoin(REQUESTS, FEEDBACK.c.request_id == REQUESTS.c.id)
    ).where(REQUESTS.c.id.is_(None))) or 0
    return {"calls": int(call_orphans), "feedback": int(feedback_orphans)}


def _reset_sequences(connection: Any) -> None:
    if connection.dialect.name != "postgresql":
        return
    for table in (FEEDBACK, DEPLOYMENTS, ALERTS):
        connection.exec_driver_sql(
            "SELECT setval(pg_get_serial_sequence(%s, 'id'), "
            f"COALESCE((SELECT MAX(id) FROM {table.name}), 1), "
            f"(SELECT MAX(id) IS NOT NULL FROM {table.name}))",
            (table.name,),
        )


def migrate(source: Path, target_url: str, *, batch_size: int = 1000,
            verify_revision: bool = True) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise DataMigrationError(f"SQLite source does not exist: {source}")
    if verify_revision:
        verify_database_revision(target_url)
    backup = backup_sqlite(source)
    source_engine = create_engine(f"sqlite+pysqlite:///{source.as_posix()}")
    target_engine = create_engine(target_url)
    report: dict[str, Any] = {"source": str(source), "backup": str(backup), "tables": {}}
    try:
        with source_engine.connect() as source_connection, target_engine.connect() as target_connection:
            source_counts = _counts(source_connection)
            with target_connection.begin():
                target_counts_before = _counts(target_connection)
                nonempty = {name: count for name, count in target_counts_before.items() if count}
                if nonempty:
                    raise DataMigrationError(f"target database is not empty: {nonempty}")
                source_orphans = _orphans(source_connection)
                if any(source_orphans.values()):
                    raise DataMigrationError(f"source has foreign-key orphans: {source_orphans}")
                now = datetime.now(UTC).isoformat(timespec="milliseconds")
                copied: dict[str, int] = defaultdict(int)
                expired_reservations = 0
                for table in TABLES:
                    source_columns = set(source_connection.exec_driver_sql(
                        f"SELECT * FROM {table.name} LIMIT 0"
                    ).keys())
                    columns = [column for column in table.c if column.name in source_columns]
                    result = source_connection.execute(select(*columns))
                    while rows := result.fetchmany(batch_size):
                        payload = [dict(row._mapping) for row in rows]
                        if table is BUDGET_RESERVATIONS:
                            active = [row for row in payload if row["expires_at"] > now]
                            expired_reservations += len(payload) - len(active)
                            payload = active
                        if payload:
                            target_connection.execute(table.insert(), payload)
                            copied[table.name] += len(payload)
                _reset_sequences(target_connection)
                target_counts = _counts(target_connection)
                expected = dict(source_counts)
                expected["budget_reservations"] -= expired_reservations
                if target_counts != expected:
                    raise DataMigrationError(f"row-count reconciliation failed: expected {expected}, got {target_counts}")
                source_total = Decimal(str(source_connection.scalar(
                    select(func.coalesce(func.sum(REQUESTS.c.cost_usd), 0))) or 0))
                target_total = Decimal(str(target_connection.scalar(
                    select(func.coalesce(func.sum(REQUESTS.c.cost_usd), 0))) or 0))
                if abs(source_total - target_total) > COST_TOLERANCE:
                    raise DataMigrationError(
                        f"cost reconciliation failed: source={source_total}, target={target_total}"
                    )
                daily_mismatch = _daily_mismatch(_daily_totals(source_connection), _daily_totals(target_connection))
                if daily_mismatch:
                    raise DataMigrationError(f"daily cost reconciliation failed: {daily_mismatch}")
                target_orphans = _orphans(target_connection)
                if any(target_orphans.values()):
                    raise DataMigrationError(f"target has foreign-key orphans: {target_orphans}")
                report.update({
                    "source_counts": source_counts,
                    "target_counts": target_counts,
                    "expired_reservations_skipped": expired_reservations,
                    "cost_usd": float(target_total),
                    "daily_groups": len(_daily_totals(target_connection)),
                    "foreign_key_orphans": target_orphans,
                    "status": "ok",
                })
                report["tables"] = {name: {"source": source_counts[name], "copied": copied[name],
                                            "target": target_counts[name]} for name in source_counts}
        return report
    except BaseException:
        if backup.exists() and backup.stat().st_size == 0:
            backup.unlink()
        raise
    finally:
        target_engine.dispose()
        source_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--postgres-url", required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = migrate(args.sqlite, args.postgres_url, batch_size=args.batch_size)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
