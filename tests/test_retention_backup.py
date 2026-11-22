"""Request-log retention (python -m gateway.manage retention) and SQLite backup/restore (scripts/db_backup.py).

The PostgreSQL backup drill runs against the Compose database: python scripts/db_backup.py drill.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

import scripts.db_backup as db_backup
from gateway.manage.__main__ import main as manage
from gateway.migrations import sqlite_url, upgrade_database
from gateway.monitoring.store import CallRecord, RequestRecord, Store, iso, utc_now


def populate(path):
    upgrade_database(sqlite_url(path))
    store = Store(path)
    now = utc_now()
    for i, age in enumerate((0, 1, 10, 100, 200)):
        ts = iso(now - timedelta(days=age))
        store.log_request(RequestRecord(id=f"r{i}", ts=ts, app_id="web", status="ok", cost_usd=0.01,
                                        calls=[CallRecord(1, "cheap", "small")]))
        store.add_feedback(f"r{i}", "web", 1, None, ts)
        store.add_alert("error_rate", 0.5, 0.2, f"alert {i}", ts)
    store.add_deployment("activate", "v1", None, 0.0, "old release change", iso(now - timedelta(days=300)))
    store.reserve_budget("live", "web", now.date().isoformat(), 1.0, 0.01, iso(now), iso(now + timedelta(minutes=5)))
    store.reserve_budget("stale", "web", now.date().isoformat(), 1.0, 0.01, iso(now - timedelta(hours=2)),
                         iso(now - timedelta(hours=1)))
    store.close()
    return path


def counts(path):
    store = Store(path)
    try:
        return {t: store.scalar(f"SELECT count(*) FROM {t}") for t in db_backup.TABLES}
    finally:
        store.close()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = populate(tmp_path / "gateway.db")
    monkeypatch.delenv("GATEWAY_DATABASE_URL", raising=False)
    monkeypatch.setenv("GATEWAY_DB", str(path))
    return path


def test_retention_dry_run_counts_and_deletes_nothing(db, capsys):
    assert manage(["retention", "--days", "30", "--dry-run"]) == 0
    assert "would delete" in capsys.readouterr().out
    assert counts(db)["requests"] == 5


def test_retention_needs_yes_and_keeps_two_days(db, capsys):
    assert manage(["retention", "--days", "30"]) == 1
    assert manage(["retention", "--days", "1", "--yes"]) == 1
    assert counts(db)["requests"] == 5


def test_retention_deletes_old_rows_keeps_recent_and_deployments(db, capsys):
    store = Store(db)
    today_spend = store.app_spend("web", utc_now().date().isoformat())
    store.close()
    assert manage(["retention", "--days", "30", "--yes"]) == 0
    after = counts(db)
    assert after["requests"] == after["calls"] == after["feedback"] == after["alerts"] == 3  # ages 0, 1, 10
    assert after["deployments"] == 1  # the release audit trail is kept
    assert after["budget_reservations"] == 1  # only the expired reservation went
    store = Store(db)
    assert store.app_spend("web", utc_now().date().isoformat()) == today_spend
    store.close()


def test_sqlite_backup_drill_and_restore(db, tmp_path, capsys):
    out = tmp_path / "backups"
    assert db_backup.main(["--out", str(out), "drill"]) == 0
    assert "drill passed" in capsys.readouterr().out
    archive = next(out.glob("gateway-*.db"))
    assert counts(archive) == counts(db)

    restored = tmp_path / "restored.db"
    with pytest.raises(SystemExit):
        db_backup.main(["--out", str(out), "restore", str(archive), "--target", str(restored)])  # no --yes
    with pytest.raises(SystemExit):
        db_backup.main(["--out", str(out), "restore", str(archive), "--target", str(db), "--yes"])  # live db
    assert db_backup.main(["--out", str(out), "restore", str(archive), "--target", str(restored), "--yes"]) == 0
    assert counts(restored) == counts(db)
