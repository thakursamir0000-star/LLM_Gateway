from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gateway.migrations import upgrade_database
from gateway.monitoring.store import CallRecord, RequestRecord, Store, iso
from scripts.migrate_sqlite_to_postgres import DataMigrationError, migrate


def sqlite_url(path) -> str:
    return f"sqlite:///{path.as_posix()}"


def source_database(path) -> None:
    store = Store(path)
    now = datetime.now(UTC)
    store.log_request(RequestRecord(
        id="request-1", ts=iso(now), app_id="app", status="ok", cost_usd=0.125,
        calls=[CallRecord(1, "cheap", "small", cost_usd=0.125)],
    ))
    store.add_feedback("request-1", "app", 1, "good", iso(now))
    store.add_deployment("activate", "v1", None, 0, "", iso(now))
    store.add_alert("daily_cost", 1, 0.5, "high", iso(now), "daily_cost:test")
    store.reserve_budget("active", "app", iso(now)[:10], 1, 0.1, iso(now), iso(now + timedelta(minutes=5)))
    store.reserve_budget("expired", "app", iso(now)[:10], 1, 0.1,
                         iso(now - timedelta(minutes=10)), iso(now - timedelta(minutes=5)))
    store.close()


def test_migration_copies_and_reconciles_without_modifying_source(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    source_database(source)
    source_size = source.stat().st_size
    target_url = sqlite_url(target)
    upgrade_database(target_url)

    report = migrate(source, target_url, batch_size=1)

    assert report["status"] == "ok" and report["cost_usd"] == pytest.approx(0.125)
    assert report["expired_reservations_skipped"] == 1
    assert source.stat().st_size == source_size
    assert Path(report["backup"]).exists()
    target_store = Store(target)
    try:
        assert target_store.request_app("request-1") == "app"
        assert [row["request_id"] for row in target_store.reservations()] == ["active"]
    finally:
        target_store.close()


def test_migration_refuses_nonempty_target(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    source_database(source)
    target_url = sqlite_url(target)
    upgrade_database(target_url)
    target_store = Store(target)
    target_store.log_request(RequestRecord(id="existing", ts="2026-11-19T00:00:00+00:00",
                                           app_id="app", status="ok"))
    target_store.close()
    with pytest.raises(DataMigrationError, match="not empty"):
        migrate(source, target_url)


def test_migration_cli_interface_with_flags(tmp_path, monkeypatch, capsys):
    from scripts.migrate_sqlite_to_postgres import main

    source = tmp_path / "cli_source.db"
    target = tmp_path / "cli_target.db"
    source_database(source)
    target_url = sqlite_url(target)
    upgrade_database(target_url)

    report_file = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate_sqlite_to_postgres.py",
            "--sqlite",
            str(source),
            "--postgres-url",
            target_url,
            "--report",
            str(report_file),
        ],
    )
    main()
    captured = capsys.readouterr()
    assert '"status": "ok"' in captured.out
    assert report_file.exists()
