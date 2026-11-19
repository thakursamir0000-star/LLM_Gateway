from __future__ import annotations

import sqlite3

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect

from gateway.migrations import (
    MigrationError,
    alembic_config,
    current_revision,
    downgrade_database,
    head_revision,
    upgrade_database,
    verify_database_revision,
)
from gateway.monitoring.store import RequestRecord, Store


def url(path) -> str:
    return f"sqlite:///{path.as_posix()}"


def test_fresh_upgrade_and_repeated_execution(tmp_path):
    database_url = url(tmp_path / "fresh.db")
    assert upgrade_database(database_url) == head_revision(database_url)
    assert upgrade_database(database_url) == head_revision(database_url)
    verify_database_revision(database_url)
    engine = create_engine(database_url)
    try:
        assert set(inspect(engine).get_table_names()) >= {
            "requests", "calls", "feedback", "deployments", "alerts", "budget_reservations", "alembic_version"
        }
    finally:
        engine.dispose()


def test_existing_database_adoption_preserves_data(tmp_path):
    path = tmp_path / "existing.db"
    store = Store(path)
    store.log_request(RequestRecord(id="kept", ts="2026-11-19T00:00:00.000+00:00", app_id="a", status="ok"))
    store.close()
    database_url = url(path)

    with pytest.raises(MigrationError, match="--adopt-existing"):
        upgrade_database(database_url)
    assert upgrade_database(database_url, adopt_existing=True) == head_revision(database_url)
    reopened = Store(path)
    try:
        assert reopened.request_app("kept") == "a"
    finally:
        reopened.close()


def test_previous_core_schema_upgrades_to_operational_state(tmp_path):
    database_url = url(tmp_path / "previous.db")
    command.upgrade(alembic_config(database_url), "0001_core_schema")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO requests (id, ts, day, app_id, status) "
                "VALUES ('kept', '2026-11-19T00:00:00+00:00', '2026-11-19', 'a', 'ok')"
            )
    finally:
        engine.dispose()
    upgrade_database(database_url)
    store = Store(tmp_path / "previous.db")
    try:
        assert store.request_app("kept") == "a"
        assert "latency_class" in {row["name"] for row in store.query("PRAGMA table_info(requests)")}
    finally:
        store.close()


def test_invalid_existing_schema_is_not_stamped(tmp_path):
    path = tmp_path / "invalid.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE requests (id TEXT PRIMARY KEY)")
    connection.close()
    database_url = url(path)
    with pytest.raises(MigrationError, match="missing tables"):
        upgrade_database(database_url, adopt_existing=True)
    assert current_revision(database_url) is None


def test_destructive_downgrade_requires_confirmation(tmp_path):
    database_url = url(tmp_path / "downgrade.db")
    upgrade_database(database_url)
    with pytest.raises(MigrationError, match="confirm"):
        downgrade_database(database_url, "0001_core_schema")
    assert downgrade_database(database_url, "0001_core_schema", confirm_destructive=True) == "0001_core_schema"
    assert upgrade_database(database_url) == head_revision(database_url)
