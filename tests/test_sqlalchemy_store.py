"""Focused compatibility tests for the SQLAlchemy Core SQLite store."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text
from sqlalchemy.dialects import postgresql, sqlite

from gateway.monitoring import metrics
from gateway.monitoring.store import REQUESTS, CallRecord, RequestRecord, Store, iso

NOW = datetime(2026, 11, 19, 10, 45, 12, tzinfo=UTC)


def request(request_id: str, *, calls: list[CallRecord] | None = None) -> RequestRecord:
    return RequestRecord(
        id=request_id,
        ts=iso(NOW),
        app_id="test-app",
        status="ok",
        tier="cheap",
        model="small",
        prompt_tokens=5,
        completion_tokens=3,
        cost_usd=0.001,
        latency_ms=20,
        calls=calls or [CallRecord(1, "cheap", "small")],
    )


def test_sqlalchemy_store_configures_sqlite_and_supports_both_query_styles(tmp_path):
    store = Store(tmp_path / "gateway.db")
    try:
        assert store.engine.dialect.name == "sqlite"
        pragmas = store.query(
            "SELECT (SELECT * FROM pragma_journal_mode) AS journal_mode, "
            "(SELECT * FROM pragma_foreign_keys) AS foreign_keys, "
            "(SELECT timeout FROM pragma_busy_timeout) AS busy_timeout"
        )[0]
        assert pragmas == {"journal_mode": "wal", "foreign_keys": 1, "busy_timeout": 10_000}

        store.log_request(request("query-styles"))
        assert store.scalar("SELECT app_id FROM requests WHERE id = ?", ("query-styles",)) == "test-app"
        assert store.scalar(text("SELECT app_id FROM requests WHERE id = :request_id").bindparams(
            request_id="query-styles")) == "test-app"
        assert store.query("SELECT '?' AS literal, ? AS bound -- ?", (7,)) == [{"literal": "?", "bound": 7}]
        assert store.query('SELECT 1 AS "value?"') == [{"value?": 1}]
        with pytest.raises(sqlite3.ProgrammingError):
            store.query("SELECT ?")
    finally:
        store.close()


def test_log_request_rolls_back_calls_request_and_reservation_together(tmp_path):
    store = Store(tmp_path / "gateway.db")
    now = iso(NOW)
    store.reserve_budget("rollback", "test-app", now[:10], 1, 0.1, now, "2026-11-19T11:45:12.000+00:00")
    duplicate_calls = [CallRecord(1, "cheap", "small"), CallRecord(1, "strong", "large")]

    with pytest.raises(sqlite3.IntegrityError):
        store.log_request(request("rollback", calls=duplicate_calls))

    assert store.scalar("SELECT COUNT(*) FROM requests WHERE id = ?", ("rollback",)) == 0
    assert store.scalar("SELECT COUNT(*) FROM calls WHERE request_id = ?", ("rollback",)) == 0
    assert [row["request_id"] for row in store.reservations()] == ["rollback"]
    store.close()


def test_existing_database_is_readable_and_receives_ad_hoc_migration(tmp_path):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE requests (
            id TEXT PRIMARY KEY, ts TEXT NOT NULL, day TEXT NOT NULL, app_id TEXT NOT NULL,
            status TEXT NOT NULL, cost_usd REAL NOT NULL DEFAULT 0
        );
        INSERT INTO requests (id, ts, day, app_id, status, cost_usd)
        VALUES ('legacy', '2026-11-19T10:45:12.000+00:00', '2026-11-19', 'old-app', 'ok', 0.25);
        """
    )
    connection.close()

    store = Store(path)
    try:
        assert store.request_app("legacy") == "old-app"
        columns = {row["name"] for row in store.query("PRAGMA table_info(requests)")}
        assert "latency_class" in columns
        indexes = {row["name"] for row in store.query(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_autoindex%'"
        )}
        assert {"requests_ts", "requests_day_app", "feedback_request", "budget_reservations_app_day"} <= indexes
    finally:
        store.close()


def test_timestamp_bucketing_preserves_iso_utc_dashboard_values(tmp_path):
    store = Store(tmp_path / "gateway.db")
    store.log_request(request("bucket"))
    try:
        assert metrics.timeseries(store, bucket="day")[0]["bucket"] == "2026-11-19"
        assert metrics.timeseries(store, bucket="hour")[0]["bucket"] == "2026-11-19T10:00"
        with pytest.raises(ValueError, match="unsupported timestamp bucket"):
            metrics.timeseries(store, bucket="minute")
    finally:
        store.close()


def test_daily_bucketing_uses_the_stored_day(tmp_path):
    store = Store(tmp_path / "gateway.db")
    store.log_request(request("stored-day"))
    try:
        with store.engine.begin() as connection:
            connection.exec_driver_sql("UPDATE requests SET day = '2026-11-18' WHERE id = 'stored-day'")
        assert metrics.daily(store)[0]["day"] == "2026-11-18"
    finally:
        store.close()


def test_postgres_qmark_translation_ignores_quoted_and_commented_question_marks():
    sql = "SELECT '?', \"?\", $tag$?$tag$, ? -- ?\n/* ? */"
    assert Store._postgres_positional_sql(sql) == "SELECT '?', \"?\", $tag$?$tag$, %s -- ?\n/* ? */"


def test_hour_bucket_compiles_for_each_database():
    """PostgreSQL stores timestamptz, which has no substr(): its hour bucket formats the UTC hour instead
    (regression; the PostgreSQL integration tests check the values against a real server)."""
    statement = select(metrics.timestamp_bucket(REQUESTS, "hour"))
    on_postgres = str(statement.compile(dialect=postgresql.dialect()))
    assert "substr" not in on_postgres
    assert """to_char((requests.ts) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24":00"')""" in on_postgres
    assert "substr(requests.ts, 1, 13) || ':00'" in str(statement.compile(dialect=sqlite.dialect()))
