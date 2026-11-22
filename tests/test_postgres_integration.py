"""PostgreSQL integration tests (Phase 4.2): the PostgreSQL-specific code paths, against a real server.

Configuration: ``TEST_POSTGRES_URL`` names a dedicated test server and database, for example
``postgresql+psycopg://gateway:<password>@localhost:5432/gateway_test``. Its database name must contain
"test". Every test creates its own uniquely named database on that server (``gwtest_<random>``) and drops
it afterwards, so tests are independent of each other and of their order, and the configured database
itself is never written to. ``GATEWAY_DATABASE_URL`` is never used here.

Without ``TEST_POSTGRES_URL`` the tests are skipped locally. In GitHub Actions they fail instead, and they
fail anywhere when the server is configured but unreachable: they never fall back to SQLite.

Areas: 1 Alembic lifecycle, 2 store initialisation, 3 money and timestamps, 4 request logging and
metrics, 5 concurrent budget reservations, 6 atomic settlement, 7 SQLite-to-PostgreSQL migration,
8 releases and alerts.
"""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest
from alembic import command
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.pool import NullPool

from gateway.app import create_app
from gateway.manage.budgets import reservation_estimate
from gateway.manage.releases import ReleaseError, ReleaseManager
from gateway.migrations import (
    MigrationError,
    alembic_config,
    current_revision,
    downgrade_database,
    head_revision,
    upgrade_database,
    verify_database_revision,
)
from gateway.monitoring import metrics
from gateway.monitoring.alerts import AlertMonitor
from gateway.monitoring.store import METADATA, REQUESTS, CallRecord, RequestRecord, Store, iso, utc_now
from gateway.providers.base import Completion
from gateway.providers.fake import FakeProvider
from scripts.migrate_sqlite_to_postgres import DataMigrationError, migrate
from tests.conftest import make_config

RAW_URL = os.environ.get("TEST_POSTGRES_URL", "").strip()
IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
MSG = [{"role": "user", "content": "hi"}]
HEAD_TABLES = {"requests", "calls", "feedback", "deployments", "alerts", "budget_reservations", "alembic_version"}


@dataclass(frozen=True)
class PostgresServer:
    url: URL  # the configured test database; only used to create and drop per-test databases
    admin: Any  # AUTOCOMMIT engine for CREATE/DROP DATABASE
    version_num: int


@pytest.fixture(scope="session")
def pg_server():
    if not RAW_URL:
        reason = "TEST_POSTGRES_URL is not set; the PostgreSQL integration tests need a dedicated test server"
        if IN_GITHUB_ACTIONS:
            pytest.fail(reason + " (required in GitHub Actions)", pytrace=False)
        pytest.skip(reason)
    url = make_url(RAW_URL)
    if url.get_backend_name() != "postgresql":
        pytest.fail(f"TEST_POSTGRES_URL must be a PostgreSQL URL, not {url.get_backend_name()!r}", pytrace=False)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    if "test" not in (url.database or "").lower():
        pytest.fail("refusing TEST_POSTGRES_URL: its database name must contain 'test'", pytrace=False)
    admin = create_engine(url, isolation_level="AUTOCOMMIT", poolclass=NullPool,
                          connect_args={"connect_timeout": 5})
    try:
        with admin.connect() as connection:
            version_num = int(connection.scalar(text("SHOW server_version_num")))
    except OperationalError as error:
        admin.dispose()
        pytest.fail(f"PostgreSQL is configured ({url.render_as_string(hide_password=True)}) but unreachable: "
                    f"{error.orig}", pytrace=False)
    yield PostgresServer(url, admin, version_num)
    admin.dispose()


@pytest.fixture
def pg_url(pg_server):
    """A fresh, empty database for one test, dropped afterwards."""
    name = f"gwtest_{uuid.uuid4().hex[:16]}"
    with pg_server.admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield pg_server.url.set(database=name).render_as_string(hide_password=False)
    finally:
        with pg_server.admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


@pytest.fixture
def migrated(pg_url):
    upgrade_database(pg_url)
    return pg_url


@pytest.fixture
def store(migrated):
    store = Store(migrated)
    yield store
    store.close()


def rows(url: str, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Raw read through a separate connection, bypassing the Store's type conversions."""
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            return [dict(row._mapping) for row in connection.execute(text(sql), params or {})]
    finally:
        engine.dispose()


def schema(url: str) -> dict[str, Any]:
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            tables = set(inspector.get_table_names())
            return {
                "tables": tables,
                "pk": {t: tuple(inspector.get_pk_constraint(t)["constrained_columns"]) for t in tables},
                "indexes": {i["name"]: (t, tuple(i["column_names"]), bool(i["unique"]))
                            for t in tables for i in inspector.get_indexes(t)},
                "fks": {fk["name"]: (t, fk["referred_table"], tuple(fk["constrained_columns"]),
                                     (fk.get("options") or {}).get("ondelete"))
                        for t in tables for fk in inspector.get_foreign_keys(t)},
                "columns": {t: {c["name"]: c for c in inspector.get_columns(t)} for t in tables},
            }
    finally:
        engine.dispose()


def request(id: str, ts: str, **fields: Any) -> RequestRecord:
    return RequestRecord(id=id, ts=ts, app_id=fields.pop("app_id", "app"), status=fields.pop("status", "ok"),
                         **fields)


def assert_head_schema(url: str) -> None:
    found = schema(url)
    assert HEAD_TABLES <= found["tables"]
    assert found["pk"]["requests"] == ("id",)
    assert found["pk"]["calls"] == ("request_id", "seq")
    assert found["pk"]["budget_reservations"] == ("request_id",)
    for table in ("feedback", "deployments", "alerts"):
        assert found["pk"][table] == ("id",)
    assert found["indexes"]["requests_ts"] == ("requests", ("ts",), False)
    assert found["indexes"]["requests_day_app"] == ("requests", ("day", "app_id"), False)
    assert found["indexes"]["feedback_request"] == ("feedback", ("request_id",), False)
    assert found["indexes"]["alerts_dedup_key"] == ("alerts", ("dedup_key",), True)
    assert found["indexes"]["budget_reservations_app_day"] == ("budget_reservations", ("app_id", "day"), False)
    assert found["fks"]["calls_request_id_fkey"] == ("calls", "requests", ("request_id",), "CASCADE")
    assert found["fks"]["feedback_request_id_fkey"] == ("feedback", "requests", ("request_id",), "CASCADE")
    columns = found["columns"]
    for table, column in (("requests", "cost_usd"), ("calls", "cost_usd"), ("budget_reservations", "amount_usd")):
        kind = columns[table][column]["type"]
        assert (type(kind).__name__, kind.precision, kind.scale) == ("NUMERIC", 20, 12)
    assert columns["requests"]["ts"]["type"].timezone is True
    assert type(columns["requests"]["day"]["type"]).__name__ == "DATE"


# ── 1. Alembic lifecycle ────────────────────────────────────────────


def test_server_is_postgresql_16_or_newer(pg_server):
    assert pg_server.version_num >= 160000, f"PostgreSQL {pg_server.version_num} is older than CI's 16"


def test_upgrade_builds_the_full_schema_and_is_repeatable(pg_url):
    head = head_revision(pg_url)
    assert current_revision(pg_url) is None
    assert upgrade_database(pg_url) == head
    assert upgrade_database(pg_url) == head  # repeating is a no-op
    verify_database_revision(pg_url)
    assert_head_schema(pg_url)


def test_downgrade_needs_confirmation_then_reupgrade_restores_the_schema(pg_url):
    upgrade_database(pg_url)
    with pytest.raises(MigrationError, match="confirm-destructive"):
        downgrade_database(pg_url, "0002_operational_state")

    assert downgrade_database(pg_url, "0002_operational_state", confirm_destructive=True) == "0002_operational_state"
    found = schema(pg_url)
    assert "dedup_key" not in found["columns"]["alerts"] and found["fks"] == {}
    with pytest.raises(MigrationError, match="expected"):
        verify_database_revision(pg_url)

    assert downgrade_database(pg_url, "base", confirm_destructive=True) is None
    assert schema(pg_url)["tables"] <= {"alembic_version"}

    assert upgrade_database(pg_url) == head_revision(pg_url)
    verify_database_revision(pg_url)
    assert_head_schema(pg_url)


def test_pre_alembic_database_is_adopted_only_on_request(pg_url):
    engine = create_engine(pg_url, poolclass=NullPool)
    try:
        METADATA.create_all(engine)  # a database created by an older gateway, without Alembic
    finally:
        engine.dispose()
    with pytest.raises(MigrationError, match="adopt-existing"):
        upgrade_database(pg_url)
    assert upgrade_database(pg_url, adopt_existing=True) == head_revision(pg_url)
    assert_head_schema(pg_url)


# ── 2. Store initialisation ─────────────────────────────────────────


def test_store_refuses_an_unmigrated_database(pg_url):
    with pytest.raises(MigrationError, match="unversioned"):
        Store(pg_url)


def test_store_refuses_a_partially_migrated_database(pg_url):
    command.upgrade(alembic_config(pg_url), "0002_operational_state")
    with pytest.raises(MigrationError, match="0002_operational_state"):
        Store(pg_url)


def test_gateway_runs_on_a_migrated_postgres_database(migrated, monkeypatch):
    monkeypatch.setenv("GATEWAY_DATABASE_URL", migrated)
    monkeypatch.delenv("GATEWAY_STATE_BACKEND", raising=False)
    groq = FakeProvider("groq", lambda model, messages, params: Completion("ok", "stop", 10, 5, model))
    app = create_app(make_config(), providers={"groq": groq, "gemini": FakeProvider("gemini")})
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200 and ready.json()["database"]["ok"] is True
        answer = client.post("/v1/chat/completions", headers={"X-App-Id": "app"},
                             json={"model": "cheap", "messages": MSG})
        assert answer.status_code == 200
    [row] = rows(migrated, "SELECT app_id, status, cost_usd, pg_typeof(ts)::text AS ts_type FROM requests")
    assert (row["app_id"], row["status"], row["ts_type"]) == ("app", "ok", "timestamp with time zone")
    assert isinstance(row["cost_usd"], Decimal) and row["cost_usd"] > 0
    assert rows(migrated, "SELECT count(*) AS n FROM budget_reservations")[0]["n"] == 0  # settled


# ── 3. Money and timestamps ─────────────────────────────────────────


def test_money_is_exact_numeric(store, migrated):
    day = "2026-11-22"
    for i in range(10):
        store.log_request(request(f"tenth-{i}", f"{day}T01:00:{i:02d}.000+00:00", cost_usd=0.1))
    store.log_request(request("tiny", f"{day}T02:00:00.000+00:00", app_id="other", cost_usd=0.000000123456))
    store.log_request(request("rounded", f"{day}T03:00:00.000+00:00", app_id="other",
                              cost_usd=0.0000123456789012345))
    store.log_request(request("p1", f"{day}T04:00:00.000+00:00", app_id="pair", cost_usd=0.1))
    store.log_request(request("p2", f"{day}T04:00:01.000+00:00", app_id="pair", cost_usd=0.2))

    assert 0.1 + 0.2 != 0.3  # float addition rounds...
    assert store.app_spend("pair", day) == 0.3  # ...NUMERIC sums exactly
    assert store.app_spend("app", day) == 1.0
    [total] = rows(migrated, "SELECT sum(cost_usd) AS s FROM requests WHERE app_id = 'app'")
    assert total["s"] == Decimal("1.000000000000")
    stored = {r["id"]: r["cost_usd"] for r in rows(migrated, "SELECT id, cost_usd FROM requests")}
    assert stored["tiny"] == Decimal("0.000000123456")
    assert stored["rounded"] == Decimal("0.000012345679")  # NUMERIC(20, 12) rounds past 12 decimals
    assert store.query(select(REQUESTS.c.cost_usd).where(REQUESTS.c.id == "tiny"))[0]["cost_usd"] == 0.000000123456


def test_timestamps_are_stored_as_utc_whatever_the_session_timezone(migrated):
    kolkata = make_url(migrated).update_query_dict({"options": "-c TimeZone=Asia/Kolkata"})
    store = Store(kolkata.render_as_string(hide_password=False))
    try:
        assert store.scalar("SELECT current_setting('TimeZone')") == "Asia/Kolkata"
        store.log_request(request("offset", "2026-11-22T10:15:30.123+05:30"))
        store.log_request(request("naive", "2026-11-22T06:00:00"))  # no offset: taken as UTC
        read = {r["id"]: r["ts"] for r in store.query(select(REQUESTS.c.id, REQUESTS.c.ts))}
        assert read == {"offset": "2026-11-22T04:45:30.123+00:00", "naive": "2026-11-22T06:00:00.000+00:00"}
        utc = rows(migrated, "SELECT id, to_char(ts AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.MS') AS t, "
                             "day::text AS day FROM requests ORDER BY id")
        assert utc == [{"id": "naive", "t": "2026-11-22 06:00:00.000", "day": "2026-11-22"},
                       {"id": "offset", "t": "2026-11-22 04:45:30.123", "day": "2026-11-22"}]
        # Range filters compare instants, whatever offset the bound is written in.
        assert metrics.summary(store, since="2026-11-22T10:00:00+05:30")["requests"] == 2  # 04:30 UTC
        assert metrics.summary(store, since="2026-11-22T11:00:00+05:30")["requests"] == 1  # 05:30 UTC
    finally:
        store.close()


# ── 4. Request logging and metrics ──────────────────────────────────


def seed_traffic(store: Store) -> None:
    store.log_request(request(
        "r1", "2026-11-20T10:05:00.000+00:00", release="v1", tier="strong", model="large", latency_ms=100.0,
        router_ms=1.0, prompt_tokens=10, completion_tokens=5, cost_usd=0.25, latency_class="interactive",
        calls=[CallRecord(1, "strong", "large", "openai/gpt-oss-120b", 10, 5, 0.25, 100.0)]))
    store.log_request(request(
        "r2", "2026-11-20T10:40:00.000+00:00", release="v1", tier="cheap", model="small", latency_ms=300.0,
        prompt_tokens=20, completion_tokens=10, cost_usd=0.125, latency_class="batch", escalated=True,
        fallback=True, calls=[CallRecord(1, "cheap", "small", status="error", error_type="overloaded"),
                              CallRecord(2, "cheap", "small", "openai/gpt-oss-20b", 20, 10, 0.125, 300.0)]))
    store.log_request(request("r3", "2026-11-20T11:10:00.000+00:00", release="v1", tier="cheap", model="small",
                              status="error", latency_ms=50.0, latency_class="interactive"))
    store.log_request(request("r4", "2026-11-21T09:00:00.000+00:00", app_id="b", status="rejected"))
    store.log_request(request(
        "r5", "2026-11-21T09:30:00.000+00:00", app_id="b", release="v2", canary=True, tier="strong",
        model="large", latency_ms=200.0, prompt_tokens=30, completion_tokens=20, cost_usd=0.5,
        latency_class="interactive"))
    store.add_feedback("r1", "app", 1, None, "2026-11-20T10:06:00.000+00:00")
    store.add_feedback("r2", "app", -1, "slow", "2026-11-20T10:41:00.000+00:00")
    store.add_feedback("r5", "b", 1, None, "2026-11-21T09:31:00.000+00:00")


def test_requests_and_calls_are_logged_with_their_constraints(store, migrated):
    seed_traffic(store)
    calls = store.query("SELECT request_id, seq, status FROM calls ORDER BY request_id, seq")
    assert [(c["request_id"], c["seq"], c["status"]) for c in calls] == [
        ("r1", 1, "ok"), ("r2", 1, "error"), ("r2", 2, "ok")]
    with pytest.raises(IntegrityError):  # PostgreSQL enforces calls.request_id -> requests.id
        store.query("INSERT INTO calls (request_id, seq, ts, status) VALUES (?, ?, ?, ?)",
                    ("missing", 1, "2026-11-20T00:00:00+00:00", "ok"))
    engine = create_engine(migrated, poolclass=NullPool)
    try:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM requests WHERE id = 'r2'"))
    finally:
        engine.dispose()
    assert [c["request_id"] for c in store.query("SELECT request_id FROM calls")] == ["r1"]  # ON DELETE CASCADE
    assert store.request_app("r1") == "app" and store.request_app("r2") is None


def test_raw_queries_with_qmark_parameters_run_on_postgres(store):
    seed_traffic(store)
    found = store.query("SELECT id, '?' AS mark FROM requests WHERE app_id = ? AND status = ? ORDER BY id",
                        ("app", "ok"))
    assert found == [{"id": "r1", "mark": "?"}, {"id": "r2", "mark": "?"}]
    assert store.scalar("SELECT count(*) FROM requests WHERE day = ?", ("2026-11-21",)) == 2


def test_metrics_summary(store):
    seed_traffic(store)
    everything = metrics.summary(store)
    assert {k: everything[k] for k in ("requests", "ok", "errors", "rejected", "tokens", "escalated",
                                       "fallbacks", "strong", "feedback_up", "feedback_down")} == {
        "requests": 5, "ok": 3, "errors": 1, "rejected": 1, "tokens": 95, "escalated": 1, "fallbacks": 1,
        "strong": 2, "feedback_up": 2, "feedback_down": 1}
    assert everything["cost_usd"] == pytest.approx(0.875)
    assert everything["error_rate"] == pytest.approx(0.25) and everything["strong_share"] == pytest.approx(2 / 3)
    assert (everything["p50_ms"], everything["p95_ms"], everything["router_p50_ms"]) == (200.0, 290.0, 1.0)
    second_day = metrics.summary(store, since="2026-11-21T00:00:00+00:00")
    assert (second_day["requests"], second_day["ok"], second_day["rejected"]) == (2, 1, 1)
    assert second_day["cost_usd"] == pytest.approx(0.5)
    assert metrics.summary(store, app_id="app")["requests"] == 3


def test_metrics_timeseries_daily_and_hourly(store):
    seed_traffic(store)
    days = metrics.timeseries(store, bucket="day")
    assert [(d["bucket"], d["requests"], d["ok"], d["errors"], d["rejected"], d["tokens"]) for d in days] == [
        ("2026-11-20", 3, 2, 1, 0, 45), ("2026-11-21", 2, 1, 0, 1, 50)]
    assert [d["cost_usd"] for d in days] == pytest.approx([0.375, 0.5])
    assert days[0]["p50_ms"] == 200.0 and days[0]["error_rate"] == pytest.approx(1 / 3)
    hours = metrics.timeseries(store, bucket="hour")
    assert [(h["bucket"], h["requests"]) for h in hours] == [
        ("2026-11-20T10:00", 2), ("2026-11-20T11:00", 1), ("2026-11-21T09:00", 2)]
    assert [d["day"] for d in metrics.daily(store)] == ["2026-11-20", "2026-11-21"]


def test_metrics_breakdowns(store):
    seed_traffic(store)
    [spend] = metrics.app_spend(store, "2026-11-20")
    assert (spend["app_id"], spend["requests"]) == ("app", 3) and spend["cost_usd"] == pytest.approx(0.375)
    shares = {(m["model"], m["tier"]): (m["requests"], m["share"]) for m in metrics.model_share(store)}
    assert shares == {("large", "strong"): (2, pytest.approx(2 / 3)), ("small", "cheap"): (1, pytest.approx(1 / 3))}
    releases = {(r["release"], r["canary"]): r for r in metrics.release_stats(store)}
    assert set(releases) == {("v1", 0), ("v2", 1), (None, 0)}
    assert (releases[("v1", 0)]["requests"], releases[("v1", 0)]["errors"], releases[("v1", 0)]["feedback_up"],
            releases[("v1", 0)]["feedback_down"]) == (3, 1, 1, 1)
    assert releases[("v2", 1)]["thumbs_up_rate"] == 1.0
    classes = {c["latency_class"]: c for c in metrics.latency_class_stats(store)}
    assert {k: (v["requests"], v["ok"], v["errors"]) for k, v in classes.items()} == {
        "batch": (1, 1, 0), "interactive": (3, 2, 1), "not recorded": (1, 0, 0)}
    assert classes["interactive"]["p50_ms"] == 150.0
    assert [r["id"] for r in metrics.recent(store, limit=2)] == ["r5", "r4"]
    assert metrics.app_ids(store) == ["app", "b"]


# ── 5. Concurrent budget reservations ───────────────────────────────


def test_concurrent_reservations_from_independent_connections_fit_the_budget(migrated):
    stores = [Store(migrated), Store(migrated)]  # two gateway instances, separate connection pools
    try:
        now = datetime.now(UTC)
        day, limit, estimate = iso(now)[:10], 0.001, 0.00013
        barrier = threading.Barrier(24)
        results = []

        def reserve(i: int) -> None:
            barrier.wait()
            results.append(stores[i % 2].reserve_budget(f"r{i}", "app", day, limit, estimate, iso(now),
                                                        iso(now + timedelta(minutes=5))))

        threads = [threading.Thread(target=reserve, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        admitted = [r for r in results if r.admitted]
        assert len(admitted) == math.ceil(limit / estimate)  # 7 full reservations and one for the rest
        [held] = rows(migrated, "SELECT sum(amount_usd) AS s, count(*) AS n FROM budget_reservations")
        assert held["n"] == len(admitted) and held["s"] == Decimal("0.001000000000")
    finally:
        for s in stores:
            s.close()


def test_a_settlement_between_the_reservation_reads_cannot_hide_spend(migrated):
    """READ COMMITTED: a settlement that commits while a reservation is being computed must not make the
    same money disappear from both the logged spend and the active reservations."""
    reserver, settler = Store(migrated), Store(migrated)
    try:
        now = datetime.now(UTC)
        day, later = iso(now)[:10], iso(now + timedelta(minutes=5))
        assert settler.reserve_budget("first", "app", day, 1.0, 0.6, iso(now), later).amount == pytest.approx(0.6)
        settled = []

        def settle_first(conn, cursor, statement, parameters, context, executemany):
            if not settled and "requests.cost_usd" in statement:  # right after the spend has been read
                settled.append(True)
                settler.log_request(request("first", iso(now), cost_usd=0.6))

        event.listen(reserver.engine, "after_cursor_execute", settle_first)
        try:
            second = reserver.reserve_budget("second", "app", day, 1.0, 0.5, iso(now), later)
        finally:
            event.remove(reserver.engine, "after_cursor_execute", settle_first)
        assert settled
        held = sum(r["amount_usd"] for r in reserver.reservations("app"))
        assert reserver.app_spend("app", day) + held <= 1.0 + 1e-12
        assert second.admitted and second.amount == pytest.approx(0.4)
    finally:
        reserver.close()
        settler.close()


async def test_two_gateway_instances_share_one_postgres_budget(migrated):
    limit = 0.0005
    config = make_config(budgets={"apps": {"tight": {"daily_usd": limit}}})
    stores = [Store(migrated), Store(migrated)]

    def answer(model, messages, params):
        return Completion("ok", "stop", 10, 50, model)  # $0.000039 on the strong model

    async def no_sleep(_):
        return None

    apps = [create_app(config, providers={"groq": FakeProvider("groq", answer, latency_s=0.2),
                                          "gemini": FakeProvider("gemini")}, store=s, sleep=no_sleep)
            for s in stores]
    estimate = reservation_estimate([config.models["large"], config.models["small"]], MSG, {"max_tokens": 100},
                                    config.budgets.reserve_output_tokens)
    try:
        clients = [httpx.AsyncClient(transport=httpx.ASGITransport(app=a), base_url="http://t") for a in apps]
        responses = await asyncio.gather(*[
            clients[i % 2].post("/v1/chat/completions", headers={"X-App-Id": "tight"},
                                json={"model": "strong", "messages": MSG, "max_tokens": 100}) for i in range(12)])
        for client in clients:
            await client.aclose()
        assert sum(r.status_code == 200 for r in responses) == math.ceil(limit / estimate)
        assert all(r.json()["error"]["code"] == "budget_exceeded" for r in responses if r.status_code != 200)
        assert stores[0].app_spend("tight", iso(datetime.now(UTC))[:10]) <= limit
        assert stores[0].reservations() == []
        assert rows(migrated, "SELECT count(*) AS n FROM requests")[0]["n"] == 12
    finally:
        for s in stores:
            s.close()


# ── 6. Atomic settlement ────────────────────────────────────────────


def reserve(store: Store, request_id: str, amount: float = 0.2) -> None:
    now = datetime.now(UTC)
    assert store.reserve_budget(request_id, "app", iso(now)[:10], 1.0, amount, iso(now),
                                iso(now + timedelta(minutes=5))).admitted


def test_settlement_logs_request_and_calls_and_releases_the_reservation_together(store):
    reserve(store, "req")
    store.log_request(request("req", iso(datetime.now(UTC)), cost_usd=0.05,
                              calls=[CallRecord(1, "cheap", "small", cost_usd=0.05)]))
    assert store.reservations() == []
    assert store.scalar("SELECT count(*) FROM calls WHERE request_id = ?", ("req",)) == 1
    assert store.app_spend("app", iso(datetime.now(UTC))[:10]) == pytest.approx(0.05)


def test_a_failed_settlement_rolls_everything_back(store):
    reserve(store, "req")
    broken = request("req", iso(datetime.now(UTC)), cost_usd=0.05,
                     calls=[CallRecord(1, "cheap", "small"), CallRecord(1, "cheap", "small")])  # duplicate seq
    with pytest.raises(IntegrityError):
        store.log_request(broken)
    assert store.request_app("req") is None  # no request row
    assert store.scalar("SELECT count(*) FROM calls") == 0
    assert [r["request_id"] for r in store.reservations()] == ["req"]  # still held


def test_a_request_cannot_be_settled_twice(store):
    reserve(store, "req")
    record = request("req", iso(datetime.now(UTC)), cost_usd=0.05, calls=[CallRecord(1, "cheap", "small")])
    store.log_request(record)
    with pytest.raises(IntegrityError):
        store.log_request(record)
    assert store.scalar("SELECT count(*) FROM requests") == 1 and store.scalar("SELECT count(*) FROM calls") == 1
    assert store.app_spend("app", iso(datetime.now(UTC))[:10]) == pytest.approx(0.05)
    reserve(store, "next")  # the failed attempt left no transaction or lock behind


# ── 7. SQLite to PostgreSQL migration ───────────────────────────────


def sqlite_source(path) -> None:
    store = Store(path)
    now = datetime.now(UTC)
    today, yesterday = iso(now), iso(now - timedelta(days=1))
    # Costs as the gateway computes them (tokens x list price): their float sums are not exact.
    for i, (ts, app, cost) in enumerate([(yesterday, "a", 0.1), (yesterday, "a", 0.2), (yesterday, "b", 0.00038955),
                                         (today, "a", 37 * 0.15e-6 + 512 * 0.75e-6), (today, "a", 0.3),
                                         (today, "b", 0.000123456789)]):
        store.log_request(request(f"q{i}", ts, app_id=app, cost_usd=cost,
                                  calls=[CallRecord(1, "cheap", "small", cost_usd=cost)]))
    for i in range(3):
        store.add_feedback("q0", "a", 1 if i % 2 == 0 else -1, None, today)
    store.add_deployment("activate", "v2", None, 0.0, "", yesterday)
    store.add_deployment("canary", "v2", "v1", 0.1, "", today)
    store.add_alert("daily_cost", 2.0, 1.0, "over", today, "daily_cost:x")
    store.add_alert("error_rate", 0.5, 0.2, "errors", today, "error_rate:after:initial")
    store.reserve_budget("active", "a", today[:10], 10.0, 0.1, today, iso(now + timedelta(minutes=5)))
    store.reserve_budget("expired", "a", today[:10], 10.0, 0.1, iso(now - timedelta(minutes=10)),
                         iso(now - timedelta(minutes=5)))
    store.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM feedback WHERE id = 2")  # a gap in the ids
        connection.commit()
    finally:
        connection.close()


def test_sqlite_data_migrates_reconciles_and_keeps_ids_and_sequences(tmp_path, migrated):
    source = tmp_path / "gateway.db"
    sqlite_source(source)
    report = migrate(source, migrated, batch_size=2)
    assert report["status"] == "ok" and report["expired_reservations_skipped"] == 1
    assert report["target_counts"] == {"requests": 6, "calls": 6, "feedback": 2, "deployments": 2, "alerts": 2,
                                       "budget_reservations": 1}

    origin, target = Store(source), Store(migrated)
    try:
        wanted = select(REQUESTS.c.id, REQUESTS.c.ts, REQUESTS.c.day, REQUESTS.c.app_id, REQUESTS.c.cost_usd)
        before, after = origin.query(wanted.order_by(REQUESTS.c.id)), target.query(wanted.order_by(REQUESTS.c.id))
        assert [(r["id"], r["ts"], r["day"], r["app_id"]) for r in after] == [
            (r["id"], r["ts"], r["day"], r["app_id"]) for r in before]
        assert [r["cost_usd"] for r in after] == pytest.approx([r["cost_usd"] for r in before], abs=1e-12)
        assert [r["id"] for r in target.query("SELECT id FROM feedback ORDER BY id")] == [1, 3]
        assert [r["id"] for r in target.deployments()] == [2, 1]
        assert [r["request_id"] for r in target.reservations()] == ["active"]
        # Sequences continue after the migrated ids instead of colliding with them.
        assert target.add_feedback("q0", "a", 1, None, iso(datetime.now(UTC))) == 4
        assert target.add_deployment("activate", "v1", None, 0.0, "", iso(datetime.now(UTC))) == 3
        assert target.add_alert("daily_cost", 3.0, 1.0, "later", iso(datetime.now(UTC)), "daily_cost:y") == 3
    finally:
        origin.close()
        target.close()


def test_migration_refuses_a_nonempty_postgres_target(tmp_path, migrated):
    source = tmp_path / "gateway.db"
    sqlite_source(source)
    target = Store(migrated)
    try:
        target.log_request(request("existing", iso(datetime.now(UTC))))
    finally:
        target.close()
    with pytest.raises(DataMigrationError, match="not empty"):
        migrate(source, migrated)


# ── 8. Releases and alerts ──────────────────────────────────────────


def test_release_activation_canary_rollback_and_history(store):
    releases = ReleaseManager(make_config(), store)
    assert (releases.state().active, releases.state().source) == ("v1", "config")
    assert releases.activate("v2", "promote").active == "v2"
    assert releases.set_canary("v1", 0.25).canary == "v1"
    assert releases.state().canary_share == 0.25
    assert releases.clear_canary().canary is None
    assert releases.rollback("undo").active == "v1"
    state = releases.state()
    assert (state.active, state.canary, state.source) == ("v1", None, "log")
    history = releases.history(10)
    assert [h["action"] for h in history] == ["rollback", "canary-off", "canary", "activate"]
    assert [h["id"] for h in history] == sorted((h["id"] for h in history), reverse=True)
    assert history[-1]["note"] == "promote" and history[0]["ts"].endswith("+00:00")


def test_concurrent_release_changes_are_serialised_across_instances(migrated):
    stores = [Store(migrated), Store(migrated)]
    managers = [ReleaseManager(make_config(), s) for s in stores]
    barrier = threading.Barrier(10)
    outcomes: list[str] = []

    def promote(i: int) -> None:
        barrier.wait()
        try:
            managers[i % 2].activate("v2")
            outcomes.append("activated")
        except ReleaseError:
            outcomes.append("refused")

    try:
        threads = [threading.Thread(target=promote, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(outcomes) == ["activated"] + ["refused"] * 9  # each change saw the latest state
        assert len(stores[0].deployments()) == 1
    finally:
        for s in stores:
            s.close()


def test_alerts_persist_filter_and_fire_once_across_instances(migrated):
    stores = [Store(migrated), Store(migrated)]
    try:
        now = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
        for i in range(6):  # 6 of 6 requests failed in the window: above the 50% error-rate limit
            stores[0].log_request(request(f"e{i}", iso(now - timedelta(minutes=i)), status="error"))
        config = make_config().alerts
        sent: list[str] = []
        monitors = [AlertMonitor(s, config, notify=lambda alert: sent.append(alert.kind)) for s in stores]
        barrier = threading.Barrier(2)

        def check(monitor: AlertMonitor) -> None:
            barrier.wait()
            monitor.check(now)

        threads = [threading.Thread(target=check, args=(m,)) for m in monitors]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sent == ["error_rate"]  # both instances saw it; the unique dedup key let one send it
        assert stores[1].add_alert("error_rate", 1.0, 0.5, "again", iso(now), "error_rate:after:initial") == 0
        stores[1].add_alert("daily_cost", 9.0, 5.0, "spend", iso(now + timedelta(seconds=1)), "daily_cost:2026-11-22")
        assert [a["kind"] for a in stores[1].alerts()] == ["daily_cost", "error_rate"]
        assert stores[1].last_alert("error_rate")["dedup_key"] == "error_rate:after:initial"
        assert stores[1].last_alert("missing") is None
    finally:
        for s in stores:
            s.close()


def test_retention_purge_on_postgres(store):
    now = utc_now()
    for i, age in enumerate((0, 40, 400)):
        ts = iso(now - timedelta(days=age))
        store.log_request(request(f"ret{i}", ts, calls=[CallRecord(1, "cheap", "small")]))
        store.add_feedback(f"ret{i}", "app", 1, None, ts)
    cutoff = iso(now - timedelta(days=30))
    assert store.purge_older_than(cutoff, iso(now), dry_run=True)["requests"] == 2
    deleted = store.purge_older_than(cutoff, iso(now))
    assert (deleted["requests"], deleted["calls"], deleted["feedback"]) == (2, 2, 2)
    assert store.scalar("SELECT count(*) FROM requests") == 1
