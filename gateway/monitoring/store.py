"""SQLAlchemy Core storage for the SQLite request log and gateway state.

One row is stored per gateway request (``requests``) and provider call
(``calls``). Only metadata is stored, never prompts or answers. Timestamps
remain ISO-8601 UTC strings, and ``day`` remains a separate UTC date string.

The public Store API keeps accepting raw SQL because the dashboard, tests and
operational code use it for ad-hoc reads. Positional SQLite parameters are
translated to named SQLAlchemy binds so existing callers and database files
remain compatible while new code can use SQLAlchemy Core statements directly.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    Numeric,
    Table,
    Text,
    create_engine,
    delete,
    event,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import URL, Dialect, make_url
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator


class UTCISOString(TypeDecorator[str]):
    """ISO text on SQLite and timezone-aware timestamps on PostgreSQL."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(Text() if dialect.name == "sqlite" else DateTime(timezone=True))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "sqlite":
            return iso(value) if isinstance(value, datetime) else value
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: Any, _dialect: Dialect) -> str | None:
        if value is None or isinstance(value, str):
            return value
        return iso(value)


class UTCDateString(TypeDecorator[str]):
    """UTC date text on SQLite and a native date on PostgreSQL."""

    impl = Date
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(Text() if dialect.name == "sqlite" else Date())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None or dialect.name == "sqlite":
            return value.isoformat() if isinstance(value, date) else value
        return date.fromisoformat(value) if isinstance(value, str) else value

    def process_result_value(self, value: Any, _dialect: Dialect) -> str | None:
        return value.isoformat() if isinstance(value, date) else value


class Money(TypeDecorator[float]):
    """SQLite-compatible float storage and fixed-precision PostgreSQL money."""

    impl = Numeric(20, 12)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(Float() if dialect.name == "sqlite" else Numeric(20, 12))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None or dialect.name == "sqlite":
            return value
        return Decimal(str(value))

    def process_result_value(self, value: Any, _dialect: Dialect) -> float | None:
        return None if value is None else float(value)


UTC_TIMESTAMP = UTCISOString()
UTC_DAY = UTCDateString()
MONEY = Money()

METADATA = MetaData()

REQUESTS = Table(
    "requests",
    METADATA,
    Column("id", Text, primary_key=True),
    Column("ts", UTC_TIMESTAMP, nullable=False),
    Column("day", UTC_DAY, nullable=False),
    Column("app_id", Text, nullable=False),
    Column("requested_model", Text),
    Column("release", Text),
    Column("prompt_version", Text),
    Column("canary", Integer, nullable=False, server_default="0"),
    Column("mode", Text),
    Column("router", Text),
    Column("route_score", Float),
    Column("route_reason", Text),
    Column("threshold", Float),
    Column("latency_class", Text),
    Column("tier", Text),
    Column("model", Text),
    Column("provider_model", Text),
    Column("escalated", Integer, nullable=False, server_default="0"),
    Column("fallback", Integer, nullable=False, server_default="0"),
    Column("prompt_tokens", Integer, nullable=False, server_default="0"),
    Column("completion_tokens", Integer, nullable=False, server_default="0"),
    Column("cost_usd", MONEY, nullable=False, server_default="0"),
    Column("latency_ms", Float),
    Column("router_ms", Float),
    Column("status", Text, nullable=False),
    Column("http_status", Integer),
    Column("error_type", Text),
    Column("error_message", Text),
    Column("stream", Integer, nullable=False, server_default="0"),
)
Index("requests_ts", REQUESTS.c.ts)
Index("requests_day_app", REQUESTS.c.day, REQUESTS.c.app_id)

CALLS = Table(
    "calls",
    METADATA,
    Column("request_id", Text, primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("ts", UTC_TIMESTAMP, nullable=False),
    Column("tier", Text),
    Column("model", Text),
    Column("provider_model", Text),
    Column("prompt_tokens", Integer, nullable=False, server_default="0"),
    Column("completion_tokens", Integer, nullable=False, server_default="0"),
    Column("cost_usd", MONEY, nullable=False, server_default="0"),
    Column("latency_ms", Float),
    Column("status", Text, nullable=False),
    Column("error_type", Text),
    Column("error_message", Text),
    Column("confidence", Float),
    Column("check_reason", Text),
)

FEEDBACK = Table(
    "feedback",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", Text, nullable=False),
    Column("ts", UTC_TIMESTAMP, nullable=False),
    Column("app_id", Text, nullable=False),
    Column("score", Integer, nullable=False),
    Column("comment", Text),
    CheckConstraint("score IN (-1, 1)"),
    sqlite_autoincrement=True,
)
Index("feedback_request", FEEDBACK.c.request_id)

DEPLOYMENTS = Table(
    "deployments",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", UTC_TIMESTAMP, nullable=False),
    Column("action", Text, nullable=False),
    Column("active", Text, nullable=False),
    Column("canary_release", Text),
    Column("canary_share", Float, nullable=False, server_default="0"),
    Column("note", Text),
    sqlite_autoincrement=True,
)

ALERTS = Table(
    "alerts",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", UTC_TIMESTAMP, nullable=False),
    Column("kind", Text, nullable=False),
    Column("value", Float),
    Column("threshold", Float),
    Column("message", Text, nullable=False),
    Column("dedup_key", Text),
    sqlite_autoincrement=True,
)
Index("alerts_dedup_key", ALERTS.c.dedup_key, unique=True)

BUDGET_RESERVATIONS = Table(
    "budget_reservations",
    METADATA,
    Column("request_id", Text, primary_key=True),
    Column("app_id", Text, nullable=False),
    Column("day", UTC_DAY, nullable=False),
    Column("amount_usd", MONEY, nullable=False),
    Column("created_at", UTC_TIMESTAMP, nullable=False),
    Column("expires_at", UTC_TIMESTAMP, nullable=False),
)
Index("budget_reservations_app_day", BUDGET_RESERVATIONS.c.app_id, BUDGET_RESERVATIONS.c.day)

TABLES = {table.name: table for table in METADATA.tables.values()}

INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts)",
    "CREATE INDEX IF NOT EXISTS requests_day_app ON requests (day, app_id)",
    "CREATE INDEX IF NOT EXISTS feedback_request ON feedback (request_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS alerts_dedup_key ON alerts (dedup_key)",
    "CREATE INDEX IF NOT EXISTS budget_reservations_app_day ON budget_reservations (app_id, day)",
)

# Columns added after the first release, created on older databases at startup.
MIGRATIONS = {"requests": {"latency_class": "TEXT"}, "alerts": {"dedup_key": "TEXT"}}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat(timespec="milliseconds")


@dataclass
class CallRecord:
    seq: int
    tier: str
    model: str
    provider_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    status: str = "ok"
    error_type: str | None = None
    error_message: str | None = None
    confidence: float | None = None
    check_reason: str | None = None


@dataclass
class RequestRecord:
    id: str
    ts: str
    app_id: str
    requested_model: str | None = None
    release: str | None = None
    prompt_version: str | None = None
    canary: bool = False
    mode: str | None = None
    router: str | None = None
    route_score: float | None = None
    route_reason: str | None = None
    threshold: float | None = None
    latency_class: str | None = None
    tier: str | None = None
    model: str | None = None
    provider_model: str | None = None
    escalated: bool = False
    fallback: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    router_ms: float | None = None
    status: str = "error"
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    stream: bool = False
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def day(self) -> str:
        return self.ts[:10]


@dataclass(frozen=True)
class BudgetReservation:
    """The outcome of ``Store.reserve_budget``."""

    admitted: bool
    amount: float
    spent: float
    reserved: float
    limit: float


class Store:
    def __init__(self, path: str | Path, *, pool_size: int = 5, max_overflow: int = 10,
                 pool_timeout_s: float = 30, pool_recycle_s: int = 1800, verify_schema: bool = True):
        target = str(path)
        is_url = "://" in target
        self.path = target
        self._is_sqlite = not is_url or make_url(target).get_backend_name() == "sqlite"
        if self._is_sqlite and target != ":memory:" and not is_url:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        if self._is_sqlite:
            database = target if not is_url else make_url(target).database or ":memory:"
            options: dict[str, Any] = {"connect_args": {"check_same_thread": False, "timeout": 10}}
            if database == ":memory:":
                options["poolclass"] = StaticPool
            url = make_url(target) if is_url else URL.create("sqlite+pysqlite", database=database)
        else:
            url = make_url(target)
            options = {
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": pool_timeout_s,
                "pool_recycle": pool_recycle_s,
                "pool_pre_ping": True,
            }
        self.engine = create_engine(url, **options)
        if self._is_sqlite:
            event.listen(self.engine, "connect", self._configure_sqlite)
        try:
            if self._is_sqlite:
                with self._lock:
                    METADATA.create_all(self.engine)
                    self._migrate()
                    self._ensure_indexes()
            elif verify_schema:
                from gateway.migrations import verify_database_revision

                verify_database_revision(target)
        except DBAPIError as error:
            self._raise_dbapi(error)

    def _configure_sqlite(self, connection: sqlite3.Connection, _record: Any) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=10000")
            if self.engine.url.database != ":memory:":
                cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

    @staticmethod
    def _raise_dbapi(error: DBAPIError) -> None:
        if isinstance(error.orig, sqlite3.Error):
            raise error.orig from error
        raise error

    def _handle_dbapi(self, error: DBAPIError) -> None:
        if self._is_sqlite:
            self._raise_dbapi(error)
        raise error

    def _check_open(self) -> None:
        if self._closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

    def _migrate(self) -> None:
        with self.engine.begin() as connection:
            inspector = inspect(connection)
            for table, columns in MIGRATIONS.items():
                existing = {column["name"] for column in inspector.get_columns(table)}
                for column, kind in columns.items():
                    if column not in existing:
                        connection.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {kind}')

    def _ensure_indexes(self) -> None:
        """Restore indexes that may be missing from an existing SQLite database."""
        with self.engine.begin() as connection:
            for statement in INDEX_DDL:
                connection.exec_driver_sql(statement)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.engine.dispose()
                self._closed = True

    # ── generic compatibility API ──────────────────────────────────
    @staticmethod
    def _driver_params(params: Sequence[Any] | Mapping[str, Any]) -> tuple[Any, ...] | Mapping[str, Any]:
        """Return DBAPI parameters without interpreting SQL text."""
        return params if isinstance(params, Mapping) else tuple(params)

    @staticmethod
    def _postgres_positional_sql(sql: str) -> str:
        """Translate SQLite-style qmarks outside SQL quotes/comments for psycopg."""
        output: list[str] = []
        index = 0
        state = "plain"
        dollar_tag = ""
        while index < len(sql):
            char = sql[index]
            pair = sql[index:index + 2]
            if state == "plain":
                if pair == "--":
                    state = "line_comment"
                    output.append(pair)
                    index += 2
                    continue
                if pair == "/*":
                    state = "block_comment"
                    output.append(pair)
                    index += 2
                    continue
                if char in ("'", '"'):
                    state = "single_quote" if char == "'" else "double_quote"
                elif char == "$":
                    end = sql.find("$", index + 1)
                    candidate = sql[index:end + 1] if end >= 0 else ""
                    if candidate and candidate[1:-1].replace("_", "a").isalnum():
                        state, dollar_tag = "dollar_quote", candidate
                output.append("%s" if char == "?" else char)
                index += 1
                continue
            output.append(char)
            if state in ("single_quote", "double_quote") and char == ("'" if state == "single_quote" else '"'):
                if index + 1 < len(sql) and sql[index + 1] == char:
                    output.append(char)
                    index += 2
                    continue
                state = "plain"
            elif state == "line_comment" and char in "\r\n":
                state = "plain"
            elif state == "block_comment" and pair == "*/":
                output.append("/")
                index += 2
                state = "plain"
                continue
            elif state == "dollar_quote" and sql.startswith(dollar_tag, index):
                output.extend(dollar_tag[1:])
                index += len(dollar_tag)
                state = "plain"
                continue
            index += 1
        return "".join(output)

    def _execute_raw(self, connection: Any, sql: str,
                     params: Sequence[Any] | Mapping[str, Any]) -> Any:
        bindings = self._driver_params(params)
        if self._is_sqlite:
            return connection.exec_driver_sql(sql, bindings)
        if isinstance(bindings, Mapping):
            return connection.execute(text(sql), bindings)
        return connection.exec_driver_sql(self._postgres_positional_sql(sql), bindings)

    def query(self, sql: Any, params: Sequence[Any] | Mapping[str, Any] = ()) -> list[dict[str, Any]]:
        self._check_open()
        try:
            with self._lock, self.engine.connect() as connection:
                if isinstance(sql, str):
                    result = self._execute_raw(connection, sql, params)
                else:
                    if params:
                        raise TypeError("parameters for SQLAlchemy statements must be bound in the statement")
                    result = connection.execute(sql)
                return [dict(row._mapping) for row in result.fetchall()]
        except DBAPIError as error:
            self._handle_dbapi(error)

    def scalar(self, sql: Any, params: Sequence[Any] | Mapping[str, Any] = ()) -> Any:
        self._check_open()
        try:
            with self._lock, self.engine.connect() as connection:
                if isinstance(sql, str):
                    return self._execute_raw(connection, sql, params).scalar()
                if params:
                    raise TypeError("parameters for SQLAlchemy statements must be bound in the statement")
                return connection.execute(sql).scalar()
        except DBAPIError as error:
            self._handle_dbapi(error)

    def _insert(self, table_name: str, values: dict[str, Any]) -> int:
        self._check_open()
        try:
            with self._lock, self.engine.begin() as connection:
                result = connection.execute(TABLES[table_name].insert().values(**values))
                return int(result.inserted_primary_key[0] or 0)
        except DBAPIError as error:
            self._handle_dbapi(error)

    # ── requests ───────────────────────────────────────────────────
    def log_request(self, record: RequestRecord) -> None:
        """Log a request and calls and settle its reservation atomically."""
        self._check_open()
        values = asdict(record)
        calls = values.pop("calls")
        values["day"] = record.day
        for key in ("canary", "escalated", "fallback", "stream"):
            values[key] = int(bool(values[key]))
        try:
            with self._lock, self.engine.begin() as connection:
                connection.execute(REQUESTS.insert().values(**values))
                if calls:
                    connection.execute(CALLS.insert(), [{"request_id": record.id, "ts": record.ts, **call}
                                                        for call in calls])
                connection.execute(delete(BUDGET_RESERVATIONS).where(
                    BUDGET_RESERVATIONS.c.request_id == record.id))
        except DBAPIError as error:
            self._handle_dbapi(error)

    # ── budget reservations ────────────────────────────────────────
    def reserve_budget(self, request_id: str, app_id: str, day: str, limit: float, estimate: float, now: str,
                       expires_at: str) -> BudgetReservation:
        """Atomically admit and reserve budget using SQLite's immediate write lock."""
        self._check_open()
        with self._lock:
            connection = self.engine.connect()
            try:
                if self._is_sqlite:
                    connection.exec_driver_sql("BEGIN IMMEDIATE")
                else:
                    connection.begin()
                    lock_key = f"gateway-budget:{app_id}:{day}"
                    connection.execute(select(func.pg_advisory_xact_lock(func.hashtextextended(lock_key, 0))))
                connection.execute(delete(BUDGET_RESERVATIONS).where(BUDGET_RESERVATIONS.c.expires_at <= now))
                spent = float(connection.scalar(select(func.coalesce(func.sum(REQUESTS.c.cost_usd), 0)).where(
                    REQUESTS.c.day == day, REQUESTS.c.app_id == app_id)) or 0.0)
                reserved = float(connection.scalar(select(func.coalesce(func.sum(BUDGET_RESERVATIONS.c.amount_usd),
                                                                          0)).where(
                    BUDGET_RESERVATIONS.c.day == day, BUDGET_RESERVATIONS.c.app_id == app_id)) or 0.0)
                left = limit - spent - reserved
                admitted = left > 0
                amount = min(max(estimate, 0.0), left) if admitted else 0.0
                if admitted:
                    connection.execute(BUDGET_RESERVATIONS.insert().values(
                        request_id=request_id, app_id=app_id, day=day, amount_usd=amount,
                        created_at=now, expires_at=expires_at))
                connection.commit()
            except DBAPIError as error:
                connection.rollback()
                self._handle_dbapi(error)
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        return BudgetReservation(admitted, amount, spent, reserved, limit)

    def reservations(self, app_id: str | None = None) -> list[dict[str, Any]]:
        statement = select(BUDGET_RESERVATIONS)
        if app_id is not None:
            statement = statement.where(BUDGET_RESERVATIONS.c.app_id == app_id)
        return self.query(statement.order_by(BUDGET_RESERVATIONS.c.created_at))

    def request_app(self, request_id: str) -> str | None:
        return self.scalar(select(REQUESTS.c.app_id).where(REQUESTS.c.id == request_id))

    def app_spend(self, app_id: str, day: str) -> float:
        return float(self.scalar(select(func.coalesce(func.sum(REQUESTS.c.cost_usd), 0)).where(
            REQUESTS.c.day == day, REQUESTS.c.app_id == app_id)) or 0.0)

    def total_spend(self, day: str) -> float:
        return float(self.scalar(select(func.coalesce(func.sum(REQUESTS.c.cost_usd), 0)).where(
            REQUESTS.c.day == day)) or 0.0)

    # ── feedback ───────────────────────────────────────────────────
    def add_feedback(self, request_id: str, app_id: str, score: int, comment: str | None, ts: str) -> int:
        return self._insert("feedback", {"request_id": request_id, "ts": ts, "app_id": app_id, "score": score,
                                         "comment": comment})

    # ── deployments ────────────────────────────────────────────────
    def add_deployment(self, action: str, active: str, canary_release: str | None, canary_share: float,
                       note: str, ts: str) -> int:
        return self._insert("deployments", {"ts": ts, "action": action, "active": active,
                                            "canary_release": canary_release, "canary_share": canary_share,
                                            "note": note})

    def mutate_deployment(self, mutator: Callable[[list[dict[str, Any]]], dict[str, Any]]) -> dict[str, Any]:
        """Serialize a read/validate/append deployment transition across gateway instances."""
        self._check_open()
        with self._lock:
            connection = self.engine.connect()
            try:
                if self._is_sqlite:
                    connection.exec_driver_sql("BEGIN IMMEDIATE")
                else:
                    connection.begin()
                    connection.execute(select(func.pg_advisory_xact_lock(func.hashtextextended(
                        "gateway-deployment-state", 0))))
                rows = [dict(row._mapping) for row in connection.execute(
                    select(DEPLOYMENTS).order_by(DEPLOYMENTS.c.id.desc()).limit(1000)).fetchall()]
                values = mutator(rows)
                result = connection.execute(DEPLOYMENTS.insert().values(**values))
                values = {"id": int(result.inserted_primary_key[0] or 0), **values}
                connection.commit()
                return values
            except DBAPIError as error:
                connection.rollback()
                self._handle_dbapi(error)
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def deployments(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(select(DEPLOYMENTS).order_by(DEPLOYMENTS.c.id.desc()).limit(limit))

    # ── alerts ─────────────────────────────────────────────────────
    def add_alert(self, kind: str, value: float, threshold: float, message: str, ts: str,
                  dedup_key: str | None = None) -> int:
        try:
            return self._insert("alerts", {"ts": ts, "kind": kind, "value": value, "threshold": threshold,
                                           "message": message, "dedup_key": dedup_key})
        except (sqlite3.IntegrityError, IntegrityError):
            if dedup_key is None:
                raise
            return 0

    def last_alert(self, kind: str) -> dict[str, Any] | None:
        rows = self.query(select(ALERTS).where(ALERTS.c.kind == kind).order_by(ALERTS.c.id.desc()).limit(1))
        return rows[0] if rows else None

    def alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(select(ALERTS).order_by(ALERTS.c.id.desc()).limit(limit))
