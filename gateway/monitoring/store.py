"""SQLite storage: the request log, provider calls, feedback, deployments and alerts.

One row per gateway request (``requests``) and one per provider call
(``calls``; a cascade or a fallback makes two). Only metadata is stored:
never the prompt or the answer. Timestamps are ISO-8601 UTC strings, so they
sort and compare as text; ``day`` (UTC) is kept separately for budgets and
daily charts.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    day TEXT NOT NULL,
    app_id TEXT NOT NULL,
    requested_model TEXT,
    release TEXT,
    prompt_version TEXT,
    canary INTEGER NOT NULL DEFAULT 0,
    mode TEXT,
    router TEXT,
    route_score REAL,
    route_reason TEXT,
    threshold REAL,
    tier TEXT,
    model TEXT,
    provider_model TEXT,
    escalated INTEGER NOT NULL DEFAULT 0,
    fallback INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL,
    router_ms REAL,
    status TEXT NOT NULL,
    http_status INTEGER,
    error_type TEXT,
    error_message TEXT,
    stream INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
CREATE INDEX IF NOT EXISTS requests_day_app ON requests (day, app_id);

CREATE TABLE IF NOT EXISTS calls (
    request_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts TEXT NOT NULL,
    tier TEXT,
    model TEXT,
    provider_model TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL,
    status TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT,
    confidence REAL,
    check_reason TEXT,
    PRIMARY KEY (request_id, seq)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    app_id TEXT NOT NULL,
    score INTEGER NOT NULL CHECK (score IN (-1, 1)),
    comment TEXT
);
CREATE INDEX IF NOT EXISTS feedback_request ON feedback (request_id);

CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    action TEXT NOT NULL,
    active TEXT NOT NULL,
    canary_release TEXT,
    canary_share REAL NOT NULL DEFAULT 0,
    note TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    value REAL,
    threshold REAL,
    message TEXT NOT NULL
);
"""


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
    status: str = "ok"  # ok | error
    error_type: str | None = None
    error_message: str | None = None
    confidence: float | None = None  # cascade answer check
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
    status: str = "error"  # ok | error (counts toward the error rate) | rejected (client error, budget)
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    stream: bool = False
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def day(self) -> str:
        return self.ts[:10]


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── generic ─────────────────────────────────────────────────────
    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def scalar(self, sql: str, params: tuple | dict = ()) -> Any:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def _insert(self, table: str, values: dict[str, Any]) -> int:
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self._lock:
            cursor = self._conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values()))
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    # ── requests ────────────────────────────────────────────────────
    def log_request(self, record: RequestRecord) -> None:
        values = asdict(record)
        calls = values.pop("calls")
        values["day"] = record.day
        for key in ("canary", "escalated", "fallback", "stream"):
            values[key] = int(bool(values[key]))
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self._lock:
            self._conn.execute(f"INSERT INTO requests ({columns}) VALUES ({marks})", tuple(values.values()))
            for call in calls:
                call_values = {"request_id": record.id, "ts": record.ts, **call}
                call_columns = ", ".join(call_values)
                call_marks = ", ".join("?" for _ in call_values)
                self._conn.execute(f"INSERT INTO calls ({call_columns}) VALUES ({call_marks})",
                                   tuple(call_values.values()))
            self._conn.commit()

    def request_app(self, request_id: str) -> str | None:
        """The app that made ``request_id`` (None if there is no such request)."""
        return self.scalar("SELECT app_id FROM requests WHERE id = ?", (request_id,))

    def app_spend(self, app_id: str, day: str) -> float:
        return float(self.scalar("SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE day = ? AND app_id = ?",
                                 (day, app_id)) or 0.0)

    def total_spend(self, day: str) -> float:
        return float(self.scalar("SELECT COALESCE(SUM(cost_usd), 0) FROM requests WHERE day = ?", (day,)) or 0.0)

    # ── feedback ────────────────────────────────────────────────────
    def add_feedback(self, request_id: str, app_id: str, score: int, comment: str | None, ts: str) -> int:
        return self._insert("feedback", {"request_id": request_id, "ts": ts, "app_id": app_id, "score": score,
                                         "comment": comment})

    # ── deployments ─────────────────────────────────────────────────
    def add_deployment(self, action: str, active: str, canary_release: str | None, canary_share: float,
                       note: str, ts: str) -> int:
        return self._insert("deployments", {"ts": ts, "action": action, "active": active,
                                            "canary_release": canary_release, "canary_share": canary_share,
                                            "note": note})

    def deployments(self, limit: int = 50) -> list[dict[str, Any]]:
        """Deployment log, newest first."""
        return self.query("SELECT * FROM deployments ORDER BY id DESC LIMIT ?", (limit,))

    # ── alerts ──────────────────────────────────────────────────────
    def add_alert(self, kind: str, value: float, threshold: float, message: str, ts: str) -> int:
        return self._insert("alerts", {"ts": ts, "kind": kind, "value": value, "threshold": threshold,
                                       "message": message})

    def last_alert(self, kind: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM alerts WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,))
        return rows[0] if rows else None

    def alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
