"""Aggregations over the request log, shared by the dashboard, admin API and alerts.

SQLAlchemy Core builds portable aggregation queries; Python computes latency
percentiles so the gateway still needs no dataframe or database extension.
"""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import case, func, literal, select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Table

from gateway.monitoring.store import FEEDBACK, REQUESTS, Store


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in [0, 100]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _conditions(table: Table, since: str | None = None, until: str | None = None,
                app_id: str | None = None) -> list[ColumnElement[bool]]:
    conditions = []
    if since:
        conditions.append(table.c.ts >= since)
    if until:
        conditions.append(table.c.ts < until)
    if app_id:
        conditions.append(table.c.app_id == app_id)
    return conditions


def _count_if(condition: ColumnElement[bool]) -> ColumnElement[Any]:
    """Portable replacement for SQLite's ``SUM(boolean expression)``."""
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def timestamp_bucket(table: Table, bucket: str) -> ColumnElement[str]:
    """Return a UTC bucket expression for the existing ISO timestamp text."""
    if bucket == "day":
        return table.c.day
    if bucket == "hour":
        return func.substr(table.c.ts, 1, 13) + literal(":00")
    raise ValueError(f"unsupported timestamp bucket: {bucket}")


def summary(store: Store, since: str | None = None, until: str | None = None,
            app_id: str | None = None) -> dict[str, Any]:
    """Headline numbers for a time range."""
    conditions = _conditions(REQUESTS, since, until, app_id)
    row = store.query(select(
        func.count().label("requests"),
        _count_if(REQUESTS.c.status == "ok").label("ok"),
        _count_if(REQUESTS.c.status == "error").label("errors"),
        _count_if(REQUESTS.c.status == "rejected").label("rejected"),
        func.coalesce(func.sum(REQUESTS.c.cost_usd), 0).label("cost_usd"),
        func.coalesce(func.sum(REQUESTS.c.prompt_tokens + REQUESTS.c.completion_tokens), 0).label("tokens"),
        func.coalesce(func.sum(REQUESTS.c.escalated), 0).label("escalated"),
        func.coalesce(func.sum(REQUESTS.c.fallback), 0).label("fallbacks"),
        _count_if(REQUESTS.c.tier == "strong").label("strong"),
    ).where(*conditions))[0]
    latencies = [r["latency_ms"] for r in store.query(
        select(REQUESTS.c.latency_ms).where(
            *conditions, REQUESTS.c.status == "ok", REQUESTS.c.latency_ms.is_not(None)))]
    router_ms = [r["router_ms"] for r in store.query(
        select(REQUESTS.c.router_ms).where(*conditions, REQUESTS.c.router_ms.is_not(None)))]
    feedback_conditions = _conditions(FEEDBACK, since, until, app_id)
    feedback = store.query(select(
        _count_if(FEEDBACK.c.score == 1).label("up"),
        _count_if(FEEDBACK.c.score == -1).label("down"),
    ).where(*feedback_conditions))[0]
    counted = row["ok"] + row["errors"]
    answered = row["ok"] or 0
    return {
        **row,
        "error_rate": row["errors"] / counted if counted else 0.0,
        "strong_share": row["strong"] / answered if answered else 0.0,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "router_p50_ms": percentile(router_ms, 50),
        "router_p95_ms": percentile(router_ms, 95),
        "feedback_up": feedback["up"],
        "feedback_down": feedback["down"],
    }


def timeseries(store: Store, since: str | None = None, app_id: str | None = None,
               bucket: str = "day") -> list[dict[str, Any]]:
    """One row per UTC day (or hour): requests, errors, cost and latency."""
    key = timestamp_bucket(REQUESTS, bucket).label("bucket")
    conditions = _conditions(REQUESTS, since, app_id=app_id)
    rows = store.query(select(
        key,
        func.count().label("requests"),
        _count_if(REQUESTS.c.status == "error").label("errors"),
        _count_if(REQUESTS.c.status == "rejected").label("rejected"),
        _count_if(REQUESTS.c.status == "ok").label("ok"),
        func.coalesce(func.sum(REQUESTS.c.cost_usd), 0).label("cost_usd"),
        func.coalesce(func.sum(REQUESTS.c.prompt_tokens + REQUESTS.c.completion_tokens), 0).label("tokens"),
    ).where(*conditions).group_by(key).order_by(key))
    latencies: dict[str, list[float]] = {}
    for row in store.query(select(key, REQUESTS.c.latency_ms).where(
            *conditions, REQUESTS.c.status == "ok", REQUESTS.c.latency_ms.is_not(None))):
        latencies.setdefault(row["bucket"], []).append(row["latency_ms"])
    for row in rows:
        counted = row["ok"] + row["errors"]
        row["error_rate"] = row["errors"] / counted if counted else 0.0
        row["p50_ms"] = percentile(latencies.get(row["bucket"], []), 50)
        row["p95_ms"] = percentile(latencies.get(row["bucket"], []), 95)
    return rows


def daily(store: Store, since: str | None = None, app_id: str | None = None) -> list[dict[str, Any]]:
    """One row per UTC day (``day`` key)."""
    return [{"day": row.pop("bucket"), **row} for row in timeseries(store, since, app_id, "day")]


def app_ids(store: Store) -> list[str]:
    statement = select(REQUESTS.c.app_id).distinct().order_by(REQUESTS.c.app_id)
    return [row["app_id"] for row in store.query(statement)]


def model_share(store: Store, since: str | None = None, app_id: str | None = None) -> list[dict[str, Any]]:
    """Share of answered requests per model (and tier)."""
    conditions = _conditions(REQUESTS, since, app_id=app_id)
    rows = store.query(select(
        REQUESTS.c.model,
        REQUESTS.c.tier,
        func.count().label("requests"),
        func.coalesce(func.sum(REQUESTS.c.cost_usd), 0).label("cost_usd"),
    ).where(*conditions, REQUESTS.c.status == "ok").group_by(
        REQUESTS.c.model, REQUESTS.c.tier).order_by(func.count().desc()))
    total = sum(row["requests"] for row in rows)
    for row in rows:
        row["share"] = row["requests"] / total if total else 0.0
    return rows


def release_stats(store: Store, since: str | None = None) -> list[dict[str, Any]]:
    """Stable vs canary: requests, error rate, latency, cost and feedback per release."""
    feedback = select(
        FEEDBACK.c.request_id,
        _count_if(FEEDBACK.c.score == 1).label("up"),
        _count_if(FEEDBACK.c.score == -1).label("down"),
    ).group_by(FEEDBACK.c.request_id).subquery()
    conditions = _conditions(REQUESTS, since)
    rows = store.query(select(
        REQUESTS.c.release,
        REQUESTS.c.canary,
        func.count().label("requests"),
        _count_if(REQUESTS.c.status == "error").label("errors"),
        _count_if(REQUESTS.c.status == "ok").label("ok"),
        func.coalesce(func.sum(REQUESTS.c.cost_usd), 0).label("cost_usd"),
        func.coalesce(func.sum(feedback.c.up), 0).label("feedback_up"),
        func.coalesce(func.sum(feedback.c.down), 0).label("feedback_down"),
    ).select_from(REQUESTS.outerjoin(feedback, feedback.c.request_id == REQUESTS.c.id)).where(
        *conditions).group_by(REQUESTS.c.release, REQUESTS.c.canary).order_by(
            REQUESTS.c.canary, REQUESTS.c.release))
    for row in rows:
        counted = row["ok"] + row["errors"]
        row["error_rate"] = row["errors"] / counted if counted else 0.0
        release_match = (REQUESTS.c.release.is_(None) if row["release"] is None
                         else REQUESTS.c.release == row["release"])
        latencies = [item["latency_ms"] for item in store.query(select(REQUESTS.c.latency_ms).where(
            *conditions, release_match, REQUESTS.c.canary == row["canary"], REQUESTS.c.status == "ok",
            REQUESTS.c.latency_ms.is_not(None)))]
        row["p50_ms"] = percentile(latencies, 50)
        rated = row["feedback_up"] + row["feedback_down"]
        row["thumbs_up_rate"] = row["feedback_up"] / rated if rated else None
    return rows


def latency_class_stats(store: Store, since: str | None = None,
                        app_id: str | None = None) -> list[dict[str, Any]]:
    """Per latency class: requests, errors, fallbacks, cost and latency."""
    conditions = _conditions(REQUESTS, since, app_id=app_id)
    latency_class = func.coalesce(REQUESTS.c.latency_class, "not recorded").label("latency_class")
    rows = store.query(select(
        latency_class,
        func.count().label("requests"),
        _count_if(REQUESTS.c.status == "ok").label("ok"),
        _count_if(REQUESTS.c.status == "error").label("errors"),
        func.coalesce(func.sum(REQUESTS.c.fallback), 0).label("fallbacks"),
        func.coalesce(func.sum(REQUESTS.c.cost_usd), 0).label("cost_usd"),
    ).where(*conditions).group_by(latency_class).order_by(latency_class))
    for row in rows:
        counted = row["ok"] + row["errors"]
        row["error_rate"] = row["errors"] / counted if counted else 0.0
        latencies = [item["latency_ms"] for item in store.query(select(REQUESTS.c.latency_ms).where(
            *conditions, func.coalesce(REQUESTS.c.latency_class, "not recorded") == row["latency_class"],
            REQUESTS.c.status == "ok", REQUESTS.c.latency_ms.is_not(None)))]
        row["p50_ms"] = percentile(latencies, 50)
        row["p95_ms"] = percentile(latencies, 95)
    return rows


def app_spend(store: Store, day: str) -> list[dict[str, Any]]:
    cost = func.coalesce(func.sum(REQUESTS.c.cost_usd), 0).label("cost_usd")
    return store.query(select(
        REQUESTS.c.app_id, func.count().label("requests"), cost,
    ).where(REQUESTS.c.day == day).group_by(REQUESTS.c.app_id).order_by(cost.desc()))


def recent(store: Store, limit: int = 50, app_id: str | None = None) -> list[dict[str, Any]]:
    columns = [REQUESTS.c[name] for name in (
        "ts", "id", "app_id", "release", "canary", "mode", "router", "route_score", "latency_class",
        "tier", "model", "escalated", "fallback", "prompt_tokens", "completion_tokens", "cost_usd",
        "latency_ms", "router_ms", "status", "http_status", "error_type",
    )]
    return store.query(select(*columns).where(
        *_conditions(REQUESTS, app_id=app_id)).order_by(REQUESTS.c.ts.desc()).limit(limit))
