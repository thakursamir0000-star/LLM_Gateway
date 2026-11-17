"""Aggregations over the request log, shared by the dashboard, the admin API and alerts.

Plain SQL plus Python percentiles, so the gateway needs no pandas. Latency
percentiles cover successful requests; rejected requests (client errors,
budget limits) are counted but are not errors.
"""

from __future__ import annotations

import math
from typing import Any

from gateway.monitoring.store import Store


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in [0, 100]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _where(since: str | None = None, until: str | None = None, app_id: str | None = None,
           prefix: str = "") -> tuple[str, list[Any]]:
    clauses, params = [], []
    if since:
        clauses.append(f"{prefix}ts >= ?")
        params.append(since)
    if until:
        clauses.append(f"{prefix}ts < ?")
        params.append(until)
    if app_id:
        clauses.append(f"{prefix}app_id = ?")
        params.append(app_id)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def summary(store: Store, since: str | None = None, until: str | None = None,
            app_id: str | None = None) -> dict[str, Any]:
    """Headline numbers for a time range."""
    where, params = _where(since, until, app_id)
    row = store.query(
        "SELECT COUNT(*) AS requests, "
        "COALESCE(SUM(status = 'ok'), 0) AS ok, COALESCE(SUM(status = 'error'), 0) AS errors, "
        "COALESCE(SUM(status = 'rejected'), 0) AS rejected, COALESCE(SUM(cost_usd), 0) AS cost_usd, "
        "COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens, "
        "COALESCE(SUM(escalated), 0) AS escalated, COALESCE(SUM(fallback), 0) AS fallbacks, "
        "COALESCE(SUM(tier = 'strong'), 0) AS strong "
        f"FROM requests{where}", params)[0]
    latencies = [r["latency_ms"] for r in store.query(
        f"SELECT latency_ms FROM requests{where}{' AND' if where else ' WHERE'} status = 'ok' "
        "AND latency_ms IS NOT NULL", params)]
    router_ms = [r["router_ms"] for r in store.query(
        f"SELECT router_ms FROM requests{where}{' AND' if where else ' WHERE'} router_ms IS NOT NULL", params)]
    fb_where, fb_params = _where(since, until, app_id)
    feedback = store.query(
        f"SELECT COALESCE(SUM(score = 1), 0) AS up, COALESCE(SUM(score = -1), 0) AS down FROM feedback{fb_where}",
        fb_params)[0]
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


_BUCKETS = {"day": "day", "hour": "substr(ts, 1, 13) || ':00'"}


def timeseries(store: Store, since: str | None = None, app_id: str | None = None,
               bucket: str = "day") -> list[dict[str, Any]]:
    """One row per UTC day (or hour): requests, errors, error rate, cost, p50/p95 latency."""
    key = _BUCKETS[bucket]
    where, params = _where(since, None, app_id)
    rows = store.query(
        f"SELECT {key} AS bucket, COUNT(*) AS requests, COALESCE(SUM(status = 'error'), 0) AS errors, "
        "COALESCE(SUM(status = 'rejected'), 0) AS rejected, COALESCE(SUM(status = 'ok'), 0) AS ok, "
        "COALESCE(SUM(cost_usd), 0) AS cost_usd, COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens "
        f"FROM requests{where} GROUP BY bucket ORDER BY bucket", params)
    latencies: dict[str, list[float]] = {}
    for r in store.query(f"SELECT {key} AS bucket, latency_ms FROM requests{where}{' AND' if where else ' WHERE'} "
                         "status = 'ok' AND latency_ms IS NOT NULL", params):
        latencies.setdefault(r["bucket"], []).append(r["latency_ms"])
    for row in rows:
        counted = row["ok"] + row["errors"]
        row["error_rate"] = row["errors"] / counted if counted else 0.0
        row["p50_ms"] = percentile(latencies.get(row["bucket"], []), 50)
        row["p95_ms"] = percentile(latencies.get(row["bucket"], []), 95)
    return rows


def daily(store: Store, since: str | None = None, app_id: str | None = None) -> list[dict[str, Any]]:
    """One row per UTC day (``day`` key): requests, errors, error rate, cost, p50/p95 latency."""
    return [{"day": row.pop("bucket"), **row} for row in timeseries(store, since, app_id, "day")]


def app_ids(store: Store) -> list[str]:
    return [r["app_id"] for r in store.query("SELECT DISTINCT app_id FROM requests ORDER BY app_id")]


def model_share(store: Store, since: str | None = None, app_id: str | None = None) -> list[dict[str, Any]]:
    """Share of answered requests per model (and tier)."""
    where, params = _where(since, None, app_id)
    rows = store.query(
        f"SELECT model, tier, COUNT(*) AS requests, COALESCE(SUM(cost_usd), 0) AS cost_usd FROM requests{where}"
        f"{' AND' if where else ' WHERE'} status = 'ok' GROUP BY model, tier ORDER BY requests DESC", params)
    total = sum(r["requests"] for r in rows)
    for r in rows:
        r["share"] = r["requests"] / total if total else 0.0
    return rows


def release_stats(store: Store, since: str | None = None) -> list[dict[str, Any]]:
    """Stable vs canary: requests, error rate, latency, cost and feedback per release."""
    where, params = _where(since, None, None, prefix="r.")
    rows = store.query(
        "SELECT r.release, r.canary, COUNT(*) AS requests, COALESCE(SUM(r.status = 'error'), 0) AS errors, "
        "COALESCE(SUM(r.status = 'ok'), 0) AS ok, COALESCE(SUM(r.cost_usd), 0) AS cost_usd, "
        "COALESCE(SUM(f.up), 0) AS feedback_up, COALESCE(SUM(f.down), 0) AS feedback_down "
        "FROM requests r LEFT JOIN (SELECT request_id, SUM(score = 1) AS up, SUM(score = -1) AS down "
        f"FROM feedback GROUP BY request_id) f ON f.request_id = r.id{where} "
        "GROUP BY r.release, r.canary ORDER BY r.canary, r.release", params)
    for row in rows:
        counted = row["ok"] + row["errors"]
        row["error_rate"] = row["errors"] / counted if counted else 0.0
        latencies = [r["latency_ms"] for r in store.query(
            f"SELECT latency_ms FROM requests r{where}{' AND' if where else ' WHERE'} r.release IS ? "
            "AND r.canary = ? AND r.status = 'ok' AND r.latency_ms IS NOT NULL",
            [*params, row["release"], row["canary"]])]
        row["p50_ms"] = percentile(latencies, 50)
        rated = row["feedback_up"] + row["feedback_down"]
        row["thumbs_up_rate"] = row["feedback_up"] / rated if rated else None
    return rows


def latency_class_stats(store: Store, since: str | None = None, app_id: str | None = None) -> list[dict[str, Any]]:
    """Per latency class: requests, error rate, fallbacks, cost and p50/p95 latency of successful requests."""
    where, params = _where(since, None, app_id)
    rows = store.query(
        "SELECT COALESCE(latency_class, 'not recorded') AS latency_class, COUNT(*) AS requests, "
        "COALESCE(SUM(status = 'ok'), 0) AS ok, COALESCE(SUM(status = 'error'), 0) AS errors, "
        "COALESCE(SUM(fallback), 0) AS fallbacks, COALESCE(SUM(cost_usd), 0) AS cost_usd "
        f"FROM requests{where} GROUP BY 1 ORDER BY 1", params)
    for row in rows:
        counted = row["ok"] + row["errors"]
        row["error_rate"] = row["errors"] / counted if counted else 0.0
        latencies = [r["latency_ms"] for r in store.query(
            f"SELECT latency_ms FROM requests{where}{' AND' if where else ' WHERE'} "
            "COALESCE(latency_class, 'not recorded') = ? AND status = 'ok' AND latency_ms IS NOT NULL",
            [*params, row["latency_class"]])]
        row["p50_ms"] = percentile(latencies, 50)
        row["p95_ms"] = percentile(latencies, 95)
    return rows


def app_spend(store: Store, day: str) -> list[dict[str, Any]]:
    return store.query("SELECT app_id, COUNT(*) AS requests, COALESCE(SUM(cost_usd), 0) AS cost_usd "
                       "FROM requests WHERE day = ? GROUP BY app_id ORDER BY cost_usd DESC", (day,))


def recent(store: Store, limit: int = 50, app_id: str | None = None) -> list[dict[str, Any]]:
    where, params = _where(None, None, app_id)
    return store.query(
        "SELECT ts, id, app_id, release, canary, mode, router, route_score, latency_class, tier, model, escalated, "
        "fallback, "
        "prompt_tokens, completion_tokens, cost_usd, latency_ms, router_ms, status, http_status, error_type "
        f"FROM requests{where} ORDER BY ts DESC LIMIT ?", [*params, limit])
