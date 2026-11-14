from datetime import UTC, datetime, timedelta

import pytest

from gateway.config import AlertConfig
from gateway.monitoring import metrics
from gateway.monitoring.alerts import AlertMonitor, evaluate, webhook_notifier
from gateway.monitoring.store import CallRecord, RequestRecord, Store, iso

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "gw.db")
    yield s
    s.close()


_counter = iter(range(10_000))


def add(store, *, at=NOW, app="a", status="ok", tier="cheap", model="small", cost=0.001, latency=100.0,
        release="v1", canary=False, router_ms=None, escalated=False):
    record = RequestRecord(id=f"req-{next(_counter)}", ts=iso(at), app_id=app, status=status, tier=tier, model=model,
                           cost_usd=cost, latency_ms=latency, release=release, canary=canary, router_ms=router_ms,
                           escalated=escalated, prompt_tokens=10, completion_tokens=5,
                           calls=[CallRecord(1, tier, model, cost_usd=cost)])
    store.log_request(record)
    return record


def test_log_request_with_calls_and_spend(store):
    record = add(store, cost=0.25, app="x")
    add(store, cost=0.5, app="y")
    add(store, cost=1.0, app="x", at=NOW - timedelta(days=1))
    assert store.request_app(record.id) == "x" and store.request_app("missing") is None
    assert store.app_spend("x", "2026-11-15") == pytest.approx(0.25)
    assert store.total_spend("2026-11-15") == pytest.approx(0.75)
    calls = store.query("SELECT * FROM calls WHERE request_id = ?", (record.id,))
    assert len(calls) == 1 and calls[0]["model"] == "small"


def test_store_works_in_memory():
    s = Store(":memory:")
    add(s)
    assert metrics.summary(s)["requests"] == 1


def test_percentile():
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([5.0], 95) == 5.0
    assert metrics.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert metrics.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_summary_counts_errors_but_not_rejections(store):
    for latency in (100, 200, 300, 400):
        add(store, latency=latency)
    add(store, tier="strong", model="large", latency=1000, router_ms=2.0)
    add(store, status="error", latency=5)
    add(store, status="rejected", cost=0)
    s = metrics.summary(store)
    assert s["requests"] == 7 and s["ok"] == 5 and s["errors"] == 1 and s["rejected"] == 1
    assert s["error_rate"] == pytest.approx(1 / 6)
    assert s["p50_ms"] == 300  # successful requests only
    assert s["strong_share"] == pytest.approx(1 / 5)
    assert s["router_p50_ms"] == 2.0


def test_summary_filters_by_time_and_app(store):
    add(store, app="a", at=NOW - timedelta(days=2))
    add(store, app="a")
    add(store, app="b")
    assert metrics.summary(store, since=iso(NOW - timedelta(days=1)))["requests"] == 2
    assert metrics.summary(store, app_id="a")["requests"] == 2


def test_daily_and_model_share(store):
    add(store, at=NOW - timedelta(days=1), cost=0.1, latency=50)
    add(store, cost=0.2, latency=100)
    add(store, cost=0.3, tier="strong", model="large", latency=300)
    add(store, status="error")
    days = metrics.daily(store)
    assert [d["day"] for d in days] == ["2026-11-14", "2026-11-15"]
    assert days[1]["requests"] == 3 and days[1]["cost_usd"] == pytest.approx(0.501)
    assert days[1]["error_rate"] == pytest.approx(1 / 3) and days[1]["p95_ms"] == pytest.approx(290)
    share = {r["model"]: r["share"] for r in metrics.model_share(store)}
    assert share == {"small": pytest.approx(2 / 3), "large": pytest.approx(1 / 3)}


def test_release_stats_compare_stable_and_canary(store):
    stable = [add(store, release="v1") for _ in range(3)]
    canary = [add(store, release="v2", canary=True), add(store, release="v2", canary=True, status="error")]
    store.add_feedback(stable[0].id, "a", 1, None, iso(NOW))
    store.add_feedback(canary[0].id, "a", -1, "wrong", iso(NOW))
    rows = {(r["release"], r["canary"]): r for r in metrics.release_stats(store)}
    assert rows[("v1", 0)]["requests"] == 3 and rows[("v1", 0)]["thumbs_up_rate"] == 1.0
    assert rows[("v2", 1)]["error_rate"] == 0.5 and rows[("v2", 1)]["feedback_down"] == 1


def test_feedback_score_is_constrained(store):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.add_feedback("r", "a", 5, None, iso(NOW))


# ── alerts ──────────────────────────────────────────────────────────

ALERTS = AlertConfig(error_rate=0.25, window_minutes=15, min_requests=4, daily_cost_usd=1.0, check_interval_s=60)


def test_no_error_rate_alert_below_min_requests(store):
    add(store, status="error")
    add(store, status="error")
    assert evaluate(store, ALERTS, NOW) == []


def test_error_rate_alert_fires_once_per_window(store):
    for status in ("ok", "ok", "error", "error", "rejected", "rejected"):
        add(store, status=status, at=NOW - timedelta(minutes=1))
    add(store, status="error", at=NOW - timedelta(hours=1))  # outside the window
    sent = []
    monitor = AlertMonitor(store, ALERTS, notify=sent.append, clock=lambda: NOW)
    fired = monitor.check(NOW)
    assert [a.kind for a in fired] == ["error_rate"] and fired[0].value == 0.5
    assert "2/4 requests" in fired[0].message
    assert monitor.check(NOW + timedelta(minutes=5)) == []  # same window: not repeated
    assert len(sent) == 1 and store.alerts()[0]["kind"] == "error_rate"


def test_daily_cost_alert_fires_once_per_day(store):
    add(store, cost=0.6)
    sent = []
    monitor = AlertMonitor(store, ALERTS, notify=sent.append)
    assert monitor.check(NOW) == []
    add(store, cost=0.6)
    assert [a.kind for a in monitor.check(NOW)] == ["daily_cost"]
    assert monitor.check(NOW + timedelta(hours=1)) == []
    add(store, cost=2.0, at=NOW + timedelta(days=1))
    assert [a.kind for a in monitor.check(NOW + timedelta(days=1))] == ["daily_cost"]


def test_maybe_check_is_throttled(store):
    clock = {"now": NOW}
    calls = []
    monitor = AlertMonitor(store, ALERTS, notify=calls.append, clock=lambda: clock["now"])
    add(store, cost=5.0)
    assert len(monitor.maybe_check()) == 1
    add(store, cost=5.0, at=NOW + timedelta(days=1))
    clock["now"] = NOW + timedelta(seconds=30)
    assert monitor.maybe_check() == []  # not due yet


def test_webhook_notifier_posts_json(monkeypatch):
    import threading

    posted = []
    done = threading.Event()

    def fake_post(url, json, timeout):
        posted.append((url, json))
        done.set()

    monkeypatch.setattr("httpx.post", fake_post)
    from gateway.monitoring.alerts import Alert

    webhook_notifier("https://hooks.test/x")(Alert("daily_cost", 2.0, 1.0, "too much"))
    assert done.wait(5)
    assert posted[0][0] == "https://hooks.test/x"
    assert posted[0][1]["kind"] == "daily_cost" and posted[0][1]["source"] == "llm-gateway"
