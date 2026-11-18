"""Concurrency-safe daily budgets (audit finding C1): requests reserve their estimated maximum cost
atomically before the provider call, and the reservation is settled exactly once when they are logged.

Fake providers and temporary SQLite files only.
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from gateway.app import create_app
from gateway.manage.budgets import reservation_estimate
from gateway.monitoring.store import RequestRecord, Store, iso
from gateway.providers.base import OVERLOADED, Completion, ProviderError
from gateway.providers.fake import FakeProvider
from gateway.schemas import ChatRequest
from gateway.service import RequestContext
from tests.conftest import make_config

MSG = [{"role": "user", "content": "hi"}]


async def no_sleep(_: float) -> None:
    return None


def answer(prompt: int = 10, completion: int = 50):
    def script(model, messages, params):
        return Completion("ok", "stop", prompt, completion, model)
    return script


def build(store: Store, budgets: dict, script=None, latency_s: float = 0.0, groq: FakeProvider | None = None):
    config = make_config(budgets=budgets)
    groq = groq or FakeProvider("groq", script or answer(), latency_s=latency_s)
    app = create_app(config, providers={"groq": groq, "gemini": FakeProvider("gemini")}, store=store, sleep=no_sleep)
    return app, config


def spent(store: Store, app_id: str) -> float:
    return store.app_spend(app_id, iso(datetime.now(UTC))[:10])


async def post_many(app, n: int, app_id: str, **body) -> list[httpx.Response]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        return await asyncio.gather(*[
            client.post("/v1/chat/completions", headers={"X-App-Id": app_id},
                        json={"model": "strong", "messages": MSG, **body}) for _ in range(n)])


class WatchedStore(Store):
    """Records the app's logged spend plus active reservations after every reservation."""

    def __init__(self, path):
        super().__init__(path)
        self.peaks: list[float] = []

    def reserve_budget(self, request_id, app_id, day, limit, estimate, now, expires_at):
        held = super().reserve_budget(request_id, app_id, day, limit, estimate, now, expires_at)
        reserved = sum(r["amount_usd"] for r in self.reservations(app_id))
        self.peaks.append(self.app_spend(app_id, day) + reserved)
        return held


# ── concurrency ─────────────────────────────────────────────────────


async def test_concurrent_requests_cannot_share_the_same_budget(tmp_path):
    limit = 0.0005
    store = WatchedStore(tmp_path / "gateway.db")
    app, config = build(store, {"apps": {"tight": {"daily_usd": limit}}}, latency_s=0.2)
    # mode "strong" may call large, then fall back to small: the reservation prices the dearer one.
    estimate = reservation_estimate([config.models["large"], config.models["small"]], MSG, {"max_tokens": 100},
                                    config.budgets.reserve_output_tokens)
    expected = math.ceil(limit / estimate)  # full reservations, then one capped at what is left
    assert 1 < expected < 12

    responses = await post_many(app, 12, "tight", max_tokens=100)

    admitted = [r for r in responses if r.status_code == 200]
    refused = [r for r in responses if r.status_code == 429]
    assert len(admitted) == expected and len(refused) == 12 - expected
    assert all(r.json()["error"]["code"] == "budget_exceeded" for r in refused)
    assert all("reserved by requests in progress" in r.json()["error"]["message"] for r in refused)
    assert max(store.peaks) <= limit + 1e-12  # logged spend + active reservations never passed the limit
    assert spent(store, "tight") <= limit  # each real cost ($0.000039) is below its reservation
    assert store.reservations() == []  # every reservation settled
    assert len(store.query("SELECT id FROM requests WHERE app_id = 'tight'")) == 12  # each request logged once


async def test_the_audit_reproduction_admits_one_request_instead_of_ten(tmp_path):
    # $0.001 budget, 10 concurrent requests costing $0.00165 each: before the fix all 10 ran ($0.0165).
    store = Store(tmp_path / "gateway.db")
    app, _ = build(store, {"apps": {"tiny": {"daily_usd": 0.001}}}, script=answer(1000, 2000), latency_s=0.2)
    responses = await post_many(app, 10, "tiny")
    assert sorted(r.status_code for r in responses) == [200] + [429] * 9
    # The one admitted request reserved all that was left; its real cost is over it, the same one-request
    # overshoot a lone request always had.
    assert spent(store, "tiny") == pytest.approx(0.00165)
    assert store.reservations() == []


def test_reservations_are_atomic_across_connections_to_one_database(tmp_path):
    path = tmp_path / "gateway.db"
    stores = [Store(path), Store(path)]  # e.g. two gateway workers on one SQLite file
    limit, estimate = 0.001, 0.00013
    now, expires = iso(datetime.now(UTC)), iso(datetime.now(UTC) + timedelta(minutes=5))
    barrier = threading.Barrier(20)
    results = []

    def reserve(i: int) -> None:
        barrier.wait()
        results.append(stores[i % 2].reserve_budget(f"r{i}", "app", now[:10], limit, estimate, now, expires))

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    admitted = [r for r in results if r.admitted]
    assert len(admitted) == math.ceil(limit / estimate)  # 7 full reservations and one for the rest
    assert sum(r.amount for r in admitted) == pytest.approx(limit)
    assert sum(r["amount_usd"] for r in stores[0].reservations("app")) == pytest.approx(limit)


# ── settlement: success, failure, cancellation ──────────────────────


async def test_a_successful_request_holds_a_reservation_then_settles_it_once(tmp_path):
    store = Store(tmp_path / "gateway.db")
    seen = []

    def script(model, messages, params):
        seen.append(store.reservations("solo"))  # during the provider call
        return Completion("ok", "stop", 10, 50, model)

    app, _ = build(store, {"apps": {"solo": {"daily_usd": 1.0}}}, script=script)
    [response] = await post_many(app, 1, "solo")
    assert response.status_code == 200
    assert len(seen) == 1 and len(seen[0]) == 1 and seen[0][0]["amount_usd"] > 0
    assert store.reservations() == []
    assert spent(store, "solo") == pytest.approx((10 * 0.15 + 50 * 0.75) / 1e6)  # the real cost, not the estimate


@pytest.mark.parametrize("stream", [False, True])
async def test_a_failed_request_releases_its_reservation(tmp_path, stream):
    def down(model, messages, params):
        raise ProviderError("upstream down", kind=OVERLOADED, status=503)

    store = Store(tmp_path / "gateway.db")
    app, _ = build(store, {"apps": {"tight": {"daily_usd": 0.0001}}}, script=down)
    [failed] = await post_many(app, 1, "tight", stream=stream)
    assert failed.status_code == 503
    assert store.reservations() == [] and spent(store, "tight") == 0.0
    # The budget wasn't used up by the failure: the next request is admitted (and fails the same way).
    [again] = await post_many(app, 1, "tight", stream=stream)
    assert again.status_code == 503


async def test_a_cancelled_request_releases_its_reservation(tmp_path):
    store = Store(tmp_path / "gateway.db")
    slow = FakeProvider("groq", answer(), latency_s=5.0)
    app, _ = build(store, {"apps": {"solo": {"daily_usd": 1.0}}}, groq=slow)
    request = ChatRequest(model="cheap", messages=MSG)
    task = asyncio.create_task(app.state.gateway.chat(request, RequestContext(app_id="solo")))
    await asyncio.sleep(0.1)
    assert len(store.reservations("solo")) == 1  # held while the provider works
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.reservations() == []
    [row] = store.query("SELECT status, error_type FROM requests")
    assert (row["status"], row["error_type"]) == ("error", "cancelled")


async def test_a_cancelled_stream_releases_its_reservation(tmp_path):
    store = Store(tmp_path / "gateway.db")
    app, _ = build(store, {"apps": {"solo": {"daily_usd": 1.0}}})
    request = ChatRequest(model="cheap", messages=MSG, stream=True)
    events, _ = await app.state.gateway.stream(request, RequestContext(app_id="solo"))
    await anext(events)
    assert len(store.reservations("solo")) == 1
    await events.aclose()  # the client went away
    assert store.reservations() == []
    assert store.query("SELECT error_type FROM requests")[0]["error_type"] == "cancelled"


# ── abandoned and double settlement ─────────────────────────────────


def test_expired_reservations_are_cleaned_up_and_free_the_budget(tmp_path):
    path = tmp_path / "gateway.db"
    t0 = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
    day = iso(t0)[:10]
    crashed = Store(path)
    held = crashed.reserve_budget("lost", "app", day, 0.001, 0.001, iso(t0), iso(t0 + timedelta(seconds=60)))
    assert held.admitted and held.amount == pytest.approx(0.001)
    crashed.close()  # the process died without settling

    restarted = Store(path)  # the reservation survives the restart and still counts until it expires
    blocked = restarted.reserve_budget("next", "app", day, 0.001, 0.0001, iso(t0 + timedelta(seconds=30)),
                                       iso(t0 + timedelta(seconds=90)))
    assert not blocked.admitted and blocked.reserved == pytest.approx(0.001)
    freed = restarted.reserve_budget("later", "app", day, 0.001, 0.0001, iso(t0 + timedelta(seconds=61)),
                                     iso(t0 + timedelta(seconds=121)))
    assert freed.admitted and freed.reserved == 0.0
    assert [r["request_id"] for r in restarted.reservations()] == ["later"]


def test_a_reservation_cannot_be_settled_twice(tmp_path):
    store = Store(tmp_path / "gateway.db")
    now = datetime.now(UTC)
    held = store.reserve_budget("req-1", "app", iso(now)[:10], 1.0, 0.5, iso(now), iso(now + timedelta(minutes=5)))
    assert held.admitted
    record = RequestRecord(id="req-1", ts=iso(now), app_id="app", status="ok", cost_usd=0.02)
    store.log_request(record)
    assert store.reservations() == [] and spent(store, "app") == pytest.approx(0.02)
    with pytest.raises(sqlite3.IntegrityError):
        store.log_request(record)  # a second settlement is refused and changes nothing
    assert spent(store, "app") == pytest.approx(0.02)
    other = store.reserve_budget("req-2", "app", iso(now)[:10], 1.0, 0.1, iso(now), iso(now + timedelta(minutes=5)))
    assert other.admitted and other.spent == pytest.approx(0.02) and other.reserved == 0.0  # no transaction left open


async def test_a_storage_failure_while_reserving_fails_closed(tmp_path):
    class BrokenStore(Store):
        def reserve_budget(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

    groq = FakeProvider("groq", answer())
    app, _ = build(BrokenStore(tmp_path / "gateway.db"), {"apps": {"solo": {"daily_usd": 1.0}}}, groq=groq)
    [response] = await post_many(app, 1, "solo")
    assert response.status_code == 500 and response.json()["error"]["code"] == "internal_error"
    assert groq.calls == []  # no provider call without a reservation


# ── unchanged behaviour ─────────────────────────────────────────────


async def test_apps_without_a_limit_do_not_reserve(tmp_path):
    store = Store(tmp_path / "gateway.db")
    seen = []

    def script(model, messages, params):
        seen.append(store.reservations())
        return Completion("ok", "stop", 10, 50, model)

    app, _ = build(store, {"default_daily_usd": None, "apps": {}}, script=script)
    responses = await post_many(app, 3, "anyone")
    assert [r.status_code for r in responses] == [200, 200, 200] and seen == [[], [], []]


async def test_sequential_requests_keep_the_existing_limit_and_message(tmp_path):
    store = Store(tmp_path / "gateway.db")
    app, _ = build(store, {"apps": {"solo": {"daily_usd": 0.00005}}})  # one strong call costs $0.000039
    first, second, third = [(await post_many(app, 1, "solo"))[0] for _ in range(3)]
    assert (first.status_code, second.status_code, third.status_code) == (200, 200, 429)
    message = third.json()["error"]["message"]
    assert message == "daily budget for app 'solo' is used up: $0.0001 of $0.00 (resets at 00:00 UTC)"
    assert int(third.headers["retry-after"]) > 0
