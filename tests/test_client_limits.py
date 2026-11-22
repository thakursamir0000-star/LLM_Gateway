"""Per-client request limits (config ``client_limits``): requests per minute and requests in flight per app.

The limiter tests run on the in-memory backend with a fake clock; the shared-state tests need the Redis at
$GATEWAY_REDIS_URL (default redis://127.0.0.1:56379/1) and are skipped, with the reason, when it isn't running.
"""

from __future__ import annotations

import asyncio
import os
import random

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import RedisConfig
from gateway.monitoring.store import Store
from gateway.providers.base import Completion
from gateway.providers.fake import FakeProvider
from gateway.routing import RulesRouter
from gateway.state.memory import InMemoryClientLimiter, InMemoryStateBackend
from gateway.state.redis_backend import RedisStateBackend
from tests.conftest import make_config

EASY = {"messages": [{"role": "user", "content": "Hello"}]}
REDIS_URL = os.environ.get("GATEWAY_REDIS_URL", "redis://127.0.0.1:56379/1")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ── the limiter itself ──────────────────────────────────────────────


def test_rpm_refuses_with_retry_after_and_resets_after_the_window():
    clock = Clock()
    limiter = InMemoryClientLimiter(clock=clock)
    for _ in range(3):
        assert limiter.admit("web", 3, 0, 600).admitted
    clock.now += 20
    refused = limiter.admit("web", 3, 0, 600)
    assert not refused.admitted and refused.reason == "rpm"
    assert refused.retry_after == pytest.approx(40.0)  # the oldest of the three leaves the window in 40 s
    clock.now += 40
    assert limiter.admit("web", 3, 0, 600).admitted


def test_clients_are_isolated():
    limiter = InMemoryClientLimiter(clock=Clock())
    assert limiter.admit("web", 1, 0, 600).admitted
    assert not limiter.admit("web", 1, 0, 600).admitted
    assert limiter.admit("batch", 1, 0, 600).admitted  # another client's allowance is untouched


def test_concurrency_slots_are_freed_on_release_and_by_lease_expiry():
    clock = Clock()
    limiter = InMemoryClientLimiter(clock=clock)
    first = limiter.admit("web", 0, 2, 30)
    second = limiter.admit("web", 0, 2, 30)
    refused = limiter.admit("web", 0, 2, 30)
    assert first.admitted and second.admitted and not refused.admitted and refused.reason == "concurrency"
    limiter.release("web", first.lease_id)
    assert limiter.admit("web", 0, 2, 30).admitted
    clock.now += 31  # a crashed request's slots come back once its lease expires
    assert limiter.admit("web", 0, 2, 30).admitted


def test_refused_requests_do_not_use_up_the_minute():
    clock = Clock()
    limiter = InMemoryClientLimiter(clock=clock)
    held = limiter.admit("web", 5, 1, 600)
    for _ in range(10):
        assert not limiter.admit("web", 5, 1, 600).admitted  # refused for concurrency, not counted
    limiter.release("web", held.lease_id)
    for _ in range(4):
        assert limiter.admit("web", 5, 0, 600).admitted


# ── through the gateway ────────────────────────────────────────────


def gateway(monkeypatch, tmp_path, limits, state=None, provider=None, keys="web:key-web,batch:key-batch"):
    monkeypatch.setenv("TEST_GATEWAY_KEYS", keys)
    config = make_config(auth={"allow_anonymous": False}, client_limits=limits)
    providers = {"groq": provider or FakeProvider("groq"), "gemini": FakeProvider("gemini")}
    store = Store(tmp_path / "gateway.db")
    app = create_app(config, providers=providers, store=store, router=RulesRouter(), rng=random.Random(0),
                     state=state or InMemoryStateBackend())
    return app, store


def post(client, key, stream=False):
    return client.post("/v1/chat/completions", json={**EASY, "stream": stream},
                       headers={"Authorization": f"Bearer {key}"})


def test_gateway_answers_429_with_retry_after_per_client(monkeypatch, tmp_path):
    app, store = gateway(monkeypatch, tmp_path, {"default": {"rpm": 2}, "apps": {"batch": {"rpm": 5}}})
    with TestClient(app) as client:
        assert [post(client, "key-web").status_code for _ in range(2)] == [200, 200]
        refused = post(client, "key-web")
        assert [post(client, "key-batch").status_code for _ in range(5)] == [200] * 5  # its own, larger limit
    rows = store.query("SELECT status, http_status, error_type FROM requests WHERE app_id = 'web'")
    store.close()
    assert refused.status_code == 429
    error = refused.json()["error"]
    assert (error["code"], error["type"]) == ("client_rate_limited", "rate_limit_error")
    assert 1 <= int(refused.headers["retry-after"]) <= 60
    assert sorted(r["status"] for r in rows) == ["ok", "ok", "rejected"]  # logged once, as rejected


def test_invalid_and_revoked_keys_get_401_and_use_no_allowance(monkeypatch, tmp_path):
    state = InMemoryStateBackend()
    # "key-old" was web's key; it was revoked (removed from the key list) and the gateway restarted.
    app, store = gateway(monkeypatch, tmp_path, {"default": {"rpm": 1}}, state=state, keys="web:key-web")
    with TestClient(app) as client:
        for key in ("wrong", "key-old", "key-old"):
            response = post(client, key)
            assert response.status_code == 401 and response.json()["error"]["code"] == "invalid_api_key"
        assert post(client, "key-web").status_code == 200  # the client's one request a minute is still there
        assert post(client, "key-web").status_code == 429
    store.close()


def test_streams_hold_a_concurrency_slot_until_they_end(monkeypatch, tmp_path):
    # TestClient reads a streamed body to the end before returning, so the open stream is driven directly.
    from gateway.schemas import ChatRequest
    from gateway.service import GatewayError, RequestContext

    state = InMemoryStateBackend()
    app, store = gateway(monkeypatch, tmp_path, {"default": {"concurrency": 1}}, state=state)
    service = app.state.gateway

    async def scenario():
        request = ChatRequest.model_validate({**EASY, "stream": True})
        events, _ = await service.stream(request, RequestContext(app_id="web"))
        assert len(state.clients._leases["web"]) == 1  # the open stream holds the slot
        with pytest.raises(GatewayError) as busy:
            await service.stream(request, RequestContext(app_id="web"))
        assert busy.value.code == "client_rate_limited"
        chunks = [event async for event in events]
        assert chunks and state.clients._leases["web"] == {}  # released when the stream ended

        events, _ = await service.stream(request, RequestContext(app_id="web"))
        await events.__anext__()
        await events.aclose()  # a client that disconnects mid-stream frees its slot too
        assert state.clients._leases["web"] == {}

    asyncio.run(scenario())
    store.close()


def test_a_failed_request_releases_its_slot(monkeypatch, tmp_path):
    def broken(model, messages, params):
        raise RuntimeError("boom")

    state = InMemoryStateBackend()
    app, store = gateway(monkeypatch, tmp_path, {"default": {"concurrency": 1}}, state=state,
                         provider=FakeProvider("groq", broken))
    with TestClient(app) as client:
        assert post(client, "key-web").status_code >= 500
        assert state.clients._leases["web"] == {}
    store.close()


def test_no_limits_configured_means_no_state_calls(monkeypatch, tmp_path):
    state = InMemoryStateBackend()
    app, store = gateway(monkeypatch, tmp_path, {}, state=state)
    with TestClient(app) as client:
        assert post(client, "key-web").status_code == 200
    store.close()
    assert state.clients._recent == {} and state.clients._leases == {}


# ── shared through Redis ────────────────────────────────────────────


@pytest.fixture
def redis_url():
    redis = pytest.importorskip("redis")
    try:
        client = redis.Redis.from_url(REDIS_URL, socket_timeout=1.0)
        client.ping()
        client.flushdb()
    except Exception:
        pytest.skip("Redis not available at " + REDIS_URL)
    yield REDIS_URL
    client.flushdb()


def test_two_replicas_share_one_client_allowance(redis_url):
    one = RedisStateBackend(RedisConfig(url=redis_url))
    two = RedisStateBackend(RedisConfig(url=redis_url))
    try:
        assert one.admit_client("web", 3, 0, 600).admitted
        assert two.admit_client("web", 3, 0, 600).admitted
        assert one.admit_client("web", 3, 0, 600).admitted
        refused = two.admit_client("web", 3, 0, 600)
        assert not refused.admitted and refused.reason == "rpm" and 0 < refused.retry_after <= 60
        assert two.admit_client("batch", 3, 0, 600).admitted  # other clients unaffected

        held = one.admit_client("api", 0, 1, 600)
        assert held.admitted and two.admit_client("api", 0, 1, 600).reason == "concurrency"
        two.release_client("api", held.lease_id)  # released by either replica
        assert one.admit_client("api", 0, 1, 600).admitted
    finally:
        one.close()
        two.close()


def test_concurrent_admissions_never_exceed_the_limit(redis_url):
    backends = [RedisStateBackend(RedisConfig(url=redis_url)) for _ in range(4)]

    async def burst():
        return await asyncio.gather(*(asyncio.to_thread(backends[i % 4].admit_client, "web", 10, 0, 600)
                                      for i in range(40)))

    try:
        admitted = sum(a.admitted for a in asyncio.run(burst()))
        assert admitted == 10
    finally:
        for backend in backends:
            backend.close()


def test_lease_expiry_frees_redis_slots(redis_url):
    backend = RedisStateBackend(RedisConfig(url=redis_url))
    try:
        assert backend.admit_client("web", 0, 1, 0.2).admitted
        assert not backend.admit_client("web", 0, 1, 0.2).admitted
        import time

        time.sleep(0.3)
        assert backend.admit_client("web", 0, 1, 0.2).admitted
    finally:
        backend.close()


# ── Redis outage ────────────────────────────────────────────────────

DOWN = "redis://127.0.0.1:1/0"


def test_outage_fail_closed_refuses_limited_clients():
    state = RedisStateBackend(RedisConfig(url=DOWN, socket_timeout_s=0.2, connect_timeout_s=0.2))
    refused = state.admit_client("web", 10, 0, 600)
    assert not refused.admitted and refused.reason == "state_unavailable" and refused.retry_after == 5.0


def test_outage_local_share_gives_each_replica_its_part():
    state = RedisStateBackend(RedisConfig(url=DOWN, socket_timeout_s=0.2, connect_timeout_s=0.2,
                                          outage_policy="local_share", replicas=2))
    results = [state.admit_client("web", 6, 0, 600).admitted for _ in range(4)]
    assert results == [True, True, True, False]  # 6 per minute over 2 replicas: 3 here


def test_gateway_outage_answers_429_without_calling_the_provider(monkeypatch, tmp_path):
    groq = FakeProvider("groq", lambda model, messages, params: Completion("ok", "stop", 5, 5, model))
    state = RedisStateBackend(RedisConfig(url=DOWN, socket_timeout_s=0.2, connect_timeout_s=0.2))
    app, store = gateway(monkeypatch, tmp_path, {"default": {"rpm": 10}}, state=state, provider=groq)
    with TestClient(app) as client:
        response = post(client, "key-web")
    store.close()
    assert response.status_code == 429 and response.json()["error"]["code"] == "client_rate_limited"
    assert groq.calls == []
