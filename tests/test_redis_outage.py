"""Redis outage policy: while Redis is unreachable, provider rate limits are never multiplied by the replicas.

These tests need no Redis: the backend points at a closed port, which is what an outage looks like to it.
The live drill (stop Redis, watch /ready and requests, restart, watch recovery) is scripts/redis_outage_drill.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import RedisConfig, load_config
from gateway.monitoring.store import Store
from gateway.providers.base import Completion
from gateway.providers.fake import FakeProvider
from gateway.ratelimit import RateLimited
from gateway.state.redis_backend import RedisStateBackend
from tests.conftest import make_config

DOWN = "redis://127.0.0.1:1/0"  # nothing listens on port 1


def backend(**policy) -> RedisStateBackend:
    return RedisStateBackend(RedisConfig(url=DOWN, socket_timeout_s=0.2, connect_timeout_s=0.2, **policy))


async def test_fail_closed_refuses_rate_limited_models_and_reports_not_ready():
    state = backend()
    with pytest.raises(RateLimited) as refused:
        await state.get_rate_limiter("large", 30, 8000).acquire(100, max_wait=0)
    assert refused.value.wait_s == 5.0
    assert state.is_ready()[0] is False
    status = state.status()
    assert status["degraded"] is True and status["outage_policy"] == "fail_closed"


async def test_models_without_limits_keep_working_during_an_outage():
    reservation = await backend().get_rate_limiter("free", 0, 0).acquire(100, max_wait=0)
    assert reservation.tokens == 100


async def test_local_share_gives_each_replica_only_its_part_of_the_limit():
    state = backend(outage_policy="local_share", replicas=3)
    limiter = state.get_rate_limiter("large", 30, 0)
    for _ in range(10):  # 30 requests per minute shared by 3 replicas: 10 here
        await limiter.acquire(10, max_wait=0)
    with pytest.raises(RateLimited):
        await limiter.acquire(10, max_wait=0)
    shared = state.outage_rate_limiter("large", 30, 9000)
    assert (shared.rpm, shared.tpm) == (10, 3000)  # never the full 30 / 9000 on every replica


def test_gateway_answers_429_while_shared_limits_cannot_be_enforced(tmp_path):
    config = make_config(models={
        "small": {"provider": "groq", "model": "openai/gpt-oss-20b", "limits": {"rpm": 30}},
        "large": {"provider": "groq", "model": "openai/gpt-oss-120b", "limits": {"rpm": 30}},
        "gemma": {"provider": "gemini", "model": "gemma-4-31b-it"},
    })
    groq = FakeProvider("groq", lambda model, messages, params: Completion("ok", "stop", 5, 5, model))
    # Its own SQLite store, so the test never follows a GATEWAY_DATABASE_URL set in the environment (CI sets one).
    store = Store(tmp_path / "gateway.db")
    app = create_app(config, providers={"groq": groq, "gemini": FakeProvider("gemini")}, state=backend(),
                     store=store)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        assert client.get("/health").json()["status"] == "degraded"
        response = client.post("/v1/chat/completions", headers={"X-App-Id": "app"},
                               json={"model": "strong", "messages": [{"role": "user", "content": "hi"}]})
    store.close()
    assert response.status_code == 429 and response.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(response.headers["retry-after"]) >= 5
    assert groq.calls == []  # nothing reached the provider without shared enforcement


def test_outage_policy_and_replicas_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("GATEWAY_REDIS_OUTAGE_POLICY", "local_share")
    monkeypatch.setenv("GATEWAY_REPLICAS", "2")
    redis_config = load_config().state.redis
    assert (redis_config.outage_policy, redis_config.replicas) == ("local_share", 2)
    monkeypatch.setenv("GATEWAY_REDIS_OUTAGE_POLICY", "full_limits")
    with pytest.raises(ValueError):
        load_config()
