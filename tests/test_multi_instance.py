"""Multi-instance validation tests: Two gateway instances sharing state and storage."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.circuit import CircuitState
from gateway.config import GatewayConfig, RedisConfig, load_config
from gateway.monitoring.store import Store
from gateway.providers.base import (
    OVERLOADED,
    QUOTA_EXHAUSTED,
    Completion,
    Provider,
    ProviderError,
)
from gateway.state.redis_backend import RedisStateBackend

REDIS_URL = os.environ.get("GATEWAY_REDIS_URL", "redis://127.0.0.1:56379/2")


class MockProvider(Provider):
    def __init__(self, answer: str = "ok"):
        self.answer = answer
        self.call_count = 0
        self.fail_with = None

    async def complete(self, model, messages, params):
        self.call_count += 1
        if self.fail_with:
            raise self.fail_with
        return Completion(model=model, text=self.answer, prompt_tokens=10, completion_tokens=5)


@pytest.fixture
def multi_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEYS", "free-app:free-key,test-app:test-key,persist-app:persist-key")
    try:
        import redis
        client = redis.Redis.from_url(REDIS_URL, socket_timeout=1.0)
        client.ping()
        client.flushdb()
    except Exception:
        pytest.skip("Redis not reachable at " + REDIS_URL)

    config_dict = load_config().model_dump()
    # Configure tight limits for testing
    config_dict["models"]["m1"] = {
        "provider": "p1",
        "model": "model-1",
        "price_per_1m": {"input": 10.0, "output": 20.0},
        "limits": {"rpm": 2, "tpm": 1000},
        "params": {},
    }
    config_dict["models"]["m_unlimited"] = {
        "provider": "p1",
        "model": "model-unlimited",
        "price_per_1m": {},
        "limits": {"rpm": 0, "tpm": 0},
        "params": {},
    }
    config_dict["providers"]["p1"] = {"base_url": "http://mock", "api_key_env": "K", "timeout_s": 5.0}
    config_dict["prompts"]["default"] = "You are a helpful assistant."
    config_dict["releases"]["r1"] = {"cheap": "m1", "strong": "m1", "prompt": "default", "note": ""}
    config_dict["releases"]["r_unlimited"] = {
        "cheap": "m_unlimited",
        "strong": "m_unlimited",
        "prompt": "default",
        "note": "",
    }
    config_dict["deployment"]["active"] = "r1"
    config_dict["deployment"]["canary"] = None
    config_dict["budgets"]["apps"]["test-app"] = {"daily_usd": 0.001}
    config_dict["circuit_breaker"] = {"failure_threshold": 2, "recovery_timeout_s": 1.0, "half_open_probes": 1}

    config = GatewayConfig.model_validate(config_dict)
    store = Store(tmp_path / "shared.db")

    redis_cfg = RedisConfig(url=REDIS_URL)
    state1 = RedisStateBackend(redis_cfg)
    state2 = RedisStateBackend(redis_cfg)

    p1 = MockProvider("hello from p1")
    p2 = MockProvider("hello from p2")

    app1 = create_app(config, providers={"p1": p1}, store=store, state=state1)
    app2 = create_app(config, providers={"p1": p2}, store=store, state=state2)

    client1 = TestClient(app1)
    client2 = TestClient(app2)

    yield {
        "client1": client1,
        "client2": client2,
        "store": store,
        "p1": p1,
        "p2": p2,
        "state1": state1,
        "state2": state2,
        "config": config,
    }

    try:
        store.close()
        state1.close()
        state2.close()
    except Exception:
        pass


def test_shared_rate_limits_across_instances(multi_setup):
    c1 = multi_setup["client1"]
    c2 = multi_setup["client2"]
    headers = {"Authorization": "Bearer free-key"}

    # Request 1 via instance 1: success (1/2 RPM)
    r1 = c1.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                 headers=headers)
    assert r1.status_code == 200

    # Request 2 via instance 2: success (2/2 RPM)
    r2 = c2.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                 headers=headers)
    assert r2.status_code == 200

    # Request 3 via instance 1: rate limited! (RPM exhausted across both instances)
    r3 = c1.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                 headers=headers)
    assert r3.status_code == 429
    assert "rate limit" in r3.json()["error"]["message"]


def test_shared_provider_cooldown_across_instances(multi_setup):
    c1 = multi_setup["client1"]
    c2 = multi_setup["client2"]
    p1 = multi_setup["p1"]
    headers = {"Authorization": "Bearer free-key"}

    # Simulate quota exhaustion on instance 1
    p1.fail_with = ProviderError("quota exceeded", kind=QUOTA_EXHAUSTED, status=429, retry_after=30.0)

    r1 = c1.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                 headers=headers)
    assert r1.status_code == 429

    # Instance 2 is now immediately in cooldown without ever touching its provider!
    r2 = c2.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                 headers=headers)
    assert r2.status_code == 429
    assert multi_setup["p2"].call_count == 0  # skipped entirely!


def test_shared_circuit_breaker_trips_and_recovers_across_instances(multi_setup):
    c1 = multi_setup["client1"]
    c2 = multi_setup["client2"]
    p1 = multi_setup["p1"]
    p2 = multi_setup["p2"]
    headers = {"Authorization": "Bearer free-key"}

    # Both fail with 503
    p1.fail_with = ProviderError("server overloaded", kind=OVERLOADED, status=503)
    p2.fail_with = ProviderError("server overloaded", kind=OVERLOADED, status=503)

    # 1st failure on instance 1
    c1.post("/v1/chat/completions", json={"model": "m_unlimited", "messages": [{"role": "user", "content": "hi"}]},
            headers=headers)

    # 2nd failure on instance 2 -> Trips shared circuit to OPEN!
    c2.post("/v1/chat/completions", json={"model": "m_unlimited", "messages": [{"role": "user", "content": "hi"}]},
            headers=headers)

    cb1 = multi_setup["state1"].get_circuit_breaker("p1", multi_setup["config"].circuit_breaker)
    cb2 = multi_setup["state2"].get_circuit_breaker("p1", multi_setup["config"].circuit_breaker)
    assert cb1.state == CircuitState.OPEN
    assert cb2.state == CircuitState.OPEN

    # Recovery after timeout
    time.sleep(1.1)
    p1.fail_with = None
    p2.fail_with = None

    # Probe on instance 1 succeeds -> closes circuit for both!
    r_probe = c1.post("/v1/chat/completions", json={"model": "m_unlimited", "messages": [{"role": "user", "content": "hi"}]},
                      headers=headers)
    assert r_probe.status_code == 200
    assert cb2.state == CircuitState.CLOSED


def test_shared_storage_and_instance_restart(multi_setup, tmp_path):
    c1 = multi_setup["client1"]
    store = multi_setup["store"]
    headers = {"Authorization": "Bearer persist-key"}

    r = c1.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]},
                headers=headers)
    assert r.status_code == 200
    req_id = r.json()["id"]

    # Verify store has the request logged
    assert store.request_app(req_id) == "persist-app"

    # Simulate instance 1 restart (new app instance sharing store & redis)
    restarted_state = RedisStateBackend(RedisConfig(url=REDIS_URL))
    restarted_app = create_app(multi_setup["config"], providers={"p1": multi_setup["p1"]},
                               store=store, state=restarted_state)
    restarted_client = TestClient(restarted_app)

    # Recheck readiness
    ready = restarted_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["database"]["ok"] is True
    restarted_state.close()
