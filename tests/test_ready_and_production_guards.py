"""Tests for /ready endpoint, /health status reporting, and production configuration guards."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import load_config
from gateway.state.base import StateBackend


class FakeDegradedState(StateBackend):
    def cooldown_left(self, alias: str) -> float:
        return 0.0

    def set_cooldown(self, alias: str, duration_s: float) -> None:
        pass

    def get_rate_limiter(self, alias, rpm, tpm):
        from gateway.ratelimit import RateLimiter
        return RateLimiter()

    def get_circuit_breaker(self, name, config):
        from gateway.circuit import CircuitBreaker
        return CircuitBreaker(name, config)

    def is_ready(self) -> tuple[bool, str]:
        return False, "redis connection refused"

    def status(self) -> dict:
        return {"backend": "redis", "ready": False, "degraded": True, "details": "down"}

    def close(self) -> None:
        pass


def test_ready_endpoint_healthy(tmp_path):
    client = TestClient(create_app(load_config()))
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert data["database"]["ok"] is True


def test_ready_endpoint_returns_503_when_state_backend_unready():
    degraded = FakeDegradedState()
    client = TestClient(create_app(load_config(), state=degraded))
    r = client.get("/ready")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "not_ready"
    assert data["state_backend"]["ready"] is False


def test_health_reports_degraded_when_state_degraded():
    degraded = FakeDegradedState()
    client = TestClient(create_app(load_config(), state=degraded))
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "degraded"
    assert data["state"]["degraded"] is True


def test_production_guard_rejects_sqlite(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENV", "production")
    monkeypatch.setenv("GATEWAY_API_KEYS", "app:secret-key")
    monkeypatch.setenv("GATEWAY_ADMIN_KEY", "admin-secret")
    monkeypatch.setenv("GATEWAY_STATE_BACKEND", "redis")
    monkeypatch.delenv("GATEWAY_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="PostgreSQL database"):
        create_app(load_config())


def test_production_guard_rejects_anonymous_access(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENV", "production")
    monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://user:pass@localhost:5432/gw")
    monkeypatch.setenv("GATEWAY_ALLOW_ANONYMOUS", "1")
    monkeypatch.setenv("GATEWAY_API_KEYS", "app:key")
    monkeypatch.setenv("GATEWAY_ADMIN_KEY", "admin-key")
    monkeypatch.setenv("GATEWAY_STATE_BACKEND", "redis")

    with pytest.raises(RuntimeError, match="anonymous access"):
        create_app(load_config())


def test_production_guard_rejects_missing_api_keys(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENV", "production")
    monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://user:pass@localhost:5432/gw")
    monkeypatch.setenv("GATEWAY_ALLOW_ANONYMOUS", "0")
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.setenv("GATEWAY_ADMIN_KEY", "admin-key")
    monkeypatch.setenv("GATEWAY_STATE_BACKEND", "redis")

    with pytest.raises(RuntimeError, match="client API keys"):
        create_app(load_config())


def test_production_guard_rejects_missing_admin_key(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENV", "production")
    monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://user:pass@localhost:5432/gw")
    monkeypatch.setenv("GATEWAY_ALLOW_ANONYMOUS", "0")
    monkeypatch.setenv("GATEWAY_API_KEYS", "app:key")
    monkeypatch.delenv("GATEWAY_ADMIN_KEY", raising=False)
    monkeypatch.setenv("GATEWAY_STATE_BACKEND", "redis")

    with pytest.raises(RuntimeError, match="admin key"):
        create_app(load_config())


def test_production_guard_rejects_memory_state(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENV", "production")
    monkeypatch.setenv("GATEWAY_DATABASE_URL", "postgresql://user:pass@localhost:5432/gw")
    monkeypatch.setenv("GATEWAY_ALLOW_ANONYMOUS", "0")
    monkeypatch.setenv("GATEWAY_API_KEYS", "app:key")
    monkeypatch.setenv("GATEWAY_ADMIN_KEY", "admin-key")
    monkeypatch.setenv("GATEWAY_STATE_BACKEND", "memory")

    with pytest.raises(RuntimeError, match="Redis state backend"):
        create_app(load_config())
