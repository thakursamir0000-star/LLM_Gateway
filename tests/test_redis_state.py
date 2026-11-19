"""Integration tests for Redis distributed state: rate limits, cooldowns, circuit breakers, and outage handling."""

from __future__ import annotations

import os
import time

import pytest
import redis

from gateway.circuit import CircuitState
from gateway.config import CircuitBreakerConfig, RedisConfig
from gateway.ratelimit import RateLimited
from gateway.state.redis_backend import RedisStateBackend

REDIS_URL = os.environ.get("GATEWAY_REDIS_URL", "redis://127.0.0.1:56379/1")


@pytest.fixture
def redis_backend():
    try:
        client = redis.Redis.from_url(REDIS_URL, socket_timeout=1.0)
        client.ping()
        client.flushdb()
    except Exception:
        pytest.skip("Redis container not available at " + REDIS_URL)

    config = RedisConfig(url=REDIS_URL, socket_timeout_s=1.0, connect_timeout_s=1.0)
    backend = RedisStateBackend(config)
    yield backend
    try:
        backend.client.flushdb()
        backend.close()
    except Exception:
        pass


def test_redis_cooldown_shared_and_monotonic(redis_backend):
    backend1 = redis_backend
    backend2 = RedisStateBackend(redis_backend.config)
    try:
        alias = "test-model"
        assert backend1.cooldown_left(alias) == 0.0
        assert backend2.cooldown_left(alias) == 0.0

        # Set 30s cooldown from instance 1
        backend1.set_cooldown(alias, 30.0)

        # Instance 2 immediately sees it
        left2 = backend2.cooldown_left(alias)
        assert 25.0 <= left2 <= 30.0

        # A later shorter cooldown (10s) must NOT shorten the existing 30s cooldown
        backend2.set_cooldown(alias, 10.0)
        left_after = backend1.cooldown_left(alias)
        assert left_after > 20.0

        # A longer cooldown (60s) DOES extend it
        backend1.set_cooldown(alias, 60.0)
        left_extended = backend2.cooldown_left(alias)
        assert 55.0 <= left_extended <= 60.0
    finally:
        backend2.close()


def test_redis_rate_limiter_rpm_and_tpm(redis_backend):
    limiter = redis_backend.get_rate_limiter("rpm-test", rpm=2, tpm=1000)

    # First reservation fits
    res1 = limiter.reserve(100)
    assert res1.tokens == 100
    assert limiter.wait_time(100) == 0.0

    # Second reservation fits (count = 2)
    res2 = limiter.reserve(200)
    assert res2.tokens == 200

    # Third reservation hits RPM limit (rpm=2)
    wait = limiter.wait_time(100)
    assert 55.0 <= wait <= 60.0

    # Settle res2 with fewer tokens
    limiter.settle(res2, 50)
    assert res2.tokens == 50


@pytest.mark.asyncio
async def test_redis_rate_limiter_shared_across_instances(redis_backend):
    backend2 = RedisStateBackend(redis_backend.config)
    try:
        limiter1 = redis_backend.get_rate_limiter("shared-rpm", rpm=2, tpm=0)
        limiter2 = backend2.get_rate_limiter("shared-rpm", rpm=2, tpm=0)

        _ = limiter1.reserve(1)
        _ = limiter2.reserve(1)

        # Both instances agree the limit (2 RPM) is exhausted
        assert limiter1.wait_time(1) > 0.0
        assert limiter2.wait_time(1) > 0.0

        with pytest.raises(RateLimited):
            await limiter1.acquire(1, max_wait=0.01)
        with pytest.raises(RateLimited):
            await limiter2.acquire(1, max_wait=0.01)
    finally:
        backend2.close()


def test_redis_circuit_breaker_transitions_across_instances(redis_backend):
    cfg = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=1.0, half_open_probes=1)
    backend2 = RedisStateBackend(redis_backend.config)
    try:
        cb1 = redis_backend.get_circuit_breaker("prov", cfg)
        cb2 = backend2.get_circuit_breaker("prov", cfg)

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

        # Instance 1 records 1 failure
        _, token1 = cb1.can_execute()
        cb1.record_failure(token1, is_qualifying=True)
        assert cb1.state == CircuitState.CLOSED
        assert cb2.consecutive_failures == 1

        # Instance 2 records 2nd failure -> Trips circuit to OPEN on both!
        _, token2 = cb2.can_execute()
        cb2.record_failure(token2, is_qualifying=True)
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        # Requests rejected immediately while OPEN
        allowed1, _ = cb1.can_execute()
        allowed2, _ = cb2.can_execute()
        assert not allowed1 and not allowed2

        # Wait for recovery timeout (1.0s)
        time.sleep(1.1)

        # Circuit transitions to HALF_OPEN and admits exactly 1 probe
        allowed_probe, probe_token = cb1.can_execute()
        assert allowed_probe and probe_token.is_probe

        # Concurrent request from instance 2 is rejected (max half_open_probes=1 reached)
        allowed_blocked, _ = cb2.can_execute()
        assert not allowed_blocked

        # Probe succeeds -> Circuit resets to CLOSED across both instances
        cb1.record_success(probe_token)
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED
        assert cb2.consecutive_failures == 0
    finally:
        backend2.close()


def test_redis_circuit_breaker_probe_neutral_cancellation(redis_backend):
    cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_s=0.5, half_open_probes=1)
    cb = redis_backend.get_circuit_breaker("cancel-test", cfg)

    # Trip breaker
    _, t = cb.can_execute()
    cb.record_failure(t, is_qualifying=True)
    assert cb.state == CircuitState.OPEN

    time.sleep(0.6)
    # Transition to HALF_OPEN and grab probe
    allowed, probe_token = cb.can_execute()
    assert allowed and probe_token.is_probe

    # No more probes allowed right now
    allowed2, _ = cb.can_execute()
    assert not allowed2

    # Client cancels before call starts -> record_neutral releases probe
    cb.record_neutral(probe_token)

    # Now another probe CAN be admitted
    allowed3, probe_token3 = cb.can_execute()
    assert allowed3 and probe_token3.is_probe


def test_redis_outage_and_degraded_reporting():
    # Configure with dead port
    bad_config = RedisConfig(url="redis://127.0.0.1:59999/0", socket_timeout_s=0.1, connect_timeout_s=0.1)
    backend = RedisStateBackend(bad_config)
    try:
        ready, msg = backend.is_ready()
        assert not ready
        assert "redis error" in msg

        status = backend.status()
        assert status["ready"] is False
        assert status["degraded"] is True

        # Operations fall back to local state rather than crashing unhandled
        limiter = backend.get_rate_limiter("dead-model", rpm=10, tpm=100)
        res = limiter.reserve(50)
        assert res.tokens == 50

        cb = backend.get_circuit_breaker("dead-cb", CircuitBreakerConfig())
        allowed, _ = cb.can_execute()
        assert allowed
    finally:
        backend.close()
