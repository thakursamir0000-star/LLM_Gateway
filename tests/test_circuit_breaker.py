"""Tests for provider circuit breakers."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from gateway.circuit import CircuitBreaker, CircuitState, is_qualifying_failure
from gateway.config import GatewayConfig
from gateway.providers import ModelRegistry
from gateway.providers.base import (
    AUTH,
    BAD_REQUEST,
    OVERLOADED,
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    TIMEOUT,
    Completion,
    Provider,
    ProviderError,
)
from gateway.providers.fake import FakeProvider


class MockClock:
    def __init__(self, start: float = 1000.0):
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def test_failure_classification():
    assert is_qualifying_failure(ProviderError("overloaded", kind=OVERLOADED, status=503)) is True
    assert is_qualifying_failure(ProviderError("timeout", kind=TIMEOUT, status=504)) is True
    assert is_qualifying_failure(ProviderError("server error", kind="other", status=500)) is True
    assert is_qualifying_failure(ConnectionError("network down")) is True
    assert is_qualifying_failure(TimeoutError("timed out")) is True

    # Non-qualifying
    assert is_qualifying_failure(ProviderError("bad request", kind=BAD_REQUEST, status=400)) is False
    assert is_qualifying_failure(ProviderError("auth failed", kind=AUTH, status=401)) is False
    assert is_qualifying_failure(ProviderError("rate limit", kind=RATE_LIMIT, status=429)) is False
    assert is_qualifying_failure(ProviderError("quota", kind=QUOTA_EXHAUSTED, status=429)) is False


def test_httpx_transport_failures_are_qualifying():
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    assert is_qualifying_failure(httpx.ReadError("read failed", request=request)) is True
    assert is_qualifying_failure(httpx.WriteError("write failed", request=request)) is True
    assert is_qualifying_failure(httpx.ConnectError("connect failed", request=request)) is True
    assert is_qualifying_failure(httpx.ReadTimeout("timed out", request=request)) is True

    # Ordinary HTTP client errors stay excluded.
    for status in (400, 401, 404, 429):
        response = httpx.Response(status, request=request)
        error = httpx.HTTPStatusError("request failed", request=request, response=response)
        assert is_qualifying_failure(error) is False


def test_closed_allows_requests():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=5, recovery_timeout_s=30.0, half_open_probes=1, clock=clock)
    assert breaker.state == CircuitState.CLOSED

    allowed, token = breaker.can_execute()
    assert allowed is True
    assert token.is_probe is False
    breaker.record_success(token)
    assert breaker.consecutive_failures == 0


def test_threshold_opens_circuit():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=3, recovery_timeout_s=30.0, half_open_probes=1, clock=clock)

    for i in range(2):
        allowed, token = breaker.can_execute()
        assert allowed is True
        breaker.record_failure(token, is_qualifying=True)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == i + 1

    # 3rd failure trips the breaker to OPEN
    allowed, token = breaker.can_execute()
    assert allowed is True
    breaker.record_failure(token, is_qualifying=True)
    assert breaker.state == CircuitState.OPEN


def test_open_blocks_provider_calls():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=2, recovery_timeout_s=30.0, half_open_probes=1, clock=clock)

    # Trip the breaker
    for _ in range(2):
        _, token = breaker.can_execute()
        breaker.record_failure(token, is_qualifying=True)
    assert breaker.state == CircuitState.OPEN

    # Requests are rejected immediately
    allowed, token = breaker.can_execute()
    assert allowed is False
    assert token.allowed is False


def test_cooldown_permits_half_open_probe():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=2, recovery_timeout_s=30.0, half_open_probes=1, clock=clock)

    # Trip breaker
    for _ in range(2):
        _, token = breaker.can_execute()
        breaker.record_failure(token, is_qualifying=True)
    assert breaker.state == CircuitState.OPEN

    # 10s later: still open
    clock.advance(10.0)
    allowed, _ = breaker.can_execute()
    assert allowed is False
    assert breaker.state == CircuitState.OPEN

    # 30s elapsed total: transitions to HALF_OPEN
    clock.advance(20.0)
    allowed, token = breaker.can_execute()
    assert allowed is True
    assert token.is_probe is True
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.active_probes == 1


def test_concurrent_requests_cannot_exceed_half_open_probe_limit():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=2, recovery_timeout_s=30.0, half_open_probes=1, clock=clock)

    # Trip breaker
    for _ in range(2):
        _, token = breaker.can_execute()
        breaker.record_failure(token, is_qualifying=True)

    clock.advance(35.0)
    allowed1, token1 = breaker.can_execute()
    assert allowed1 is True
    assert token1.is_probe is True

    # Second concurrent probe must be rejected while first is in-flight
    allowed2, token2 = breaker.can_execute()
    assert allowed2 is False
    assert token2.allowed is False


def test_successful_probe_closes_circuit():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=2, recovery_timeout_s=30.0, half_open_probes=1, clock=clock)

    # Trip breaker
    for _ in range(2):
        _, token = breaker.can_execute()
        breaker.record_failure(token, is_qualifying=True)

    clock.advance(35.0)
    allowed, token = breaker.can_execute()
    assert allowed is True
    assert token.is_probe is True

    breaker.record_success(token)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0
    assert breaker.active_probes == 0

    # Normal requests can execute again
    allowed_after, token_after = breaker.can_execute()
    assert allowed_after is True
    assert token_after.is_probe is False


def test_failed_probe_reopens_circuit():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=2, recovery_timeout_s=30.0, half_open_probes=1, clock=clock)

    # Trip breaker
    for _ in range(2):
        _, token = breaker.can_execute()
        breaker.record_failure(token, is_qualifying=True)

    clock.advance(35.0)
    allowed, token = breaker.can_execute()
    assert allowed is True
    assert token.is_probe is True

    # Failed probe immediately reopens the circuit
    breaker.record_failure(token, is_qualifying=True)
    assert breaker.state == CircuitState.OPEN
    assert breaker.active_probes == 0

    # Next call blocked
    allowed_next, _ = breaker.can_execute()
    assert allowed_next is False


def test_client_4xx_and_rate_limits_do_not_trip_circuit():
    clock = MockClock()
    breaker = CircuitBreaker("test-prov", failure_threshold=2, recovery_timeout_s=30.0, clock=clock)

    # 400 Bad Request
    _, token = breaker.can_execute()
    breaker.record_failure(token, is_qualifying=False)
    assert breaker.consecutive_failures == 0
    assert breaker.state == CircuitState.CLOSED

    # 401 Auth
    _, token = breaker.can_execute()
    breaker.record_failure(token, is_qualifying=False)
    assert breaker.consecutive_failures == 0
    assert breaker.state == CircuitState.CLOSED

    # 429 Rate Limit
    _, token = breaker.can_execute()
    breaker.record_failure(token, is_qualifying=False)
    assert breaker.consecutive_failures == 0
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_single_logical_request_counts_retries_as_single_failure():
    """When a model call retries internally 3 times and fails, only 1 failure is recorded on the circuit."""
    class FailingProvider(Provider):
        name = "failing"
        def __init__(self):
            self.calls = 0
        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            self.calls += 1
            raise ProviderError("503 server error", kind=OVERLOADED, status=503)

    prov = FailingProvider()
    cfg = GatewayConfig.model_validate({
        "models": {
            "m1": {"provider": "failing", "model": "test-model"}
        },
        "providers": {
            "failing": {"base_url": "http://mock", "api_key_env": "MOCK_KEY"}
        },
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m1", "strong": "m1", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "retries": {"attempts": 3, "max_wait_s": 10.0},
        "circuit_breaker": {"failure_threshold": 2, "recovery_timeout_s": 30.0},
    })
    registry = ModelRegistry(cfg, {"failing": prov}, max_wait=10.0, sleep=lambda s: asyncio.sleep(0.0001))
    breaker = registry.circuit_breakers["failing"]

    # 1 logical request -> 3 attempts internally
    with pytest.raises(ProviderError):
        await registry.complete("m1", [{"role": "user", "content": "hi"}], {})

    assert prov.calls == 3
    # Exactly 1 failure on the breaker, NOT 3
    assert breaker.consecutive_failures == 1
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breakers_are_isolated_per_provider():
    """Tripping provider A does not affect provider B."""
    clock = MockClock()
    p_bad = FakeProvider("bad")
    p_good = FakeProvider("good")

    cfg = GatewayConfig.model_validate({
        "models": {
            "m_bad": {"provider": "bad", "model": "bad-model"},
            "m_good": {"provider": "good", "model": "good-model"},
        },
        "providers": {
            "bad": {"base_url": "http://bad", "api_key_env": "KEY"},
            "good": {"base_url": "http://good", "api_key_env": "KEY"},
        },
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m_bad", "strong": "m_good", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "circuit_breaker": {"failure_threshold": 1, "recovery_timeout_s": 30.0},
    })
    registry = ModelRegistry(cfg, {"bad": p_bad, "good": p_good}, clock=clock)

    # Trip breaker for bad
    token = registry.circuit_breakers["bad"].can_execute()[1]
    registry.circuit_breakers["bad"].record_failure(token, is_qualifying=True)

    assert registry.circuit_breakers["bad"].state == CircuitState.OPEN
    assert registry.circuit_breakers["good"].state == CircuitState.CLOSED

    # Calling m_bad fails with 503 circuit breaker is OPEN
    with pytest.raises(ProviderError) as exc_info:
        await registry.complete("m_bad", [{"role": "user", "content": "hi"}], {})
    assert exc_info.value.status == 503
    assert "circuit breaker is OPEN" in str(exc_info.value)

    # Calling m_good succeeds normally
    res = await registry.complete("m_good", [{"role": "user", "content": "hi"}], {})
    assert res.text.startswith("[good-model]")


@pytest.mark.asyncio
async def test_cancellation_during_rate_limiter_does_not_occupy_half_open_probe_slot():
    """Cancellation while waiting for rate limiter in complete() or stream() must release probe slot."""
    clock = MockClock()
    p = FakeProvider("p")

    cfg = GatewayConfig.model_validate({
        "models": {
            "m": {
                "provider": "p",
                "model": "model-1",
                "limits": {"rpm": 1},
            }
        },
        "providers": {"p": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "circuit_breaker": {"failure_threshold": 1, "recovery_timeout_s": 30.0, "half_open_probes": 1},
    })

    sleep_started = asyncio.Event()

    async def blocking_sleep(_: float):
        sleep_started.set()
        await asyncio.sleep(100.0)

    registry = ModelRegistry(cfg, {"p": p}, clock=clock, max_wait=100.0, sleep=blocking_sleep)
    breaker = registry.circuit_breakers["p"]

    # 1. Trip the circuit to OPEN
    _, token = breaker.can_execute()
    breaker.record_failure(token, is_qualifying=True)
    assert breaker.state == CircuitState.OPEN

    # 2. Advance clock past recovery timeout so it can transition to HALF_OPEN
    clock.advance(35.0)

    # 3. Exhaust the rate limiter tokens first so limiter.acquire will sleep
    limiter = registry.limiters["m"]
    limiter.reserve(10)

    # 4. Launch complete() in background task; it will acquire probe token and wait in blocking_sleep
    task = asyncio.create_task(
        registry.complete("m", [{"role": "user", "content": "hi"}], {})
    )
    await sleep_started.wait()

    # The probe is currently active
    assert breaker.active_probes == 1

    # 5. Cancel the task while it's waiting for the rate limiter
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 6. Verify probe token was released and slot is NOT permanently occupied!
    assert breaker.active_probes == 0
    allowed, token2 = breaker.can_execute()
    assert allowed is True
    assert token2.is_probe is True
    breaker.record_neutral(token2)
    assert breaker.active_probes == 0

    # Same verification for stream()
    sleep_started.clear()
    stream_iter = registry.stream("m", [{"role": "user", "content": "hi"}], {})
    task_stream = asyncio.create_task(anext(stream_iter))
    await sleep_started.wait()
    assert breaker.active_probes == 1

    task_stream.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_stream

    assert breaker.active_probes == 0
    allowed, token3 = breaker.can_execute()
    assert allowed is True
    assert token3.is_probe is True
    breaker.record_neutral(token3)
