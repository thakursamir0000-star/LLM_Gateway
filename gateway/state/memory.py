"""In-memory state backend for local development and testing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from gateway.circuit import CircuitBreaker
from gateway.config import CircuitBreakerConfig
from gateway.ratelimit import RateLimiter
from gateway.state.base import CircuitBreakerProtocol, RateLimiterProtocol, StateBackend


class InMemoryCooldowns:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._cooldown_until: dict[str, float] = {}

    def cooldown_left(self, alias: str) -> float:
        return max(0.0, self._cooldown_until.get(alias, 0.0) - self._clock())

    def set_cooldown(self, alias: str, duration_s: float) -> None:
        expiry = self._clock() + duration_s
        if expiry > self._cooldown_until.get(alias, 0.0):
            self._cooldown_until[alias] = expiry

    def reset(self) -> None:
        self._cooldown_until.clear()


class InMemoryStateBackend(StateBackend):
    """In-memory backend implementation: thread/asyncio safe, isolated to this process."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._cooldowns = InMemoryCooldowns(clock=clock)
        self._limiters: dict[str, RateLimiter] = {}
        self._breakers: dict[str, CircuitBreaker] = {}

    def cooldown_left(self, alias: str) -> float:
        return self._cooldowns.cooldown_left(alias)

    def set_cooldown(self, alias: str, duration_s: float) -> None:
        self._cooldowns.set_cooldown(alias, duration_s)

    def get_rate_limiter(self, alias: str, rpm: int, tpm: int) -> RateLimiterProtocol:
        key = f"{alias}:{rpm}:{tpm}"
        if key not in self._limiters:
            self._limiters[key] = RateLimiter(rpm=rpm, tpm=tpm, clock=self._clock)
        return self._limiters[key]

    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreakerProtocol:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config=config, clock=self._clock)
        return self._breakers[name]

    def is_ready(self) -> tuple[bool, str]:
        return True, "in-memory state backend ready"

    def status(self) -> dict[str, Any]:
        return {
            "backend": "memory",
            "ready": True,
            "distributed": False,
            "details": "in-memory (single-process)",
        }

    def close(self) -> None:
        pass
