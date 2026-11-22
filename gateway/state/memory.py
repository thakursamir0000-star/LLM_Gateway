"""In-memory state backend for local development and testing."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from gateway.circuit import CircuitBreaker
from gateway.config import CircuitBreakerConfig
from gateway.ratelimit import RateLimiter
from gateway.state.base import CircuitBreakerProtocol, ClientAdmission, RateLimiterProtocol, StateBackend


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


class InMemoryClientLimiter:
    """Per-client requests per minute and requests in flight, for one process."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: dict[str, deque[float]] = {}
        self._leases: dict[str, dict[str, float]] = {}

    def admit(self, app_id: str, rpm: int, concurrency: int, lease_s: float) -> ClientAdmission:
        with self._lock:
            now = self._clock()
            recent = self._recent.setdefault(app_id, deque())
            while recent and now - recent[0] >= 60.0:
                recent.popleft()
            leases = self._leases.setdefault(app_id, {})
            for lease_id in [lid for lid, expiry in leases.items() if expiry <= now]:
                del leases[lease_id]
            if concurrency and len(leases) >= concurrency:
                return ClientAdmission(False, retry_after=1.0, reason="concurrency")
            if rpm and len(recent) >= rpm:
                return ClientAdmission(False, retry_after=max(0.0, recent[len(recent) - rpm] + 60.0 - now),
                                       reason="rpm")
            if rpm:
                recent.append(now)
            lease_id = uuid.uuid4().hex
            if concurrency:
                leases[lease_id] = now + lease_s
            return ClientAdmission(True, lease_id=lease_id)

    def release(self, app_id: str, lease_id: str) -> None:
        with self._lock:
            self._leases.get(app_id, {}).pop(lease_id, None)


class InMemoryStateBackend(StateBackend):
    """In-memory backend implementation: thread/asyncio safe, isolated to this process."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._cooldowns = InMemoryCooldowns(clock=clock)
        self._limiters: dict[str, RateLimiter] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self.clients = InMemoryClientLimiter(clock=clock)

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

    def admit_client(self, app_id: str, rpm: int, concurrency: int, lease_s: float) -> ClientAdmission:
        return self.clients.admit(app_id, rpm, concurrency, lease_s)

    def release_client(self, app_id: str, lease_id: str) -> None:
        self.clients.release(app_id, lease_id)

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
