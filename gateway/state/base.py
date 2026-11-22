"""Base interfaces and protocols for gateway state backends."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from gateway.circuit import CircuitState, ProbeToken
from gateway.config import CircuitBreakerConfig
from gateway.ratelimit import Reservation


@runtime_checkable
class RateLimiterProtocol(Protocol):
    rpm: int
    tpm: int

    def wait_time(self, tokens: int) -> float: ...

    def reserve(self, tokens: int) -> Reservation: ...

    def settle(self, reservation: Reservation, tokens: int) -> None: ...

    async def acquire(
        self,
        tokens: int,
        max_wait: float = float("inf"),
        sleep: Callable[[float], Awaitable[None]] = ...,
    ) -> Reservation: ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    name: str

    @property
    def state(self) -> CircuitState: ...

    @property
    def consecutive_failures(self) -> int: ...

    @property
    def active_probes(self) -> int: ...

    def can_execute(self) -> tuple[bool, ProbeToken]: ...

    def record_success(self, token: ProbeToken) -> None: ...

    def record_failure(self, token: ProbeToken, is_qualifying: bool = True) -> None: ...

    def record_neutral(self, token: ProbeToken) -> None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class ClientAdmission:
    """The outcome of ``StateBackend.admit_client``."""

    admitted: bool
    lease_id: str | None = None  # release it with release_client when the request ends
    retry_after: float = 0.0
    reason: str = ""  # "rpm", "concurrency" or "state_unavailable" when refused


class StateBackend(Protocol):
    """Abstract shared state backend for rate limits, cooldowns, and circuit breakers."""

    def cooldown_left(self, alias: str) -> float: ...

    def set_cooldown(self, alias: str, duration_s: float) -> None: ...

    def get_rate_limiter(self, alias: str, rpm: int, tpm: int) -> RateLimiterProtocol: ...

    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreakerProtocol: ...

    def admit_client(self, app_id: str, rpm: int, concurrency: int, lease_s: float) -> ClientAdmission: ...

    def release_client(self, app_id: str, lease_id: str) -> None: ...

    def is_ready(self) -> tuple[bool, str]: ...

    def status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
