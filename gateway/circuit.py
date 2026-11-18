"""Provider circuit breakers: in-process, thread-safe / asyncio-safe state machine.

States:
- CLOSED: normal traffic passes through.
- OPEN: provider calls are skipped/rejected immediately with 503.
- HALF_OPEN: after recovery cooldown, allows limited probe requests to test provider health.

Failure classification:
- Qualifying failures: network/connection errors, timeouts, HTTP 5xx responses.
- Non-qualifying: ordinary 4xx client errors (bad requests, auth failure), 429 rate limits (handled by cooldown).
- Single logical request with retries counts as at most one failure.
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from gateway.config import CircuitBreakerConfig


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class ProbeToken:
    allowed: bool
    is_probe: bool = False
    provider: str = ""


def is_qualifying_failure(exc: BaseException) -> bool:
    """True if exc represents an upstream infrastructure or server failure."""
    kind = getattr(exc, "kind", None)
    if kind is not None:
        if kind in ("overloaded", "timeout"):
            return True
        status = getattr(exc, "status", None)
        if status is not None and isinstance(status, int) and status >= 500:
            return True
        # Explicitly exclude client 4xx, auth, and rate limits
        if kind in ("bad_request", "auth", "rate_limit", "quota_exhausted"):
            return False
        return False
    # HTTPX transport failures (ConnectError, ReadError, WriteError, timeouts, ...).
    if isinstance(exc, httpx.TransportError):
        return True
    # Check for network/connection/timeout exceptions (httpx, OSError, etc.)
    type_name = type(exc).__name__.lower()
    if any(keyword in type_name for keyword in ("timeout", "connect", "network", "socket", "protocol")):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        *,
        failure_threshold: int | None = None,
        recovery_timeout_s: float | None = None,
        half_open_probes: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        cfg = config or CircuitBreakerConfig()
        self.failure_threshold = failure_threshold if failure_threshold is not None else cfg.failure_threshold
        self.recovery_timeout_s = recovery_timeout_s if recovery_timeout_s is not None else cfg.recovery_timeout_s
        self.half_open_probes = half_open_probes if half_open_probes is not None else cfg.half_open_probes
        self._clock = clock

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at = 0.0
        self._active_probes = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_half_open(self._clock())
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    @property
    def active_probes(self) -> int:
        with self._lock:
            return self._active_probes

    def _maybe_transition_half_open(self, now: float) -> None:
        """Must be called while holding _lock."""
        if self._state == CircuitState.OPEN:
            if now - self._opened_at >= self.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0
                self._active_probes = 0

    def can_execute(self) -> tuple[bool, ProbeToken]:
        """Check whether a request is allowed through. Thread/asyncio safe."""
        with self._lock:
            now = self._clock()
            self._maybe_transition_half_open(now)

            if self._state == CircuitState.OPEN:
                return False, ProbeToken(allowed=False, is_probe=False, provider=self.name)

            if self._state == CircuitState.HALF_OPEN:
                if self._active_probes < self.half_open_probes:
                    self._active_probes += 1
                    return True, ProbeToken(allowed=True, is_probe=True, provider=self.name)
                # Max concurrent half-open probes reached
                return False, ProbeToken(allowed=False, is_probe=False, provider=self.name)

            # CLOSED
            return True, ProbeToken(allowed=True, is_probe=False, provider=self.name)

    def record_success(self, token: ProbeToken) -> None:
        """Record that a provider call completed successfully."""
        with self._lock:
            if token.is_probe:
                self._active_probes = max(0, self._active_probes - 1)
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.half_open_probes:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
            else:
                self._consecutive_failures = 0

    def record_failure(self, token: ProbeToken, is_qualifying: bool = True) -> None:
        """Record that a provider call failed."""
        with self._lock:
            if not is_qualifying:
                # Non-qualifying errors (4xx client mistakes, auth, rate limit) release active probe without penalty
                if token.is_probe:
                    self._active_probes = max(0, self._active_probes - 1)
                return

            now = self._clock()
            if token.is_probe:
                self._active_probes = max(0, self._active_probes - 1)
                self._state = CircuitState.OPEN
                self._opened_at = now
                self._consecutive_successes = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = now
                    self._consecutive_successes = 0

    def record_neutral(self, token: ProbeToken) -> None:
        """Release active probe tracking for non-call events (e.g. rate limit wait abort)."""
        with self._lock:
            if token.is_probe:
                self._active_probes = max(0, self._active_probes - 1)

    def reset(self) -> None:
        """Reset to initial CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._opened_at = 0.0
            self._active_probes = 0
