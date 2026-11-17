"""Client-side pacing under a provider's per-minute limits (requests and tokens).

Free tiers limit each model per minute (Groq ``openai/gpt-oss-120b``: 8,000
tokens per minute). Pacing below the limit avoids 429s instead of recovering
from them. The limiter keeps a one-minute window of reservations; a call
reserves its estimated tokens up front and settles the real count afterwards.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

WINDOW_S = 60.0


class RateLimited(Exception):
    """The next slot is further away than the caller is willing to wait."""

    def __init__(self, wait_s: float):
        super().__init__(f"local rate limit: next slot in {wait_s:.1f}s")
        self.wait_s = wait_s


@dataclass
class Reservation:
    t: float
    tokens: int


class RateLimiter:
    """Requests-per-minute and tokens-per-minute pacing (0 = no limit)."""

    def __init__(self, rpm: int = 0, tpm: int = 0, clock: Callable[[], float] = time.monotonic):
        self.rpm = rpm
        self.tpm = tpm
        self._clock = clock
        self._window: deque[Reservation] = deque()
        self._lock = asyncio.Lock()

    def _expire(self, now: float) -> None:
        while self._window and now - self._window[0].t >= WINDOW_S:
            self._window.popleft()

    def wait_time(self, tokens: int) -> float:
        """Seconds until a request of ``tokens`` fits in the window (0 = now)."""
        now = self._clock()
        self._expire(now)
        wait = 0.0
        if self.rpm and len(self._window) >= self.rpm:
            oldest_needed = self._window[len(self._window) - self.rpm]
            wait = max(wait, oldest_needed.t + WINDOW_S - now)
        if self.tpm and self._window:
            used = sum(r.tokens for r in self._window)
            # A request larger than the whole budget goes alone in an empty window.
            excess = used + min(tokens, self.tpm) - self.tpm
            if excess > 0:
                freed = 0
                for r in self._window:
                    freed += r.tokens
                    if freed >= excess:
                        wait = max(wait, r.t + WINDOW_S - now)
                        break
        return max(0.0, wait)

    def reserve(self, tokens: int) -> Reservation:
        """Record a request now (call only when ``wait_time`` is 0)."""
        reservation = Reservation(self._clock(), max(0, tokens))
        self._window.append(reservation)
        return reservation

    @staticmethod
    def settle(reservation: Reservation, tokens: int) -> None:
        """Replace the estimate with the tokens the call really used."""
        reservation.tokens = max(0, tokens)

    async def acquire(self, tokens: int, max_wait: float = float("inf"),
                      sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> Reservation:
        """Wait for a slot and reserve it; raise RateLimited if the wait exceeds ``max_wait``."""
        while True:
            async with self._lock:
                wait = self.wait_time(tokens)
                if wait <= 0:
                    return self.reserve(tokens)
                if wait > max_wait:
                    raise RateLimited(wait)
            await sleep(wait + 0.05)


def estimate_tokens(text_chars: int, max_output: int | None, default_output: int = 512) -> int:
    """Rough token estimate for pacing: ~4 characters per token plus the expected output."""
    return text_chars // 4 + (max_output if max_output else default_output)
