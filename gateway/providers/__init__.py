"""Provider adapters and the model registry that calls a configured model alias.

The registry merges the model's default parameters, paces calls under the
model's per-minute limits and retries within bounds:
- a per-minute rate limit waits the provider's suggested delay (at least an
  exponential backoff), unless that is longer than ``max_wait``;
- an overloaded/unavailable server or a timeout backs off exponentially;
- a daily quota, an auth error or a bad request fails at once.
After a quota error (daily limit, no credits) the model is skipped for a
cooldown (``retries.quota_cooldown_s``, or longer if the provider says so), so
the gateway falls back at once instead of calling a model that must fail.
The gateway uses a short ``max_wait`` (it holds a client's request); the
benchmark scripts use a long one.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from gateway.config import GatewayConfig, ModelSpec
from gateway.providers.base import (
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    RETRYABLE,
    Completion,
    Provider,
    ProviderError,
    message_text,
)
from gateway.ratelimit import RateLimited, RateLimiter, estimate_tokens

__all__ = ["Completion", "ModelRegistry", "Provider", "ProviderError", "build_providers"]


def build_providers(config: GatewayConfig, *, fake: bool = False) -> dict[str, Provider]:
    """One adapter per configured provider (fake ones when ``fake`` is true)."""
    from gateway.providers.fake import FakeProvider
    from gateway.providers.gemini import GeminiProvider
    from gateway.providers.groq import GroqProvider
    from gateway.providers.openrouter import OpenRouterProvider

    adapters: dict[str, type] = {"groq": GroqProvider, "gemini": GeminiProvider, "openrouter": OpenRouterProvider}
    providers: dict[str, Provider] = {}
    for name, spec in config.providers.items():
        if fake:
            providers[name] = FakeProvider.demo(name)
            continue
        if name not in adapters:
            raise ValueError(f"no adapter for provider {name!r} (known: {', '.join(adapters)})")
        providers[name] = adapters[name](os.environ.get(spec.api_key_env), spec.base_url, spec.timeout_s)
    return providers


class ModelRegistry:
    def __init__(self, config: GatewayConfig, providers: dict[str, Provider], *, attempts: int | None = None,
                 max_wait: float | None = None, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic, pacing: bool = True):
        self.config = config
        self.providers = providers
        self.attempts = attempts if attempts is not None else config.retries.attempts
        self.max_wait = max_wait if max_wait is not None else config.retries.max_wait_s
        self._sleep = sleep
        self._clock = clock
        self._cooldown_until: dict[str, float] = {}  # alias -> clock time when it may be called again
        # pacing=False (fake providers) ignores the per-minute limits, which only matter for real APIs.
        self.limiters = {alias: RateLimiter(spec.limits.rpm if pacing else 0, spec.limits.tpm if pacing else 0, clock)
                         for alias, spec in config.models.items()}

    def spec(self, alias: str) -> ModelSpec:
        return self.config.models[alias]

    def cooldown_left(self, alias: str) -> float:
        """Seconds until ``alias`` may be called again after a quota error (0 = now)."""
        return max(0.0, self._cooldown_until.get(alias, 0.0) - self._clock())

    async def complete(self, alias: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        """Call model ``alias``; raises ProviderError after the retries run out."""
        spec = self.spec(alias)
        left = self.cooldown_left(alias)
        if left > 0:
            raise ProviderError(f"{alias} skipped: quota exhausted, cooling down for {left:.0f}s",
                                kind=QUOTA_EXHAUSTED, status=429, retry_after=left)
        provider = self.providers[spec.provider]
        merged = {**spec.params, **{k: v for k, v in params.items() if v is not None}}
        chars = sum(len(message_text(m.get("content"))) for m in messages)
        estimate = estimate_tokens(chars, merged.get("max_completion_tokens") or merged.get("max_tokens"))
        limiter = self.limiters[alias]
        for attempt in range(1, self.attempts + 1):
            try:
                reservation = await limiter.acquire(estimate, max_wait=self.max_wait, sleep=self._sleep)
            except RateLimited as e:
                raise ProviderError(str(e), kind=RATE_LIMIT, status=429, retry_after=e.wait_s) from e
            started = time.perf_counter()
            try:
                completion = await provider.complete(spec.model, messages, merged)
                completion.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            except ProviderError as e:
                limiter.settle(reservation, 0)
                if e.kind == QUOTA_EXHAUSTED:
                    cooldown = max(self.config.retries.quota_cooldown_s, e.retry_after or 0.0)
                    self._cooldown_until[alias] = self._clock() + cooldown
                if e.kind not in RETRYABLE or attempt == self.attempts:
                    raise
                wait = max(e.retry_after or 0.0, float(2 ** (attempt - 1)))
                if wait > self.max_wait:
                    raise
                await self._sleep(wait)
                continue
            limiter.settle(reservation, completion.prompt_tokens + completion.completion_tokens)
            return completion
        raise AssertionError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()
