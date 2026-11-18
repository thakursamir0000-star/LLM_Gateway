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
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from gateway.circuit import CircuitBreaker, is_qualifying_failure
from gateway.config import GatewayConfig, ModelSpec
from gateway.providers.base import (
    OVERLOADED,
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    RETRYABLE,
    Completion,
    Provider,
    ProviderError,
    StreamChunk,
    message_text,
)
from gateway.ratelimit import RateLimited, RateLimiter, estimate_tokens

__all__ = ["Completion", "ModelRegistry", "Provider", "ProviderError", "StreamChunk", "build_providers",
           "stream_tokens"]


def stream_tokens(prompt_chars: int, streamed_chars: int, usage_prompt: int | None, usage_completion: int | None,
                  produced: bool) -> int:
    """Tokens a stream used, for the per-minute limit.

    The provider's usage when it sent it (never added to an estimate), otherwise ~4 characters per token as in
    ``estimate_tokens``: the prompt, plus at least one token for any streamed text. A stream that produced
    nothing and reported no usage (it failed before its first chunk) counts 0, like a failed ``complete``.
    """
    if not produced and usage_prompt is None and usage_completion is None:
        return 0
    prompt = usage_prompt if usage_prompt is not None else prompt_chars // 4
    if usage_completion is not None:
        completion = usage_completion
    else:
        completion = max(1, streamed_chars // 4) if streamed_chars else 0
    return max(0, prompt) + max(0, completion)


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
        self.circuit_breakers = {
            name: CircuitBreaker(name, config.circuit_breaker, clock=clock)
            for name in providers
        }
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
        breaker = self.circuit_breakers[spec.provider]
        allowed, token = breaker.can_execute()
        if not allowed:
            raise ProviderError(f"provider {spec.provider!r} circuit breaker is OPEN",
                                kind=OVERLOADED, status=503)
        provider = self.providers[spec.provider]
        merged = {**spec.params, **{k: v for k, v in params.items() if v is not None}}
        chars = sum(len(message_text(m.get("content"))) for m in messages)
        estimate = estimate_tokens(chars, merged.get("max_completion_tokens") or merged.get("max_tokens"))
        limiter = self.limiters[alias]
        settled = False
        try:
            for attempt in range(1, self.attempts + 1):
                try:
                    reservation = await limiter.acquire(estimate, max_wait=self.max_wait, sleep=self._sleep)
                except RateLimited as e:
                    settled = True
                    breaker.record_neutral(token)
                    raise ProviderError(str(e), kind=RATE_LIMIT, status=429, retry_after=e.wait_s) from e
                except BaseException:
                    settled = True
                    breaker.record_neutral(token)
                    raise
                started = time.perf_counter()
                try:
                    completion = await provider.complete(spec.model, messages, merged)
                    completion.latency_ms = round((time.perf_counter() - started) * 1000, 1)
                    settled = True
                    breaker.record_success(token)
                    limiter.settle(reservation, completion.prompt_tokens + completion.completion_tokens)
                    return completion
                except BaseException as e:
                    limiter.settle(reservation, 0)
                    if isinstance(e, ProviderError) and e.kind == QUOTA_EXHAUSTED:
                        cooldown = max(self.config.retries.quota_cooldown_s, e.retry_after or 0.0)
                        self._cooldown_until[alias] = self._clock() + cooldown
                    if not isinstance(e, ProviderError) or e.kind not in RETRYABLE or attempt == self.attempts:
                        settled = True
                        breaker.record_failure(token, is_qualifying=is_qualifying_failure(e))
                        raise
                    wait = max(e.retry_after or 0.0, float(2 ** (attempt - 1)))
                    if wait > self.max_wait:
                        settled = True
                        breaker.record_failure(token, is_qualifying=is_qualifying_failure(e))
                        raise
                    await self._sleep(wait)
                    continue
            raise AssertionError("unreachable")  # pragma: no cover
        finally:
            if not settled:
                breaker.record_neutral(token)

    async def stream(self, alias: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """Stream model ``alias``; raises ProviderError on failure.

        Retryable failures before the first chunk use the configured attempts,
        backoff and max wait. Once a chunk has been yielded the stream is never
        retried. All retries of one call count as at most one breaker failure.
        """
        spec = self.spec(alias)
        left = self.cooldown_left(alias)
        if left > 0:
            raise ProviderError(f"{alias} skipped: quota exhausted, cooling down for {left:.0f}s",
                                kind=QUOTA_EXHAUSTED, status=429, retry_after=left)
        breaker = self.circuit_breakers[spec.provider]
        allowed, token = breaker.can_execute()
        if not allowed:
            raise ProviderError(f"provider {spec.provider!r} circuit breaker is OPEN",
                                kind=OVERLOADED, status=503)
        provider = self.providers[spec.provider]
        merged = {**spec.params, **{k: v for k, v in params.items() if v is not None}}
        chars = sum(len(message_text(m.get("content"))) for m in messages)
        estimate = estimate_tokens(chars, merged.get("max_completion_tokens") or merged.get("max_tokens"))
        limiter = self.limiters[alias]
        settled = False
        try:
            for attempt in range(1, self.attempts + 1):
                try:
                    reservation = await limiter.acquire(estimate, max_wait=self.max_wait, sleep=self._sleep)
                except RateLimited as e:
                    settled = True
                    breaker.record_neutral(token)
                    raise ProviderError(str(e), kind=RATE_LIMIT, status=429, retry_after=e.wait_s) from e
                except BaseException:
                    settled = True
                    breaker.record_neutral(token)
                    raise

                upstream_gen = provider.stream(spec.model, messages, merged)
                usage_prompt: int | None = None
                usage_completion: int | None = None
                streamed_chars = 0
                first_chunk_yielded = False
                try:
                    try:
                        async for chunk in upstream_gen:
                            if chunk.prompt_tokens is not None:
                                usage_prompt = chunk.prompt_tokens
                            if chunk.completion_tokens is not None:
                                usage_completion = chunk.completion_tokens
                            streamed_chars += len(chunk.text or "")
                            first_chunk_yielded = True
                            yield chunk
                    except BaseException as e:
                        if isinstance(e, ProviderError) and e.kind == QUOTA_EXHAUSTED:
                            cooldown = max(self.config.retries.quota_cooldown_s, e.retry_after or 0.0)
                            self._cooldown_until[alias] = self._clock() + cooldown
                        retry_after = e.retry_after if isinstance(e, ProviderError) else None
                        retryable = ((isinstance(e, ProviderError) and e.kind in RETRYABLE)
                                     or isinstance(e, httpx.TransportError))
                        if first_chunk_yielded or not retryable or attempt == self.attempts:
                            settled = True
                            breaker.record_failure(token, is_qualifying=is_qualifying_failure(e))
                            raise
                        wait = max(retry_after or 0.0, float(2 ** (attempt - 1)))
                        if wait > self.max_wait:
                            settled = True
                            breaker.record_failure(token, is_qualifying=is_qualifying_failure(e))
                            raise
                        await self._sleep(wait)
                        continue
                    else:
                        settled = True
                        breaker.record_success(token)
                        return
                finally:
                    limiter.settle(reservation, stream_tokens(chars, streamed_chars, usage_prompt, usage_completion,
                                                              first_chunk_yielded))
                    await upstream_gen.aclose()
        finally:
            if not settled:
                breaker.record_neutral(token)

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()
