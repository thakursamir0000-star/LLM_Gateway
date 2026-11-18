"""A fake provider for tests and for running the gateway without API keys.

``FakeProvider()`` answers deterministically and records every call.
``FakeProvider.demo()`` adds latency and occasional errors, so the dashboard
has something realistic to show when the gateway runs with
``GATEWAY_FAKE_PROVIDERS=1``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from typing import Any

from gateway.providers.base import (
    OVERLOADED,
    Completion,
    Provider,
    ProviderError,
    StreamChunk,
    message_text,
)

Script = Callable[[str, list[dict[str, Any]], dict[str, Any]], Completion]


class FakeProvider(Provider):
    def __init__(self, name: str = "fake", script: Script | None = None, *, latency_s: float = 0.0,
                 error_rate: float = 0.0, rng: random.Random | None = None,
                 stream_script: Any = None):
        self.name = name
        self.script = script
        self.latency_s = latency_s
        self.error_rate = error_rate
        self.rng = rng or random.Random(0)
        self.stream_script = stream_script
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def demo(cls, name: str) -> FakeProvider:
        return cls(name, latency_s=0.3, error_rate=0.03, rng=random.Random())

    async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        self.calls.append({"model": model, "messages": messages, "params": params})
        if self.latency_s:
            # Bigger models answer more slowly.
            scale = 2.5 if any(s in model for s in ("120b", "pro", "hydrogen")) else 1.0
            await asyncio.sleep(self.latency_s * scale * (0.5 + self.rng.random()))
        if self.error_rate and self.rng.random() < self.error_rate:
            raise ProviderError("fake upstream error", kind=OVERLOADED, status=503)
        if self.script is not None:
            return self.script(model, messages, params)
        prompt = " ".join(message_text(m.get("content")) for m in messages)
        last = message_text(messages[-1].get("content")) if messages else ""
        text = f"[{model}] fake answer to: {last[:80]}"
        return Completion(text, "stop", max(1, len(prompt) // 4), max(1, len(text) // 4), model)

    async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        self.calls.append({"model": model, "messages": messages, "params": params, "stream": True})
        if self.latency_s:
            scale = 2.5 if any(s in model for s in ("120b", "pro", "hydrogen")) else 1.0
            await asyncio.sleep(self.latency_s * scale * (0.5 + self.rng.random()))
        if self.error_rate and self.rng.random() < self.error_rate:
            raise ProviderError("fake upstream error", kind=OVERLOADED, status=503)
        if self.stream_script is not None:
            res = self.stream_script(model, messages, params)
            if hasattr(res, "__aiter__"):
                async for chunk in res:
                    yield chunk
                return
            if isinstance(res, Exception):
                raise res
        if self.script is not None:
            res = self.script(model, messages, params)
            if isinstance(res, Exception):
                raise res
            text = res.text
            finish_reason = res.finish_reason
            prompt_tokens = res.prompt_tokens
            completion_tokens = res.completion_tokens
        else:
            prompt = " ".join(message_text(m.get("content")) for m in messages)
            last = message_text(messages[-1].get("content")) if messages else ""
            text = f"[{model}] fake answer to: {last[:80]}"
            finish_reason = "stop"
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(text) // 4)

        parts = text.split(" ")
        for i, part in enumerate(parts):
            chunk_text = part if i == 0 else " " + part
            yield StreamChunk(text=chunk_text, model=model)
        yield StreamChunk(text="", finish_reason=finish_reason, model=model,
                          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
