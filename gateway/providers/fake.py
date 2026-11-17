"""A fake provider for tests and for running the gateway without API keys.

``FakeProvider()`` answers deterministically and records every call.
``FakeProvider.demo()`` adds latency and occasional errors, so the dashboard
has something realistic to show when the gateway runs with
``GATEWAY_FAKE_PROVIDERS=1``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import Any

from gateway.providers.base import OVERLOADED, Completion, Provider, ProviderError, message_text

Script = Callable[[str, list[dict[str, Any]], dict[str, Any]], Completion]


class FakeProvider(Provider):
    def __init__(self, name: str = "fake", script: Script | None = None, *, latency_s: float = 0.0,
                 error_rate: float = 0.0, rng: random.Random | None = None):
        self.name = name
        self.script = script
        self.latency_s = latency_s
        self.error_rate = error_rate
        self.rng = rng or random.Random(0)
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
