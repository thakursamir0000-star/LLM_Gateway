"""A blocking state backend (Redis) never runs on the event loop: a slow Redis can't stall other requests."""

from __future__ import annotations

import asyncio
import threading
import time

from gateway.providers import ModelRegistry
from gateway.providers.fake import FakeProvider
from gateway.state.memory import InMemoryStateBackend
from tests.conftest import make_config


class SlowBlockingState(InMemoryStateBackend):
    """In-memory state that behaves like a network backend: every call blocks and is recorded with its thread."""

    blocking = True

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[int] = []

    def cooldown_left(self, alias: str) -> float:
        self.threads.append(threading.get_ident())
        time.sleep(0.2)  # a slow Redis round trip
        return super().cooldown_left(alias)


def test_blocking_state_calls_run_off_the_event_loop():
    state = SlowBlockingState()
    registry = ModelRegistry(make_config(), {"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini")},
                             state=state, pacing=False)
    alias = next(a for a, spec in registry.config.models.items() if spec.provider == "groq")

    async def scenario():
        loop_thread = threading.get_ident()
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        await asyncio.gather(*(registry.complete(alias, [{"role": "user", "content": "hi"}], {}) for _ in range(3)))
        task.cancel()
        return loop_thread, ticks

    loop_thread, ticks = asyncio.run(scenario())
    assert state.threads and loop_thread not in state.threads
    assert ticks >= 10  # the loop kept running while the three slow calls were waiting on "Redis"


def test_memory_backend_is_called_directly():
    assert getattr(InMemoryStateBackend(), "blocking", False) is False
