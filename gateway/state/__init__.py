"""Gateway state backend factory."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from gateway.config import GatewayConfig
from gateway.state.base import CircuitBreakerProtocol, RateLimiterProtocol, StateBackend
from gateway.state.memory import InMemoryStateBackend


def get_state_backend(
    config: GatewayConfig | None = None,
    *,
    backend: str | None = None,
    clock: Callable[[], float] | None = None,
) -> StateBackend:
    """Factory to instantiate the configured state backend (in-memory or Redis)."""
    selected = (
        backend
        or os.environ.get("GATEWAY_STATE_BACKEND")
        or (config.state.backend if config and hasattr(config, "state") else "memory")
    )
    if selected.strip().lower() == "redis":
        from gateway.state.redis_backend import RedisStateBackend

        redis_cfg = config.state.redis if config and hasattr(config, "state") else None
        if redis_cfg is None:
            from gateway.config import RedisConfig

            url = os.environ.get("GATEWAY_REDIS_URL") or os.environ.get("REDIS_URL") or "redis://localhost:6379/0"
            redis_cfg = RedisConfig(url=url)
        return RedisStateBackend(redis_cfg)

    return InMemoryStateBackend(clock=clock or time.monotonic)


__all__ = [
    "CircuitBreakerProtocol",
    "InMemoryStateBackend",
    "RateLimiterProtocol",
    "StateBackend",
    "get_state_backend",
]
