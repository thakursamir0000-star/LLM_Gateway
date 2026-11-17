"""The routing interface.

Every router returns a score: the estimated probability that the cheap model
is good enough for the request. One threshold turns any score into a tier:
cheap when ``score >= threshold``, strong otherwise. Raising the threshold
sends more traffic to the strong model (more cost, more quality).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gateway.providers.base import ProviderError, message_text


@dataclass(frozen=True)
class RouteScore:
    score: float  # P(cheap model is good enough), in [0, 1]
    router: str  # which router produced it
    reason: str  # short human-readable explanation, logged with the request


class Router(Protocol):
    name: str
    description: str

    def score(self, messages: list[dict[str, Any]]) -> RouteScore:
        """Score one request (OpenAI-style messages)."""
        ...


def choose_tier(score: float, threshold: float) -> str:
    return "cheap" if score >= threshold else "strong"


def safe_text(content: Any) -> str:
    try:
        return message_text(content)
    except ProviderError:
        return ""


def routing_text(messages: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """The text a router reads: the last user message (all text if there is none)."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return safe_text(message.get("content"))[:max_chars]
    return "\n".join(safe_text(m.get("content")) for m in messages)[:max_chars]
