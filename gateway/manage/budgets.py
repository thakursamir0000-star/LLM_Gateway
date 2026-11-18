"""Per-app daily budgets (USD, UTC day).

Over ``soft_limit_fraction`` of its budget an app's routed requests go to the
cheap model; at the limit its requests are refused with HTTP 429 until the
next UTC day.

Concurrent requests: before its provider call, a request reserves its estimated
maximum cost (``reservation_estimate``), capped at what is left of the budget,
in one atomic step (``Store.reserve_budget``). The check counts the day's
logged spend plus every other request's active reservation, so two requests in
flight can't both use the same remaining budget. Logging the finished request
records its real cost and deletes its reservation in the same transaction, so a
reservation is settled exactly once; one a crashed process never settled
expires after ``reservation_ttl_s``. A request is admitted while any budget is
left, as before, so the last one admitted can still overshoot by the part of
its real cost that its capped reservation didn't cover.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from gateway.config import BudgetConfig, ModelSpec
from gateway.providers.base import message_text

# Generous prompt estimate (as router_lab.budget's worst case): ~3 characters per token plus message overhead.
PROMPT_CHARS_PER_TOKEN = 3
PROMPT_OVERHEAD_TOKENS = 64


def reservation_estimate(specs: Iterable[ModelSpec], messages: list[dict[str, Any]], params: dict[str, Any],
                         default_output_tokens: int) -> float:
    """Estimated maximum cost (USD) of one call to the most expensive of ``specs``.

    Output tokens: the request's ``max_completion_tokens`` / ``max_tokens``, else the model's configured
    one, else ``default_output_tokens``. Call it once per model a request may call in sequence (a cascade
    calls the cheap model, then maybe the strong one) and add the results.
    """
    prompt_chars = sum(len(message_text(m.get("content"))) for m in messages)
    prompt_tokens = prompt_chars // PROMPT_CHARS_PER_TOKEN + PROMPT_OVERHEAD_TOKENS
    best = 0.0
    for spec in specs:
        output = (params.get("max_completion_tokens") or params.get("max_tokens")
                  or spec.params.get("max_completion_tokens") or spec.params.get("max_tokens")
                  or default_output_tokens)
        best = max(best, spec.cost(prompt_tokens, int(output)))
    return best


@dataclass(frozen=True)
class BudgetStatus:
    app_id: str
    limit: float | None  # None = no limit
    spent: float
    soft: bool  # over the soft limit: route to the cheap model
    exceeded: bool  # at or over the limit: refuse

    @property
    def remaining(self) -> float | None:
        return None if self.limit is None else max(0.0, self.limit - self.spent)


class BudgetPolicy:
    def __init__(self, config: BudgetConfig):
        self.config = config

    def limit(self, app_id: str) -> float | None:
        if app_id in self.config.apps:
            return self.config.apps[app_id].daily_usd
        return self.config.default_daily_usd

    def status(self, app_id: str, spent: float) -> BudgetStatus:
        limit = self.limit(app_id)
        if limit is None:
            return BudgetStatus(app_id, None, spent, False, False)
        return BudgetStatus(app_id, limit, spent, spent >= limit * self.config.soft_limit_fraction, spent >= limit)


def seconds_until_utc_midnight(now: datetime) -> int:
    now = now.astimezone(UTC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))
