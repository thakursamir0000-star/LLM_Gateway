"""Per-app daily budgets (USD, UTC day).

Over ``soft_limit_fraction`` of its budget an app's routed requests go to the
cheap model; at the limit its requests are refused with HTTP 429 until the
next UTC day. A request's cost is known only after it runs, so an app can
overshoot its budget by at most the cost of the requests in flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from gateway.config import BudgetConfig


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
