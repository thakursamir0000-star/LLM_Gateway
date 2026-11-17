"""Simple alerts: error rate over a recent window, and today's total cost.

``AlertMonitor.maybe_check`` runs after each request, at most once per
``check_interval_s``. A firing alert is logged, stored in the ``alerts`` table
(the dashboard shows it) and sent to the notifier, which POSTs it as JSON to
$ALERT_WEBHOOK_URL when that is set. To avoid repeats, an error-rate alert
fires at most once per window and a cost alert once per UTC day.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from gateway.config import AlertConfig
from gateway.monitoring.store import Store, iso, utc_now

log = logging.getLogger("gateway.alerts")


@dataclass(frozen=True)
class Alert:
    kind: str  # error_rate | daily_cost
    value: float
    threshold: float
    message: str


def evaluate(store: Store, config: AlertConfig, now: datetime) -> list[Alert]:
    """Alerts that are over their limit right now (whether or not already sent)."""
    alerts = []
    since = iso(now - timedelta(minutes=config.window_minutes))
    row = store.query("SELECT COUNT(*) AS n, COALESCE(SUM(status = 'error'), 0) AS errors FROM requests "
                      "WHERE ts >= ? AND status != 'rejected'", (since,))[0]
    if row["n"] >= config.min_requests:
        rate = row["errors"] / row["n"]
        if rate > config.error_rate:
            alerts.append(Alert("error_rate", rate, config.error_rate,
                                f"Error rate {rate:.0%} over the last {config.window_minutes} min "
                                f"({row['errors']}/{row['n']} requests) is above {config.error_rate:.0%}"))
    if config.daily_cost_usd is not None:
        cost = store.total_spend(iso(now)[:10])
        if cost > config.daily_cost_usd:
            alerts.append(Alert("daily_cost", cost, config.daily_cost_usd,
                                f"Cost today ${cost:.4f} is above the daily limit ${config.daily_cost_usd:.2f}"))
    return alerts


def log_notifier(alert: Alert) -> None:
    log.warning("ALERT %s: %s", alert.kind, alert.message)


def webhook_notifier(url: str, timeout_s: float = 5.0) -> Callable[[Alert], None]:
    """Log the alert and POST it as JSON to ``url`` in a background thread."""
    import httpx

    def post(alert: Alert) -> None:
        try:
            httpx.post(url, json={"source": "llm-gateway", **asdict(alert)}, timeout=timeout_s)
        except httpx.HTTPError as e:
            log.warning("alert webhook failed: %s: %s", type(e).__name__, e)

    def notify(alert: Alert) -> None:
        log_notifier(alert)
        threading.Thread(target=post, args=(alert,), daemon=True).start()

    return notify


class AlertMonitor:
    def __init__(self, store: Store, config: AlertConfig, notify: Callable[[Alert], None] = log_notifier,
                 clock: Callable[[], datetime] = utc_now):
        self.store = store
        self.config = config
        self.notify = notify
        self.clock = clock
        self._last_check: datetime | None = None
        self._lock = threading.Lock()

    def maybe_check(self) -> list[Alert]:
        now = self.clock()
        with self._lock:
            due = self._last_check is None or (now - self._last_check).total_seconds() >= self.config.check_interval_s
            if not due:
                return []
            self._last_check = now
        return self.check(now)

    def _is_new(self, alert: Alert, now: datetime) -> bool:
        last = self.store.last_alert(alert.kind)
        if last is None:
            return True
        if alert.kind == "daily_cost":
            return last["ts"][:10] != iso(now)[:10]
        return last["ts"] < iso(now - timedelta(minutes=self.config.window_minutes))

    def check(self, now: datetime | None = None) -> list[Alert]:
        """Evaluate, then store and send the alerts that haven't been sent recently."""
        now = now or self.clock()
        fired = []
        for alert in evaluate(self.store, self.config, now):
            if self._is_new(alert, now):
                self.store.add_alert(alert.kind, alert.value, alert.threshold, alert.message, iso(now))
                self.notify(alert)
                fired.append(alert)
        return fired
