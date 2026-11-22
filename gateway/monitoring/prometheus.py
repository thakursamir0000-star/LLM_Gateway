"""Prometheus metrics in the text exposition format (version 0.0.4), with no client library.

Every finished request is observed once, from its request record (``Gateway._finish``), so the metrics agree with
the request log. Labels carry no prompt text, keys or provider messages. Client-supplied app ids (anonymous mode)
are capped at ``MAX_APPS`` distinct values; later ones are counted as ``other``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from typing import Any

from gateway.monitoring.store import RequestRecord

MAX_APPS = 100
DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\n").replace('"', '\\"')


def _labels(names: tuple[str, ...], values: tuple[str, ...], extra: str = "") -> str:
    parts = [f'{n}="{_escape(v)}"' for n, v in zip(names, values, strict=True)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


def _number(value: float) -> str:
    if math.isinf(value):
        return "+Inf"
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


class Counter:
    def __init__(self, name: str, help_: str, labels: tuple[str, ...]):
        self.name, self.help, self.label_names = name, help_, labels
        self.values: dict[tuple[str, ...], float] = {}

    def inc(self, labels: tuple[str, ...], amount: float = 1.0) -> None:
        self.values[labels] = self.values.get(labels, 0.0) + amount

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        for labels, value in sorted(self.values.items()):
            yield f"{self.name}{_labels(self.label_names, labels)} {_number(value)}"


class Histogram:
    def __init__(self, name: str, help_: str, labels: tuple[str, ...], buckets: tuple[float, ...] = DURATION_BUCKETS):
        self.name, self.help, self.label_names, self.buckets = name, help_, labels, buckets
        self.series: dict[tuple[str, ...], list[float]] = {}  # cumulative bucket counts, +Inf count, sum

    def observe(self, labels: tuple[str, ...], value: float) -> None:
        row = self.series.setdefault(labels, [0.0] * (len(self.buckets) + 2))
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                row[i] += 1
        row[-2] += 1
        row[-1] += value

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        for labels, row in sorted(self.series.items()):
            for bound, count in zip((*self.buckets, math.inf), row[:-1], strict=True):
                le = 'le="' + _number(bound) + '"'
                yield f"{self.name}_bucket{_labels(self.label_names, labels, le)} {_number(count)}"
            yield f"{self.name}_sum{_labels(self.label_names, labels)} {_number(row[-1])}"
            yield f"{self.name}_count{_labels(self.label_names, labels)} {_number(row[-2])}"


class Metrics:
    """The gateway's metrics for one process (each replica is scraped on its own)."""

    def __init__(self, version: str = "") -> None:
        self._lock = threading.Lock()
        self.version = version
        self._apps: set[str] = set()
        self.requests = Counter("gateway_requests_total", "Finished requests by app, outcome and tier.",
                                ("app", "status", "tier", "error_type", "stream"))
        self.duration = Histogram("gateway_request_duration_seconds", "End-to-end request latency.", ("status",))
        self.tokens = Counter("gateway_tokens_total", "Tokens used, by app, model and kind.", ("app", "model", "kind"))
        self.cost = Counter("gateway_cost_usd_total", "Cost at list prices in USD, by app and model.",
                            ("app", "model"))
        self.calls = Counter("gateway_provider_calls_total", "Provider calls by model and outcome.",
                             ("model", "status", "error_type"))
        self.call_duration = Histogram("gateway_provider_call_duration_seconds", "Provider call latency.",
                                       ("model",))
        self.fallbacks = Counter("gateway_fallbacks_total", "Requests answered by a fallback model.", ("app",))
        self.escalations = Counter("gateway_escalations_total", "Cascade escalations to the strong tier.", ("app",))
        self.router_decisions = Counter("gateway_router_decisions_total",
                                        "Router-mode requests by who decided (classifier, rules, control) and tier.",
                                        ("router", "tier"))
        self.shadow = Counter("gateway_shadow_decisions_total",
                              "Shadow-scored requests: the tier used and the tier the classifier would have chosen.",
                              ("tier", "shadow_tier"))

    def _app(self, app_id: str) -> str:
        if app_id in self._apps:
            return app_id
        if len(self._apps) < MAX_APPS:
            self._apps.add(app_id)
            return app_id
        return "other"

    def observe(self, record: RequestRecord) -> None:
        with self._lock:
            app = self._app(record.app_id or "")
            self.requests.inc((app, record.status, record.tier or "", record.error_type or "",
                               "true" if record.stream else "false"))
            if record.latency_ms is not None:
                self.duration.observe((record.status,), record.latency_ms / 1000)
            if record.fallback:
                self.fallbacks.inc((app,))
            if record.escalated:
                self.escalations.inc((app,))
            if record.mode == "router" and record.router:
                self.router_decisions.inc((record.router, record.tier or ""))
            if record.shadow_tier:
                self.shadow.inc((record.tier or "", record.shadow_tier))
            for call in record.calls:
                model = call.model or ""
                self.calls.inc((model, call.status, call.error_type or ""))
                if call.latency_ms is not None:
                    self.call_duration.observe((model,), call.latency_ms / 1000)
                if call.prompt_tokens:
                    self.tokens.inc((app, model, "prompt"), call.prompt_tokens)
                if call.completion_tokens:
                    self.tokens.inc((app, model, "completion"), call.completion_tokens)
                if call.cost_usd:
                    self.cost.inc((app, model), call.cost_usd)

    def render(self, state_status: dict[str, Any] | None = None) -> str:
        with self._lock:
            lines: list[str] = [
                "# HELP gateway_build_info Gateway version.",
                "# TYPE gateway_build_info gauge",
                f'gateway_build_info{{version="{_escape(self.version)}"}} 1',
            ]
            if state_status is not None:
                backend = _escape(str(state_status.get("backend", "")))
                lines += ["# HELP gateway_state_degraded 1 while the shared state backend (Redis) is unreachable.",
                          "# TYPE gateway_state_degraded gauge",
                          f'gateway_state_degraded{{backend="{backend}"}} {1 if state_status.get("degraded") else 0}']
            for metric in (self.requests, self.duration, self.tokens, self.cost, self.calls, self.call_duration,
                           self.fallbacks, self.escalations, self.router_decisions, self.shadow):
                lines.extend(metric.render())
            return "\n".join(lines) + "\n"
