"""Logging set-up: plain text for development, one JSON object per line for production (GATEWAY_LOG_FORMAT=json).

Each finished request writes one ``gateway.access`` line with its outcome, model, tokens, cost and latency. It
never contains prompt or answer text, API keys or provider error messages (those stay in the request log).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from gateway.monitoring.store import RequestRecord

access_log = logging.getLogger("gateway.access")

_ACCESS_FIELDS = ("app_id", "status", "http_status", "error_type", "release", "canary", "mode", "tier", "model",
                  "latency_class", "fallback", "escalated", "stream", "prompt_tokens", "completion_tokens",
                  "cost_usd", "latency_ms", "router_ms")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            data.update(fields)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO")
    if os.environ.get("GATEWAY_LOG_FORMAT", "").strip().lower() == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def log_access(record: RequestRecord) -> None:
    fields = {"request_id": record.id, **{name: getattr(record, name) for name in _ACCESS_FIELDS},
              "provider_calls": len(record.calls)}
    access_log.info("request %s %s %s", record.id, record.status, record.http_status, extra={"fields": fields})
