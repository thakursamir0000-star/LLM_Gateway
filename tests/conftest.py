"""Shared test helpers. No test calls a real LLM API: providers are fakes or httpx mock transports."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from gateway.config import GatewayConfig

BASE_CONFIG: dict[str, Any] = {
    "models": {
        "small": {"provider": "groq", "model": "openai/gpt-oss-20b",
                  "price_per_1m": {"input": 0.10, "output": 0.50}, "params": {"reasoning_effort": "low"}},
        "large": {"provider": "groq", "model": "openai/gpt-oss-120b",
                  "price_per_1m": {"input": 0.15, "output": 0.75}},
        "gemma": {"provider": "gemini", "model": "gemma-4-31b-it"},
    },
    "providers": {
        "groq": {"base_url": "https://groq.test/openai/v1", "api_key_env": "TEST_GROQ_KEY"},
        "gemini": {"base_url": "https://gemini.test/v1beta", "api_key_env": "TEST_GEMINI_KEY"},
    },
    "prompts": {"plain@1": "", "concise@1": "Be concise."},
    "releases": {
        "v1": {"cheap": "small", "strong": "large", "prompt": "plain@1"},
        "v2": {"cheap": "gemma", "strong": "large", "prompt": "concise@1"},
    },
    "deployment": {"active": "v1"},
    "routing": {"default_mode": "router", "router": "rules", "threshold": 0.5},
    "retries": {"attempts": 3, "max_wait_s": 10},
    "budgets": {"default_daily_usd": 1.0, "soft_limit_fraction": 0.8, "apps": {"tiny": {"daily_usd": 0.001}}},
    "alerts": {"error_rate": 0.5, "window_minutes": 15, "min_requests": 4, "daily_cost_usd": 0.01,
               "check_interval_s": 0, "webhook_env": "TEST_ALERT_WEBHOOK"},
    # Test-only variable names, so a developer's own GATEWAY_* settings can't change the tests.
    # Isolated test compatibility settings: tests assert gateway internal metadata.
    "auth": {
        "keys_env": "TEST_GATEWAY_KEYS",
        "keys_file_env": "TEST_GATEWAY_KEYS_FILE",
        "admin_key_env": "TEST_GATEWAY_ADMIN_KEY",
        "allow_anonymous": True,
    },
    "security": {
        "max_request_body_bytes": 10485760,
        "cors_origins": [],
        "expose_internal_metadata": True,
    },
}


def make_config(**overrides: Any) -> GatewayConfig:
    """A test config; top-level keys in ``overrides`` replace (dicts are merged one level deep)."""
    data = copy.deepcopy(BASE_CONFIG)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return GatewayConfig.model_validate(data)


@pytest.fixture
def config() -> GatewayConfig:
    return make_config()


# A developer's shell may still hold the variables the README's local demo exports. Tests that load the shipped
# config would then pick them up (e.g. GATEWAY_API_KEYS turning anonymous test requests into 401s), so they are
# cleared for every test; a test that needs one sets it itself. GATEWAY_DATABASE_URL and GATEWAY_REDIS_URL are kept:
# CI sets them on purpose to point the suite at its PostgreSQL and Redis services.
_DEVELOPER_SETTINGS = (
    "GATEWAY_API_KEYS", "GATEWAY_API_KEYS_FILE", "GATEWAY_ADMIN_KEY", "GATEWAY_ALLOW_ANONYMOUS", "GATEWAY_ENV",
    "GATEWAY_FAKE_PROVIDERS", "GATEWAY_FAKE_PACING", "GATEWAY_EXPOSE_METADATA", "GATEWAY_METRICS_TOKEN",
    "GATEWAY_CONFIG", "GATEWAY_DB", "GATEWAY_STATE_BACKEND", "GATEWAY_REDIS_OUTAGE_POLICY", "GATEWAY_REPLICAS",
    "GATEWAY_MAX_BODY_BYTES", "GATEWAY_CORS_ORIGINS", "GATEWAY_LOG_FORMAT",
)


@pytest.fixture(autouse=True)
def _isolate_from_developer_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _DEVELOPER_SETTINGS:
        monkeypatch.delenv(name, raising=False)
