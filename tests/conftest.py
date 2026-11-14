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
               "check_interval_s": 0},
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
