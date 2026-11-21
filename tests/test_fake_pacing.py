"""Fake-provider mode ignores per-model rate limits unless GATEWAY_FAKE_PACING asks for them."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.monitoring.store import Store
from tests.conftest import make_config

LIMITED = {
    "small": {"provider": "groq", "model": "openai/gpt-oss-20b", "limits": {"rpm": 1}},
    "large": {"provider": "groq", "model": "openai/gpt-oss-120b", "limits": {"rpm": 1}},
    "gemma": {"provider": "gemini", "model": "gemma-4-31b-it"},
}


@pytest.mark.parametrize("pacing, second_is_rate_limited", [(None, False), ("1", True)])
def test_fake_pacing_switch(tmp_path, monkeypatch, pacing, second_is_rate_limited):
    monkeypatch.setenv("GATEWAY_FAKE_PROVIDERS", "1")
    if pacing:
        monkeypatch.setenv("GATEWAY_FAKE_PACING", pacing)
    else:
        monkeypatch.delenv("GATEWAY_FAKE_PACING", raising=False)
    config = make_config(models=LIMITED, retries={"attempts": 1, "max_wait_s": 0})
    store = Store(tmp_path / "gateway.db")
    try:
        with TestClient(create_app(config, store=store)) as client:
            codes = [client.post("/v1/chat/completions", headers={"X-App-Id": "app"},
                                 json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
                     for _ in range(2)]
    finally:
        store.close()
    second = codes[1]
    limited = second.status_code == 429 and second.json()["error"]["code"] == "rate_limit_exceeded"
    assert limited is second_is_rate_limited
