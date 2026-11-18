"""Focused security tests for LLM Gateway Phase 1 Hardening.

Covers:
1. Client authentication fail-closed by default (no anonymous access without keys).
2. Bearer key authentication and invalid key rejection.
3. Admin endpoints fail-closed when GATEWAY_ADMIN_KEY is unset, and token verification.
4. Sensitive internal routing metadata and diagnostic header suppression.
5. Request body size limits (Content-Length and streamed body chunk enforcement).
6. CORS configuration with restrictive production defaults.
7. X-Admin-Key does not authorize diagnostic metadata on chat endpoints.
"""

from __future__ import annotations

import random
from typing import Any

from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.monitoring.store import Store
from gateway.providers.fake import FakeProvider
from gateway.routing import RulesRouter
from tests.conftest import make_config

EASY = {"messages": [{"role": "user", "content": "Hello"}]}


def build_app(config: Any = None, **overrides: Any) -> TestClient:
    cfg = config or make_config(**overrides)
    providers = {"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini")}
    store = Store(":memory:")
    app = create_app(cfg, providers=providers, store=store, router=RulesRouter(), rng=random.Random(0))
    return TestClient(app)


# ── 1. Client authentication fail-closed ────────────────────────────


def test_auth_fails_closed_by_default_when_no_keys_configured(monkeypatch):
    """When allow_anonymous is false (default) and no keys are set, chat requests are rejected."""
    monkeypatch.delenv("TEST_GATEWAY_KEYS", raising=False)
    config = make_config(auth={"allow_anonymous": False})
    client = build_app(config)

    response = client.post("/v1/chat/completions", json=EASY)
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "auth_required"
    assert "authentication required" in body["error"]["message"].lower()

    # Feedback endpoint also fails closed
    fb_response = client.post("/v1/feedback", json={"request_id": "any-id", "score": 1})
    assert fb_response.status_code == 401
    assert fb_response.json()["error"]["code"] == "auth_required"


def test_auth_with_configured_keys_requires_valid_bearer_token(monkeypatch):
    monkeypatch.setenv("TEST_GATEWAY_KEYS", "prod-app:secret-token-123")
    config = make_config(auth={"allow_anonymous": False})
    client = build_app(config)

    # Missing Authorization header
    res_no_auth = client.post("/v1/chat/completions", json=EASY)
    assert res_no_auth.status_code == 401
    assert res_no_auth.json()["error"]["code"] == "invalid_api_key"

    # Wrong token
    res_wrong = client.post("/v1/chat/completions", json=EASY, headers={"Authorization": "Bearer wrong-token"})
    assert res_wrong.status_code == 401
    assert res_wrong.json()["error"]["code"] == "invalid_api_key"

    # Valid token succeeds
    res_ok = client.post("/v1/chat/completions", json=EASY, headers={"Authorization": "Bearer secret-token-123"})
    assert res_ok.status_code == 200
    assert res_ok.json()["object"] == "chat.completion"


def test_explicit_allow_anonymous_permits_x_app_id(monkeypatch):
    """When allow_anonymous is explicitly enabled, anonymous calls with X-App-Id are allowed."""
    monkeypatch.delenv("TEST_GATEWAY_KEYS", raising=False)
    config = make_config(auth={"allow_anonymous": True})
    client = build_app(config)

    response = client.post("/v1/chat/completions", json=EASY, headers={"X-App-Id": "guest-user"})
    assert response.status_code == 200


# ── 2. Admin endpoint protection ────────────────────────────────────


def test_admin_endpoints_fail_closed_when_admin_key_is_unset(monkeypatch):
    monkeypatch.delenv("TEST_GATEWAY_ADMIN_KEY", raising=False)
    client = build_app()

    for path in ["/admin/status", "/admin/deployments", "/admin/alerts"]:
        res = client.get(path)
        assert res.status_code == 403
        assert res.json()["error"]["code"] == "admin_disabled"

    res_act = client.post("/admin/activate", json={"release": "v1"})
    assert res_act.status_code == 403
    assert res_act.json()["error"]["code"] == "admin_disabled"

    res_rb = client.post("/admin/rollback", json={})
    assert res_rb.status_code == 403
    assert res_rb.json()["error"]["code"] == "admin_disabled"

    res_can = client.post("/admin/canary", json={"release": "v2", "share": 0.1})
    assert res_can.status_code == 403
    assert res_can.json()["error"]["code"] == "admin_disabled"

    res_del = client.delete("/admin/canary")
    assert res_del.status_code == 403
    assert res_del.json()["error"]["code"] == "admin_disabled"


def test_admin_endpoints_reject_invalid_key_and_accept_valid_key(monkeypatch):
    monkeypatch.setenv("TEST_GATEWAY_ADMIN_KEY", "super-admin-key-999")
    client = build_app()

    # Unauthorized without header
    res_no_key = client.get("/admin/status")
    assert res_no_key.status_code == 401
    assert res_no_key.json()["error"]["code"] == "invalid_admin_key"

    # Unauthorized with wrong key
    res_wrong = client.get("/admin/status", headers={"X-Admin-Key": "wrong-key"})
    assert res_wrong.status_code == 401
    assert res_wrong.json()["error"]["code"] == "invalid_admin_key"

    # Authorized with correct key
    res_ok = client.get("/admin/status", headers={"X-Admin-Key": "super-admin-key-999"})
    assert res_ok.status_code == 200
    assert "deployment" in res_ok.json()


# ── 3. Internal metadata exposure & OpenAI compatibility ───────────


def test_internal_metadata_hidden_by_default():
    """In production mode (expose_internal_metadata: false), the gateway block and diagnostic headers are omitted."""
    config = make_config(security={"expose_internal_metadata": False}, auth={"allow_anonymous": True})
    client = build_app(config)

    res = client.post("/v1/chat/completions", json=EASY)
    assert res.status_code == 200
    body = res.json()

    # Standard OpenAI fields must be present
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert "model" in body
    assert "choices" in body and len(body["choices"]) == 1
    assert "usage" in body

    # Non-standard internal gateway block MUST be omitted
    assert "gateway" not in body

    # Diagnostic headers must be stripped; only standard request id remains
    assert "x-gateway-request-id" in res.headers
    assert "x-gateway-model" not in res.headers
    assert "x-gateway-tier" not in res.headers
    assert "x-gateway-release" not in res.headers
    assert "x-gateway-cost-usd" not in res.headers


def test_internal_metadata_included_when_explicitly_configured():
    config = make_config(security={"expose_internal_metadata": True}, auth={"allow_anonymous": True})
    client = build_app(config)

    res = client.post("/v1/chat/completions", json=EASY)
    assert res.status_code == 200
    body = res.json()
    assert "gateway" in body
    assert "tier" in body["gateway"]
    assert "x-gateway-model" in res.headers
    assert "x-gateway-cost-usd" in res.headers


def test_admin_key_does_not_authorize_diagnostics_on_chat_endpoint(monkeypatch):
    """Never use X-Admin-Key as authorization for diagnostic metadata on normal chat endpoints."""
    monkeypatch.setenv("TEST_GATEWAY_ADMIN_KEY", "admin-key-val")
    config = make_config(security={"expose_internal_metadata": False}, auth={"allow_anonymous": True})
    client = build_app(config)

    res = client.post("/v1/chat/completions", json=EASY, headers={"X-Admin-Key": "admin-key-val"})
    assert res.status_code == 200
    body = res.json()
    assert "gateway" not in body
    assert "x-gateway-model" not in res.headers


# ── 4. Request body size limits ─────────────────────────────────────


def test_request_body_size_limit_via_content_length():
    # 1024-byte limit
    config = make_config(security={"max_request_body_bytes": 1024}, auth={"allow_anonymous": True})
    client = build_app(config)

    # Large payload exceeding limit
    big_message = "x" * 1500
    res = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": big_message}]})
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "payload_too_large"


def test_request_body_size_limit_via_streamed_chunks():
    # 1024-byte limit
    config = make_config(security={"max_request_body_bytes": 1024}, auth={"allow_anonymous": True})
    client = build_app(config)

    def payload_generator():
        # Yield chunks without Content-Length
        yield b'{"messages": [{"role": "user", "content": "'
        for _ in range(40):
            yield b"streaming chunk of payload that pushes past limit "
        yield b'"}]}'

    res = client.post("/v1/chat/completions", content=payload_generator(),
                      headers={"Content-Type": "application/json"})
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "payload_too_large"


# ── 5. CORS configuration ───────────────────────────────────────────


def test_cors_restrictive_by_default():
    """Default cors_origins is empty: cross-origin requests do not receive access-control-allow-origin."""
    config = make_config(security={"cors_origins": []}, auth={"allow_anonymous": True})
    client = build_app(config)

    res = client.options("/v1/chat/completions", headers={
        "Origin": "https://malicious-site.example.com",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in res.headers


def test_cors_allows_configured_origins():
    config = make_config(security={"cors_origins": ["https://trusted-app.internal"]}, auth={"allow_anonymous": True})
    client = build_app(config)

    # Allowed origin
    res = client.options("/v1/chat/completions", headers={
        "Origin": "https://trusted-app.internal",
        "Access-Control-Request-Method": "POST",
    })
    assert res.headers.get("access-control-allow-origin") == "https://trusted-app.internal"

    # Disallowed origin
    res_bad = client.options("/v1/chat/completions", headers={
        "Origin": "https://untrusted.com",
        "Access-Control-Request-Method": "POST",
    })
    assert "access-control-allow-origin" not in res_bad.headers
