"""Regression tests for the final audit's streaming and error-handling findings.

C4: every streaming request is logged exactly once, including early rejections.
C5: clients get generic upstream errors; the provider's own message stays server-side.
C6: a request-log failure never turns a streaming response into a plain-text 500.
C7: a stream without provider usage is estimated, not counted as zero tokens.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from gateway.app import create_app
from gateway.config import GatewayConfig
from gateway.monitoring.store import Store
from gateway.providers import ModelRegistry, stream_tokens
from gateway.providers.base import (
    AUTH,
    BAD_REQUEST,
    OTHER,
    OVERLOADED,
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    TIMEOUT,
    Completion,
    Provider,
    ProviderError,
    StreamChunk,
)
from gateway.routing.rules import RulesRouter
from gateway.schemas import ChatRequest
from gateway.service import RequestContext, from_provider_error

MSG = [{"role": "user", "content": "Explain the plan in one line, please."}]
# A provider error with details a client must not see (all made up).
LEAKY = ("Groq HTTP 429: Rate limit reached for model `openai/gpt-oss-120b` in organization `org_[redacted]` "
         "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199990")


def make_config(**overrides: Any) -> GatewayConfig:
    data: dict[str, Any] = {
        "models": {
            "m-cheap": {"provider": "p1", "model": "cheap-model", "price_per_1m": {"input": 0.1, "output": 0.2}},
            "m-strong": {"provider": "p2", "model": "strong-model", "price_per_1m": {"input": 0.5, "output": 1.0}},
        },
        "providers": {
            "p1": {"base_url": "http://p1", "api_key_env": "KEY"},
            "p2": {"base_url": "http://p2", "api_key_env": "KEY"},
        },
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m-cheap", "strong": "m-strong", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "routing": {"default_mode": "cheap", "fallback_on_error": True},
        "retries": {"attempts": 2, "max_wait_s": 0.01},
        "budgets": {"apps": {"broke": {"daily_usd": 0.0}}},
        "auth": {"allow_anonymous": True},
        "security": {"expose_internal_metadata": False},
        **overrides,
    }
    return GatewayConfig.model_validate(data)


class Scripted(Provider):
    """Streams ``words`` (and usage when given), or raises ``error`` before the first chunk."""

    def __init__(self, words: tuple[str, ...] = ("Hello", " there", " friend"), *, error: Exception | None = None,
                 usage: tuple[int, int] | None = None):
        self.words, self.error, self.usage = words, error, usage

    async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        if self.error:
            raise self.error
        return Completion("".join(self.words), "stop", 10, 10, model)

    async def stream(self, model: str, messages: list[dict[str, Any]],
                     params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        if self.error:
            raise self.error
        for word in self.words:
            yield StreamChunk(text=word, model=model)
        if self.usage:
            yield StreamChunk(finish_reason="stop", model=model, prompt_tokens=self.usage[0],
                              completion_tokens=self.usage[1])


def build(tmp_path, providers: dict[str, Provider], store: Store | None = None, **overrides: Any):
    store = store or Store(tmp_path / "gateway.db")
    app = create_app(make_config(**overrides), providers=providers, router=RulesRouter(), store=store)
    return TestClient(app), store


def rows(store: Store) -> list[dict[str, Any]]:
    return store.query("SELECT * FROM requests ORDER BY ts")


def post(client: TestClient, stream: bool, **body: Any):
    headers = body.pop("headers", {})
    return client.post("/v1/chat/completions", json={"model": "cheap", "messages": MSG, "stream": stream, **body},
                       headers=headers)


# ── C4: every streaming request is logged once ─────────────────────────


@pytest.mark.parametrize("body, status, error_type", [
    ({"n": 2}, 400, "invalid_request_error"),
    ({"tools": [{"type": "function", "function": {"name": "f"}}]}, 400, "invalid_request_error"),
    ({"model": "no-such-mode"}, 404, "model_not_found"),
    ({"headers": {"X-App-Id": "broke"}}, 429, "budget_exceeded"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_early_rejections_are_logged_once_on_both_paths(tmp_path, stream, body, status, error_type):
    client, store = build(tmp_path, {"p1": Scripted(), "p2": Scripted()})
    response = post(client, stream, **body)
    assert response.status_code == status
    assert response.json()["error"]["code"] in (error_type, None)  # OpenAI error format, not SSE
    logged = rows(store)
    assert len(logged) == 1
    row = logged[0]
    assert (row["stream"], row["http_status"], row["status"], row["error_type"]) == (int(stream), status,
                                                                                     "rejected", error_type)
    assert row["latency_ms"] is not None and row["id"].startswith("chatcmpl-")


def test_stream_failing_before_the_first_chunk_is_logged_once_with_every_attempt(tmp_path):
    down = ProviderError("upstream down", kind=OVERLOADED, status=503)
    client, store = build(tmp_path, {"p1": Scripted(error=down), "p2": Scripted(error=down)})
    response = post(client, True)
    assert response.status_code == 503 and response.json()["error"]["code"] == "upstream_unavailable"
    [row] = rows(store)
    assert (row["status"], row["error_type"], row["fallback"]) == ("error", "upstream_unavailable", 1)
    calls = store.query("SELECT model, status FROM calls WHERE request_id = ? ORDER BY seq", (row["id"],))
    assert [(c["model"], c["status"]) for c in calls] == [("m-cheap", "error"), ("m-strong", "error")]


def test_successful_stream_is_logged_once(tmp_path):
    client, store = build(tmp_path, {"p1": Scripted(usage=(9, 3)), "p2": Scripted()})
    with client.stream("POST", "/v1/chat/completions", json={"model": "cheap", "messages": MSG, "stream": True}) as r:
        lines = [line for line in r.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    [row] = rows(store)
    assert (row["status"], row["stream"], row["prompt_tokens"], row["completion_tokens"]) == ("ok", 1, 9, 3)


# ── C5: generic upstream errors for clients ───────────────────────────


@pytest.mark.parametrize("stream", [False, True])
def test_provider_error_details_stay_server_side(tmp_path, caplog, stream):
    caplog.set_level(logging.WARNING, logger="gateway")
    quota = ProviderError(LEAKY, kind=QUOTA_EXHAUSTED, status=429, retry_after=30)
    client, store = build(tmp_path, {"p1": Scripted(error=quota), "p2": Scripted(error=quota)})
    response = post(client, stream)
    assert response.status_code == 429 and response.headers["retry-after"] == "30"
    error = response.json()["error"]
    assert (error["type"], error["code"]) == ("rate_limit_error", "upstream_quota_exhausted")
    for secret in ("Groq", "gpt-oss-120b", "org_", "on_demand", "200000", "199990", "tokens per day"):
        assert secret not in response.text
    [row] = rows(store)
    assert "tokens per day" in row["error_message"]  # the detail is kept in the request log
    assert any("tokens per day" in r.getMessage() for r in caplog.records)  # and in the gateway's log


@pytest.mark.parametrize("kind, status, code", [
    (RATE_LIMIT, 429, "rate_limit_exceeded"), (QUOTA_EXHAUSTED, 429, "upstream_quota_exhausted"),
    (AUTH, 502, "upstream_auth_failed"), (BAD_REQUEST, 400, "upstream_bad_request"),
    (TIMEOUT, 504, "upstream_timeout"), (OVERLOADED, 503, "upstream_unavailable"), (OTHER, 502, "upstream_error"),
])
def test_every_upstream_error_kind_keeps_its_status_and_code_but_not_the_provider_text(kind, status, code):
    error = from_provider_error(ProviderError(f"provider says: {LEAKY}", kind=kind, status=status))
    assert (error.status, error.code) == (status, code)
    assert "provider says" not in error.message and "org_" not in error.message


def test_mid_stream_error_detail_is_redacted_in_the_log(tmp_path):
    class BreaksMidway(Scripted):
        async def stream(self, model, messages, params):
            yield StreamChunk(text="partial", model=model)
            raise ProviderError("socket closed for key sk-or-v1-0123456789abcdef", kind=OVERLOADED, status=503)

    client, store = build(tmp_path, {"p1": BreaksMidway(), "p2": Scripted()})
    with client.stream("POST", "/v1/chat/completions", json={"model": "cheap", "messages": MSG, "stream": True}) as r:
        body = r.read().decode()
    assert '"code": "stream_failed"' in body and "sk-or-v1" not in body
    [row] = rows(store)
    assert row["error_type"] == "stream_failed" and "sk-or-v1-0123456789abcdef" not in row["error_message"]


# ── C6: a request-log failure never breaks a streaming response ───────


class FailingStore(Store):
    def log_request(self, record):
        raise RuntimeError("simulated disk full")


@pytest.mark.parametrize("case", ["providers down", "early rejection", "success"])
def test_storage_failure_does_not_break_streaming_responses(tmp_path, caplog, case):
    caplog.set_level(logging.ERROR, logger="gateway")
    down = ProviderError("upstream down", kind=OVERLOADED, status=503)
    providers = ({"p1": Scripted(error=down), "p2": Scripted(error=down)} if case == "providers down"
                 else {"p1": Scripted(), "p2": Scripted()})
    client, _ = build(tmp_path, providers, store=FailingStore(tmp_path / "gateway.db"))
    if case == "success":
        with client.stream("POST", "/v1/chat/completions",
                           json={"model": "cheap", "messages": MSG, "stream": True}) as r:
            lines = [line for line in r.iter_lines() if line]
        assert r.status_code == 200 and lines[-1] == "data: [DONE]"
    else:
        response = post(client, True, **({"n": 2} if case == "early rejection" else {}))
        expected = 400 if case == "early rejection" else 503
        assert response.status_code == expected
        assert response.headers["content-type"].startswith("application/json") and "error" in response.json()
    assert any("could not log request" in r.getMessage() for r in caplog.records)


# ── C7: streams without provider usage ────────────────────────────────


async def test_stream_without_usage_is_estimated_for_the_rate_limit():
    config = make_config(models={"m-cheap": {"provider": "p1", "model": "cheap-model", "limits": {"tpm": 100000}},
                                 "m-strong": {"provider": "p2", "model": "strong-model"}})
    registry = ModelRegistry(config, {"p1": Scripted(("Hello", " there", " friend")), "p2": Scripted()})
    async for _ in registry.stream("m-cheap", MSG, {}):
        pass
    counted = sum(r.tokens for r in registry.limiters["m-cheap"]._window)
    assert counted == len(MSG[0]["content"]) // 4 + len("Hello there friend") // 4
    assert counted > 0


async def test_stream_usage_from_the_provider_is_not_double_counted():
    config = make_config(models={"m-cheap": {"provider": "p1", "model": "cheap-model", "limits": {"tpm": 100000}},
                                 "m-strong": {"provider": "p2", "model": "strong-model"}})
    registry = ModelRegistry(config, {"p1": Scripted(usage=(7, 11)), "p2": Scripted()})
    async for _ in registry.stream("m-cheap", MSG, {}):
        pass
    assert sum(r.tokens for r in registry.limiters["m-cheap"]._window) == 18


def test_stream_token_estimate_rules():
    assert stream_tokens(400, 0, None, None, produced=False) == 0  # failed before the first chunk
    assert stream_tokens(400, 3, None, None, produced=True) == 101  # short text still counts one token
    assert stream_tokens(400, 80, 50, None, produced=True) == 70  # reported prompt + estimated output
    assert stream_tokens(400, 80, 50, 9, produced=True) == 59  # full usage wins, no estimate added


def test_gateway_estimates_and_reports_usage_for_a_stream_without_usage(tmp_path):
    client, store = build(tmp_path, {"p1": Scripted(("Hello", " there", " friend")), "p2": Scripted()})
    payload = {"model": "cheap", "messages": MSG, "stream": True, "stream_options": {"include_usage": True}}
    with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {")]
    usage = events[-1]["usage"]
    prompt, completion = len(MSG[0]["content"]) // 4, len("Hello there friend") // 4
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (prompt, completion)
    [row] = rows(store)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (prompt, completion)
    assert row["cost_usd"] == pytest.approx((prompt * 0.1 + completion * 0.2) / 1e6)


async def test_a_stream_cancelled_midway_is_charged_for_what_it_produced(tmp_path):
    store = Store(tmp_path / "gateway.db")
    app = create_app(make_config(), providers={"p1": Scripted(("one", " two", " three")), "p2": Scripted()},
                     router=RulesRouter(), store=store)
    gateway = app.state.gateway
    request = ChatRequest(model="cheap", messages=MSG, stream=True)
    events, _ = await gateway.stream(request, RequestContext(app_id="default"))
    assert (await anext(events)).startswith("data: ")
    await events.aclose()  # the client went away after the first chunk
    [row] = rows(store)
    assert (row["status"], row["error_type"]) == ("error", "cancelled")
    assert row["prompt_tokens"] == len(MSG[0]["content"]) // 4 and row["completion_tokens"] >= 1
    assert row["cost_usd"] > 0
