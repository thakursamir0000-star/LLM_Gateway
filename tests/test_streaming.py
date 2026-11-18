"""Tests for genuine provider streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from gateway.app import create_app
from gateway.config import GatewayConfig
from gateway.monitoring.store import Store
from gateway.providers.base import (
    OVERLOADED,
    QUOTA_EXHAUSTED,
    Completion,
    Provider,
    ProviderError,
    StreamChunk,
)
from gateway.providers.fake import FakeProvider
from gateway.routing.rules import RulesRouter


async def no_sleep(_: float) -> None:
    pass


def make_config(**overrides: Any) -> GatewayConfig:
    data: dict[str, Any] = {
        "models": {
            "m-cheap": {"provider": "p1", "model": "cheap-model", "price_per_1m": {"input": 0.1, "output": 0.2}},
            "m-strong": {"provider": "p2", "model": "strong-model", "price_per_1m": {"input": 0.5, "output": 1.0}},
            "m-backup": {"provider": "p3", "model": "backup-model", "price_per_1m": {"input": 0.1, "output": 0.2}},
        },
        "providers": {
            "p1": {"base_url": "http://p1", "api_key_env": "KEY"},
            "p2": {"base_url": "http://p2", "api_key_env": "KEY"},
            "p3": {"base_url": "http://p3", "api_key_env": "KEY"},
        },
        "prompts": {"p": ""},
        "releases": {
            "v1": {
                "cheap": "m-cheap",
                "strong": "m-strong",
                "prompt": "p",
                "batch": {"cheap": "m-backup", "strong": "m-strong"},
            }
        },
        "deployment": {"active": "v1"},
        "routing": {"fallback_on_error": True},
        "retries": {"attempts": 2, "max_wait_s": 0.01},
        "auth": {"allow_anonymous": True},
        "security": {"expose_internal_metadata": False},
        **overrides,
    }
    return GatewayConfig.model_validate(data)


@pytest.mark.asyncio
async def test_first_chunk_emitted_before_upstream_completes():
    """Verify downstream receives the first chunk while upstream is still waiting."""
    upstream_completed = False

    class StagedStreamProvider(Provider):
        name = "staged"
        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            return Completion("text", "stop", 5, 5, model)

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            nonlocal upstream_completed
            yield StreamChunk(text="FirstChunk", model=model)
            await asyncio.sleep(0.01)
            yield StreamChunk(text="SecondChunk", finish_reason="stop", model=model, prompt_tokens=5, completion_tokens=5)
            upstream_completed = True

    prov = StagedStreamProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "staged", "model": "staged-model"}},
        "providers": {"staged": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "auth": {"allow_anonymous": True},
    })
    from gateway.manage.budgets import BudgetPolicy
    from gateway.manage.releases import ReleaseManager
    from gateway.providers import ModelRegistry
    from gateway.schemas import ChatRequest
    from gateway.service import Gateway, RequestContext

    registry = ModelRegistry(cfg, {"staged": prov})
    store = Store(":memory:")
    releases = ReleaseManager(cfg, store)
    gw = Gateway(cfg, registry, store, RulesRouter(), releases, BudgetPolicy(cfg.budgets))

    req = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    ctx = RequestContext()

    generator, headers = await gw.stream(req, ctx)

    # First downstream event received
    first_event = await anext(generator)
    assert "FirstChunk" in first_event
    # Upstream has NOT completed when the first chunk is emitted!
    assert upstream_completed is False

    # Complete the stream
    remaining = [chunk async for chunk in generator]
    assert upstream_completed is True
    assert any("SecondChunk" in c for c in remaining)
    assert remaining[-1] == "data: [DONE]\n\n"


def test_valid_sse_chunks_and_done(tmp_path):
    store = Store(tmp_path / "gw.db")
    p1 = FakeProvider("p1")
    cfg = make_config()
    app = create_app(cfg, providers={"p1": p1, "p2": FakeProvider("p2"), "p3": FakeProvider("p3")},
                     store=store, router=RulesRouter(), sleep=no_sleep)
    client = TestClient(app)

    response = client.post("/v1/chat/completions",
                           json={"model": "m-cheap", "messages": [{"role": "user", "content": "ping"}],
                                 "stream": True, "stream_options": {"include_usage": True}})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"

    chunks = [json.loads(d) for d in data_lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)

    # Check content reconstruction
    deltas = [c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]]
    full_text = "".join(deltas)
    assert full_text.startswith("[cheap-model] fake answer to: ping")

    # Check final chunk usage
    usage_chunk = [c for c in chunks if not c["choices"] and "usage" in c]
    assert len(usage_chunk) == 1
    assert usage_chunk[0]["usage"]["total_tokens"] > 0

    # Internal metadata suppressed by default
    assert "gateway" not in usage_chunk[0]
    assert "x-gateway-model" not in response.headers
    assert "x-gateway-request-id" in response.headers


@pytest.mark.asyncio
async def test_upstream_cleanup_on_completion_and_disconnect():
    """Verify upstream generator aclose is called."""
    cleanup_called = False

    class CleanupProvider(Provider):
        name = "clean"
        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            return Completion("text", "stop", 5, 5, model)

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            nonlocal cleanup_called
            try:
                yield StreamChunk(text="chunk1", model=model)
                yield StreamChunk(text="chunk2", finish_reason="stop", model=model, prompt_tokens=5, completion_tokens=5)
            finally:
                cleanup_called = True

    prov = CleanupProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "clean", "model": "m"}},
        "providers": {"clean": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "auth": {"allow_anonymous": True},
    })
    store = Store(":memory:")
    app = create_app(cfg, providers={"clean": prov}, store=store, router=RulesRouter(), sleep=no_sleep)
    client = TestClient(app)

    response = client.post("/v1/chat/completions",
                           json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200
    assert cleanup_called is True


def test_pre_stream_upstream_failure_uses_fallback(tmp_path):
    """When the first provider fails before sending any chunk, fallback to the next provider."""
    class FailingProvider(Provider):
        name = "p1"
        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise ProviderError("503 server error", kind=OVERLOADED, status=503)

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            raise ProviderError("503 stream fail", kind=OVERLOADED, status=503)
            yield StreamChunk()

    p1 = FailingProvider()
    p2 = FakeProvider("p2")
    p3 = FakeProvider("p3")
    store = Store(tmp_path / "gw.db")
    cfg = make_config()
    app = create_app(cfg, providers={"p1": p1, "p2": p2, "p3": p3}, store=store, router=RulesRouter(), sleep=no_sleep)
    client = TestClient(app)

    response = client.post("/v1/chat/completions",
                           json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(d) for d in data_lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert "fake answer" in text

    # DB record indicates fallback occurred
    req = store.query("SELECT * FROM requests")[0]
    assert req["fallback"] == 1
    assert req["status"] == "ok"


def test_exhausted_pre_stream_read_error_uses_fallback(tmp_path):
    """An exhausted raw ReadError before the first chunk falls back to another provider."""
    calls = 0

    class ReadErrorProvider(Provider):
        name = "p1"

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            nonlocal calls
            calls += 1
            raise httpx.ReadError("read failed", request=httpx.Request("POST", "http://p1/chat"))
            yield StreamChunk()

    store = Store(tmp_path / "gw.db")
    cfg = make_config(retries={"attempts": 2, "max_wait_s": 10.0})
    app = create_app(cfg, providers={"p1": ReadErrorProvider(), "p2": FakeProvider("p2"), "p3": FakeProvider("p3")},
                     store=store, router=RulesRouter(), sleep=no_sleep)
    client = TestClient(app)

    response = client.post("/v1/chat/completions",
                           json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(d) for d in data_lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert "fake answer" in text
    assert calls == 2

    req = store.query("SELECT * FROM requests")[0]
    assert req["fallback"] == 1
    assert req["status"] == "ok"


def test_exhausted_pre_stream_read_error_without_fallback_returns_503(tmp_path):
    """Without fallback, an exhausted raw ReadError is a safe upstream 503."""
    calls = 0

    class ReadErrorProvider(Provider):
        name = "p1"

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            nonlocal calls
            calls += 1
            raise httpx.ReadError("sensitive read failure", request=httpx.Request("POST", "http://p1/chat"))
            yield StreamChunk()

    store = Store(tmp_path / "gw.db")
    cfg = make_config(retries={"attempts": 2, "max_wait_s": 10.0}, routing={"fallback_on_error": False})
    app = create_app(cfg, providers={"p1": ReadErrorProvider(), "p2": FakeProvider("p2"), "p3": FakeProvider("p3")},
                     store=store, router=RulesRouter(), sleep=no_sleep)
    client = TestClient(app)

    response = client.post("/v1/chat/completions",
                           json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    assert error["code"] == "upstream_unavailable"
    assert "ReadError" not in response.text
    assert "sensitive read failure" not in response.text
    assert calls == 2

    req = store.query("SELECT * FROM requests")[0]
    assert req["status"] == "error"
    assert req["error_type"] == "upstream_unavailable"


def test_post_start_failure_does_not_trigger_fallback_and_reports_safe_error(tmp_path):
    """When a provider fails after the first chunk, fallback must NOT occur; a safe SSE error is reported."""
    p2_calls = 0

    class MidStreamFailProvider(Provider):
        name = "p1"
        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            return Completion("text", "stop", 5, 5, model)

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(text="FirstPartOfResponse", model=model)
            raise ProviderError("connection dropped mid-stream", kind=OVERLOADED, status=503)

    class TrackingProvider(FakeProvider):
        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            nonlocal p2_calls
            p2_calls += 1
            async for chunk in super().stream(model, messages, params):
                yield chunk

    p1 = MidStreamFailProvider()
    p2 = TrackingProvider("p2")
    p3 = TrackingProvider("p3")
    store = Store(tmp_path / "gw.db")
    cfg = make_config()
    app = create_app(cfg, providers={"p1": p1, "p2": p2, "p3": p3}, store=store, router=RulesRouter(), sleep=no_sleep)
    client = TestClient(app)

    response = client.post("/v1/chat/completions",
                           json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert response.status_code == 200

    data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"

    # P2 was never called because post-start failure prevents fallback
    assert p2_calls == 0

    # Mid-stream error event was sent
    error_chunks = [json.loads(d) for d in data_lines if "error" in d]
    assert len(error_chunks) == 1
    assert error_chunks[0]["error"]["code"] == "stream_failed"
    assert error_chunks[0]["error"]["type"] == "upstream_error"

    # Database recorded the failure
    req = store.query("SELECT * FROM requests")[0]
    assert req["status"] == "error"
    assert req["error_type"] == "stream_failed"


def test_non_streaming_behavior_remains_unchanged(tmp_path):
    store = Store(tmp_path / "gw.db")
    cfg = make_config()
    app = create_app(cfg, providers={"p1": FakeProvider("p1"), "p2": FakeProvider("p2"), "p3": FakeProvider("p3")},
                     store=store, router=RulesRouter(), sleep=no_sleep)
    client = TestClient(app)

    response = client.post("/v1/chat/completions",
                           json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data and len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"].startswith("[cheap-model]")


@pytest.mark.asyncio
async def test_streaming_cancellation_cleans_up_upstream_and_does_not_emit_sse_error(tmp_path):
    """When a client cancels an active streaming request, upstream is closed and no SSE error is emitted."""
    from gateway.manage.budgets import BudgetPolicy
    from gateway.manage.releases import ReleaseManager
    from gateway.providers import ModelRegistry
    from gateway.schemas import ChatRequest
    from gateway.service import Gateway, RequestContext

    upstream_closed = asyncio.Event()
    chunk_1_yielded = asyncio.Event()

    class CancellableProvider(Provider):
        name = "cancellable"

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            return Completion("text", "stop", 5, 5, model)

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            try:
                chunk_1_yielded.set()
                yield StreamChunk(text="Chunk 1", model=model)
                await asyncio.sleep(10.0)
                yield StreamChunk(text="Chunk 2", finish_reason="stop", model=model)
            finally:
                upstream_closed.set()

    prov = CancellableProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "cancellable", "model": "m"}},
        "providers": {"cancellable": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "auth": {"allow_anonymous": True},
    })
    store = Store(tmp_path / "gw.db")
    releases = ReleaseManager(cfg, store)
    gw = Gateway(cfg, ModelRegistry(cfg, {"cancellable": prov}), store, RulesRouter(), releases, BudgetPolicy(cfg.budgets))

    req = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    ctx = RequestContext()

    generator, _ = await gw.stream(req, ctx)

    first_chunk = await anext(generator)
    assert "Chunk 1" in first_chunk
    assert chunk_1_yielded.is_set()

    # Cancel by closing the generator
    await generator.aclose()

    # Upstream response/generator must be closed
    assert upstream_closed.is_set()

    # Database records request as cancelled
    req_row = store.query("SELECT * FROM requests WHERE id = ?", [ctx.request_id])[0]
    assert req_row["status"] == "error"
    assert req_row["error_type"] == "cancelled"
    assert "cancelled" in req_row["error_message"]


@pytest.mark.asyncio
async def test_streaming_quota_error_sets_cooldown_and_skips_next_call():
    """A 429 quota error on stream activates the cooldown, skips the alias, and is not a breaker failure."""
    from gateway.providers import ModelRegistry

    class QuotaProvider(Provider):
        name = "quota"

        def __init__(self):
            self.calls = 0

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            raise ProviderError("daily quota exhausted", kind=QUOTA_EXHAUSTED, status=429, retry_after=45.0)
            yield StreamChunk()

    class FrozenClock:
        def __init__(self):
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

    prov = QuotaProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "quota", "model": "m"}},
        "providers": {"quota": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "retries": {"attempts": 3, "max_wait_s": 10.0, "quota_cooldown_s": 10.0},
        "auth": {"allow_anonymous": True},
    })
    registry = ModelRegistry(cfg, {"quota": prov}, clock=FrozenClock(), sleep=no_sleep)

    with pytest.raises(ProviderError) as first:
        await anext(registry.stream("m", [{"role": "user", "content": "hi"}], {}))
    assert first.value.kind == QUOTA_EXHAUSTED
    assert prov.calls == 1
    # max(configured quota_cooldown_s, provider retry_after)
    assert registry.cooldown_left("m") == pytest.approx(45.0)
    assert registry.circuit_breakers["quota"].consecutive_failures == 0

    with pytest.raises(ProviderError) as second:
        await anext(registry.stream("m", [{"role": "user", "content": "hi"}], {}))
    assert second.value.kind == QUOTA_EXHAUSTED
    assert second.value.retry_after == pytest.approx(45.0)
    assert "cooling down" in str(second.value)
    assert prov.calls == 1


@pytest.mark.asyncio
async def test_stream_retries_retryable_failures_before_first_chunk():
    """Retryable pre-first-chunk failures use attempts and exponential backoff, then succeed."""
    from gateway.providers import ModelRegistry

    class FlakyProvider(Provider):
        name = "flaky"

        def __init__(self):
            self.calls = 0

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            if self.calls < 3:
                raise ProviderError("temporary overload", kind=OVERLOADED, status=503)
            yield StreamChunk(text="retried", model=model)
            yield StreamChunk(finish_reason="stop", prompt_tokens=3, completion_tokens=4, model=model)

    waits: list[float] = []

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    prov = FlakyProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "flaky", "model": "m"}},
        "providers": {"flaky": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "retries": {"attempts": 3, "max_wait_s": 10.0},
        "auth": {"allow_anonymous": True},
    })
    registry = ModelRegistry(cfg, {"flaky": prov}, max_wait=10.0, sleep=record_sleep)

    chunks = [chunk async for chunk in registry.stream("m", [{"role": "user", "content": "hi"}], {})]
    assert prov.calls == 3
    assert "".join(chunk.text for chunk in chunks) == "retried"
    assert waits == [1.0, 2.0]
    assert registry.circuit_breakers["flaky"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_first_chunk():
    """A failure after the first yielded chunk is terminal and counts as one breaker failure."""
    from gateway.providers import ModelRegistry

    class MidStreamProvider(Provider):
        name = "mid"

        def __init__(self):
            self.calls = 0

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            yield StreamChunk(text="first", model=model)
            raise ProviderError("connection lost mid-stream", kind=OVERLOADED, status=503)

    waits: list[float] = []

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    prov = MidStreamProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "mid", "model": "m"}},
        "providers": {"mid": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "retries": {"attempts": 3, "max_wait_s": 10.0},
        "auth": {"allow_anonymous": True},
    })
    registry = ModelRegistry(cfg, {"mid": prov}, max_wait=10.0, sleep=record_sleep)

    generator = registry.stream("m", [{"role": "user", "content": "hi"}], {})
    first = await anext(generator)
    assert first.text == "first"

    with pytest.raises(ProviderError):
        await anext(generator)

    assert prov.calls == 1
    assert waits == []
    assert registry.circuit_breakers["mid"].consecutive_failures == 1


@pytest.mark.asyncio
async def test_stream_retries_read_error_before_first_chunk():
    """A raw HTTPX ReadError before the first chunk is retried as a provider failure."""
    from gateway.providers import ModelRegistry

    class ReadErrorProvider(Provider):
        name = "read-error"

        def __init__(self):
            self.calls = 0

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadError("read failed", request=httpx.Request("POST", "http://mock/chat"))
            yield StreamChunk(text="after-read-error", model=model)
            yield StreamChunk(finish_reason="stop", prompt_tokens=2, completion_tokens=3, model=model)

    waits: list[float] = []

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    prov = ReadErrorProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "read-error", "model": "m"}},
        "providers": {"read-error": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "retries": {"attempts": 2, "max_wait_s": 10.0},
        "auth": {"allow_anonymous": True},
    })
    registry = ModelRegistry(cfg, {"read-error": prov}, max_wait=10.0, sleep=record_sleep)

    chunks = [chunk async for chunk in registry.stream("m", [{"role": "user", "content": "hi"}], {})]
    assert prov.calls == 2
    assert "".join(chunk.text for chunk in chunks) == "after-read-error"
    assert waits == [1.0]
    assert registry.circuit_breakers["read-error"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_stream_read_error_after_first_chunk_does_not_retry_and_trips_breaker():
    """A raw HTTPX ReadError after the first chunk is terminal and opens the breaker."""
    from gateway.circuit import CircuitState
    from gateway.providers import ModelRegistry

    class MidReadErrorProvider(Provider):
        name = "mid-read-error"

        def __init__(self):
            self.calls = 0

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            yield StreamChunk(text="first", model=model)
            raise httpx.ReadError("read failed mid-stream", request=httpx.Request("POST", "http://mock/chat"))

    waits: list[float] = []

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    prov = MidReadErrorProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "mid-read-error", "model": "m"}},
        "providers": {"mid-read-error": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "retries": {"attempts": 3, "max_wait_s": 10.0},
        "circuit_breaker": {"failure_threshold": 1},
        "auth": {"allow_anonymous": True},
    })
    registry = ModelRegistry(cfg, {"mid-read-error": prov}, max_wait=10.0, sleep=record_sleep)
    breaker = registry.circuit_breakers["mid-read-error"]

    generator = registry.stream("m", [{"role": "user", "content": "hi"}], {})
    first = await anext(generator)
    assert first.text == "first"

    with pytest.raises(httpx.ReadError):
        await anext(generator)

    assert prov.calls == 1
    assert waits == []
    assert breaker.consecutive_failures == 1
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_stream_cancellation_does_not_retry_or_count_breaker_failure():
    """CancelledError before the first chunk propagates without retrying or counting a failure."""
    from gateway.providers import ModelRegistry

    class CancelledProvider(Provider):
        name = "cancelled"

        def __init__(self):
            self.calls = 0

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            raise AssertionError("complete() should not be called")

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            self.calls += 1
            raise asyncio.CancelledError()
            yield StreamChunk()

    waits: list[float] = []

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    prov = CancelledProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "cancelled", "model": "m"}},
        "providers": {"cancelled": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "retries": {"attempts": 3, "max_wait_s": 10.0},
        "auth": {"allow_anonymous": True},
    })
    registry = ModelRegistry(cfg, {"cancelled": prov}, max_wait=10.0, sleep=record_sleep)

    with pytest.raises(asyncio.CancelledError):
        await anext(registry.stream("m", [{"role": "user", "content": "hi"}], {}))

    assert prov.calls == 1
    assert waits == []
    assert registry.circuit_breakers["cancelled"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_streaming_asyncio_task_cancellation_cleans_up_upstream(tmp_path):
    """Cancelling the consuming asyncio task closes the upstream stream and records cancellation."""
    from gateway.manage.budgets import BudgetPolicy
    from gateway.manage.releases import ReleaseManager
    from gateway.providers import ModelRegistry
    from gateway.schemas import ChatRequest
    from gateway.service import Gateway, RequestContext

    upstream_closed = asyncio.Event()

    class SlowProvider(Provider):
        name = "slow"

        async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
            return Completion("text", "stop", 5, 5, model)

        async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
            try:
                yield StreamChunk(text="Chunk 1", model=model)
                await asyncio.sleep(10.0)
                yield StreamChunk(text="Chunk 2", finish_reason="stop", model=model)
            finally:
                upstream_closed.set()

    prov = SlowProvider()
    cfg = GatewayConfig.model_validate({
        "models": {"m": {"provider": "slow", "model": "m"}},
        "providers": {"slow": {"base_url": "http://mock", "api_key_env": "KEY"}},
        "prompts": {"p": ""},
        "releases": {"v1": {"cheap": "m", "strong": "m", "prompt": "p"}},
        "deployment": {"active": "v1"},
        "auth": {"allow_anonymous": True},
    })
    store = Store(tmp_path / "gw.db")
    releases = ReleaseManager(cfg, store)
    gw = Gateway(cfg, ModelRegistry(cfg, {"slow": prov}), store, RulesRouter(), releases, BudgetPolicy(cfg.budgets))

    req = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    ctx = RequestContext()
    generator, _ = await gw.stream(req, ctx)

    received: list[str] = []
    first_received = asyncio.Event()

    async def consume() -> None:
        async for event in generator:
            received.append(event)
            first_received.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_received.wait(), timeout=2.0)
    assert "Chunk 1" in received[0]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert upstream_closed.is_set()
    assert all('"error"' not in event for event in received)

    req_row = store.query("SELECT * FROM requests WHERE id = ?", [ctx.request_id])[0]
    assert req_row["status"] == "error"
    assert req_row["error_type"] == "cancelled"
    assert "cancelled" in req_row["error_message"]
