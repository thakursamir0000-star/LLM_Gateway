"""Latency-aware provider routing: interactive vs batch models, cross-provider fallback, quota cooldown."""

import asyncio
import random
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gateway.app import create_app
from gateway.config import DEFAULT_CONFIG, load_config
from gateway.monitoring import metrics
from gateway.monitoring.store import RequestRecord, Store, iso, utc_now
from gateway.providers import ModelRegistry
from gateway.providers.base import OVERLOADED, QUOTA_EXHAUSTED, Completion, ProviderError
from gateway.providers.fake import FakeProvider
from gateway.routing import RouteScore, RulesRouter
from gateway.service import Gateway
from tests.conftest import make_config

LATENCY = {
    "providers": {"openrouter": {"base_url": "https://openrouter.test/api/v1", "api_key_env": "TEST_OPENROUTER_KEY"}},
    "models": {
        "or-small": {"provider": "openrouter", "model": "openai/gpt-oss-20b", "price_per_1m": {"input": 0.03,
                                                                                                "output": 0.14}},
        "or-large": {"provider": "openrouter", "model": "openai/gpt-oss-120b", "price_per_1m": {"input": 0.037,
                                                                                                 "output": 0.17}},
    },
    "releases": {"lat": {"cheap": "small", "strong": "large", "prompt": "plain@1",
                         "batch": {"cheap": "or-small", "strong": "or-large"}}},
    "deployment": {"active": "lat"},
    "routing": {"default_mode": "strong"},
    "latency": {"default": "interactive", "apps": {"etl": "batch"}},
}
EASY = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}


def latency_config(**overrides):
    return make_config(**{**LATENCY, **overrides})


def raising(errors_by_model, answer="Paris."):
    """A fake script: raise errors_by_model[model] (exception or list of them), else answer."""
    def script(model, messages, params):
        error = errors_by_model.get(model)
        if isinstance(error, list):
            error = error.pop(0) if error else None
        if error is not None:
            raise error
        return Completion(answer, "stop", 10, 5, model)

    return script


async def no_sleep(_):
    return None


def build(tmp_path, config=None, groq=None, openrouter=None, router=None):
    config = config or latency_config()
    providers = {"groq": groq or FakeProvider("groq"), "gemini": FakeProvider("gemini"),
                 "openrouter": openrouter or FakeProvider("openrouter")}
    store = Store(tmp_path / "gw.db")
    app = create_app(config, providers=providers, store=store, router=router or RulesRouter(), rng=random.Random(0),
                     sleep=no_sleep)
    return SimpleNamespace(client=TestClient(app), store=store, app=app, **providers)


def post(gw, headers=None, **body):
    return gw.client.post("/v1/chat/completions", json={**EASY, **body}, headers=headers or {})


def calls(gw):
    return [(c["tier"], c["model"], c["status"]) for c in gw.store.query("SELECT * FROM calls ORDER BY ts, seq")]


# ── choosing the provider ───────────────────────────────────────────

def test_interactive_goes_to_the_fast_models_and_batch_to_the_batch_models(tmp_path):
    gw = build(tmp_path)
    interactive = post(gw).json()
    assert interactive["gateway"]["latency_class"] == "interactive" and interactive["gateway"]["model"] == "large"
    assert interactive["gateway"]["tier"] == "strong" and len(gw.groq.calls) == 1
    batch = post(gw, {"X-Gateway-Latency": "batch"}).json()
    assert batch["gateway"]["latency_class"] == "batch" and batch["gateway"]["model"] == "or-large"
    assert [c["model"] for c in gw.openrouter.calls] == ["openai/gpt-oss-120b"]


def test_app_default_and_header_override(tmp_path):
    gw = build(tmp_path)
    assert post(gw, {"X-App-Id": "etl"}).json()["gateway"]["model"] == "or-large"
    overridden = post(gw, {"X-App-Id": "etl", "X-Gateway-Latency": "Interactive"}).json()
    assert overridden["gateway"]["model"] == "large"


def test_every_mode_uses_the_class_models(tmp_path):
    gw = build(tmp_path)
    assert post(gw, {"X-Gateway-Latency": "batch"}, model="cheap").json()["gateway"]["model"] == "or-small"
    routed = post(gw, {"X-Gateway-Latency": "batch"}, model="router").json()["gateway"]
    assert routed["model"] == ("or-small" if routed["tier"] == "cheap" else "or-large")
    pinned = post(gw, {"X-Gateway-Latency": "batch"}, model="small").json()["gateway"]
    assert pinned["model"] == "small"  # a pinned model ignores the class


def test_invalid_latency_header_is_rejected(tmp_path):
    response = post(build(tmp_path), {"X-Gateway-Latency": "urgent"})
    assert response.status_code == 400 and "interactive" in response.json()["error"]["message"]


def test_release_without_batch_models_serves_batch_with_the_same_models(tmp_path):
    gw = build(tmp_path, config=latency_config(deployment={"active": "v1"}))
    body = post(gw, {"X-Gateway-Latency": "batch"}).json()["gateway"]
    assert body["model"] == "large" and body["latency_class"] == "batch" and gw.openrouter.calls == []


def test_latency_class_is_logged_and_summarised(tmp_path):
    gw = build(tmp_path)
    post(gw)
    post(gw, {"X-Gateway-Latency": "batch"})
    post(gw, {"X-Gateway-Latency": "batch"})
    logged = [r["latency_class"] for r in gw.store.query("SELECT latency_class FROM requests ORDER BY ts")]
    assert logged == ["interactive", "batch", "batch"]
    stats = {r["latency_class"]: r for r in metrics.latency_class_stats(gw.store)}
    assert stats["batch"]["requests"] == 2 and stats["interactive"]["requests"] == 1
    assert stats["batch"]["p50_ms"] is not None
    assert "latency_class" in metrics.recent(gw.store)[0]


def test_health_reports_the_default_class(tmp_path):
    health = build(tmp_path).client.get("/health").json()
    assert health["latency_default"] == "interactive" and health["classifier_threshold"] == 0.5


class OpenRouterClassifier:
    name = "classifier"
    description = "test OpenRouter classifier"
    model_aliases = ("or-small", "or-large")

    def score(self, messages):
        return RouteScore(0.96, self.name, "P(cheap ok)=0.96")


def test_openrouter_classifier_is_batch_only_and_auto_stays_strong(tmp_path):
    config = latency_config(routing={"default_mode": "strong", "router": "classifier",
                                     "classifier_threshold": 0.955})
    gw = build(tmp_path, config=config, router=OpenRouterClassifier())

    automatic = post(gw, {"X-Gateway-Latency": "batch"}).json()["gateway"]
    assert automatic["model"] == "or-large" and automatic["mode"] == "strong"

    batch = post(gw, {"X-Gateway-Latency": "batch"}, model="router").json()["gateway"]
    assert batch["model"] == "or-small" and batch["tier"] == "cheap" and batch["threshold"] == 0.955

    interactive = post(gw, model="router").json()["gateway"]
    assert interactive["model"] == "large" and interactive["tier"] == "strong"
    assert interactive["route_score"] is None and "classifier trained for" in interactive["reason"]


# ── fallback across providers ───────────────────────────────────────

def test_fallback_order():
    release = latency_config().releases["lat"]
    assert Gateway.fallback_chain("strong", "large", release, "interactive") == [
        ("strong", "large"), ("strong", "or-large"), ("cheap", "small"), ("cheap", "or-small")]
    assert Gateway.fallback_chain("cheap", "or-small", release, "batch") == [
        ("cheap", "or-small"), ("cheap", "small"), ("strong", "or-large"), ("strong", "large")]
    plain = make_config().releases["v1"]
    assert Gateway.fallback_chain("cheap", "small", plain, "batch") == [("cheap", "small"), ("strong", "large")]


def test_quota_on_one_provider_falls_back_to_the_other(tmp_path):
    quota = ProviderError("tokens per day (TPD)", kind=QUOTA_EXHAUSTED, status=429)
    gw = build(tmp_path, groq=FakeProvider("groq", raising({"openai/gpt-oss-120b": quota})))
    body = post(gw).json()
    assert body["gateway"]["model"] == "or-large" and body["gateway"]["fallback"] is True
    assert body["gateway"]["tier"] == "strong" and "fallback to or-large" in body["gateway"]["reason"]
    assert calls(gw) == [("strong", "large", "error"), ("strong", "or-large", "ok")]


def test_a_model_out_of_quota_is_skipped_for_the_cooldown(tmp_path):
    quota = ProviderError("tokens per day (TPD)", kind=QUOTA_EXHAUSTED, status=429)
    groq = FakeProvider("groq", raising({"openai/gpt-oss-120b": quota}))
    gw = build(tmp_path, groq=groq)
    post(gw)
    second = post(gw).json()
    assert second["gateway"]["model"] == "or-large"
    assert len(groq.calls) == 1  # not called again while cooling down
    skipped = gw.store.query("SELECT error_message FROM calls WHERE status = 'error' ORDER BY ts DESC LIMIT 1")
    assert "cooling down" in skipped[0]["error_message"]


def test_both_providers_down_falls_back_to_the_other_tier(tmp_path):
    busy = ProviderError("busy", kind=OVERLOADED, status=503)
    gw = build(tmp_path, groq=FakeProvider("groq", raising({"openai/gpt-oss-120b": busy})),
               openrouter=FakeProvider("openrouter", raising({"openai/gpt-oss-120b": busy})))
    body = post(gw).json()
    assert body["gateway"]["model"] == "small" and body["gateway"]["tier"] == "cheap"
    assert calls(gw) == [("strong", "large", "error"), ("strong", "or-large", "error"), ("cheap", "small", "ok")]


def test_everything_down_returns_the_last_error(tmp_path):
    busy = ProviderError("busy", kind=OVERLOADED, status=503)
    everything = {"openai/gpt-oss-120b": busy, "openai/gpt-oss-20b": busy}
    gw = build(tmp_path, groq=FakeProvider("groq", raising(everything)),
               openrouter=FakeProvider("openrouter", raising(everything)))
    response = post(gw)
    assert response.status_code == 503 and len(calls(gw)) == 4


def test_fourth_fallback_candidate_is_used(tmp_path):
    busy = ProviderError("busy", kind=OVERLOADED, status=503)
    groq = FakeProvider("groq", raising({"openai/gpt-oss-120b": busy, "openai/gpt-oss-20b": busy}))
    openrouter = FakeProvider("openrouter", raising({"openai/gpt-oss-120b": busy}))
    gw = build(tmp_path, groq=groq, openrouter=openrouter)
    body = post(gw).json()["gateway"]
    assert body["model"] == "or-small" and body["tier"] == "cheap" and body["fallback"] is True
    assert calls(gw) == [("strong", "large", "error"), ("strong", "or-large", "error"),
                         ("cheap", "small", "error"), ("cheap", "or-small", "ok")]


def test_cascade_escalates_to_the_strong_model_of_the_same_class(tmp_path):
    def script(model, messages, params):
        text = "I'm not sure." if model.endswith("20b") else "Paris."
        return Completion(text, "stop", 10, 5, model)

    gw = build(tmp_path, openrouter=FakeProvider("openrouter", script))
    body = post(gw, {"X-Gateway-Latency": "batch"}, model="cascade").json()["gateway"]
    assert body["escalated"] is True and body["model"] == "or-large" and gw.groq.calls == []


# ── registry cooldown ───────────────────────────────────────────────

def test_registry_cooldown_follows_the_clock_and_the_provider_hint():
    clock = {"t": 1000.0}
    quota = ProviderError("daily", kind=QUOTA_EXHAUSTED, retry_after=900)
    groq = FakeProvider("groq", raising({"openai/gpt-oss-120b": [quota]}))
    registry = ModelRegistry(latency_config(), {"groq": groq, "gemini": FakeProvider(), "openrouter": FakeProvider()},
                             clock=lambda: clock["t"])
    user = [{"role": "user", "content": "hi"}]
    with pytest.raises(ProviderError):
        asyncio.run(registry.complete("large", user, {}))
    assert registry.cooldown_left("large") == pytest.approx(900)  # the provider's hint beats the 300 s default
    with pytest.raises(ProviderError, match="cooling down"):
        asyncio.run(registry.complete("large", user, {}))
    assert len(groq.calls) == 1 and registry.cooldown_left("small") == 0
    clock["t"] += 901
    assert asyncio.run(registry.complete("large", user, {})).text == "Paris."
    assert len(groq.calls) == 2


# ── config and storage ──────────────────────────────────────────────

def test_release_batch_models_must_exist():
    with pytest.raises(ValidationError, match="unknown batch.strong model"):
        latency_config(releases={"lat": {"cheap": "small", "strong": "large", "prompt": "plain@1",
                                         "batch": {"cheap": "or-small", "strong": "nope"}}})


def test_latency_values_are_validated():
    with pytest.raises(ValidationError):
        latency_config(latency={"default": "urgent"})
    with pytest.raises(ValidationError):
        latency_config(latency={"apps": {"etl": "slow"}})


def test_old_databases_get_the_new_column(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, ts TEXT NOT NULL, day TEXT NOT NULL, app_id TEXT "
                 "NOT NULL, requested_model TEXT, release TEXT, prompt_version TEXT, canary INTEGER NOT NULL "
                 "DEFAULT 0, mode TEXT, router TEXT, route_score REAL, route_reason TEXT, threshold REAL, tier TEXT, "
                 "model TEXT, provider_model TEXT, escalated INTEGER NOT NULL DEFAULT 0, fallback INTEGER NOT NULL "
                 "DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, "
                 "cost_usd REAL NOT NULL DEFAULT 0, latency_ms REAL, router_ms REAL, status TEXT NOT NULL, "
                 "http_status INTEGER, error_type TEXT, error_message TEXT, stream INTEGER NOT NULL DEFAULT 0)")
    conn.execute("INSERT INTO requests (id, ts, day, app_id, status) VALUES ('old', '2026-11-01T00:00:00+00:00', "
                 "'2026-11-01', 'a', 'ok')")
    conn.commit()
    conn.close()
    store = Store(path)
    store.log_request(RequestRecord(id="new", ts=iso(utc_now()), app_id="a", status="ok", latency_class="batch"))
    rows = {r["id"]: r["latency_class"] for r in store.query("SELECT id, latency_class FROM requests")}
    assert rows == {"old": None, "new": "batch"}
    assert {r["latency_class"] for r in metrics.latency_class_stats(store)} == {"not recorded", "batch"}


def test_shipped_config_routes_by_latency_class():
    config = load_config(DEFAULT_CONFIG)
    release = config.releases[config.deployment.active]
    assert config.deployment.active == "v4-latency" and config.routing.default_mode == "strong"
    assert config.models[release.model_for("strong", "interactive")].provider == "groq"
    assert config.models[release.model_for("strong", "batch")].provider == "openrouter"
    assert config.latency.default == "interactive" and "v1" in config.releases  # v1 kept for rollback
    assert config.routing.default_mode == "strong" and config.routing.classifier_threshold == 0.955
