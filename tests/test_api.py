"""End-to-end tests of the HTTP API with fake providers (no real LLM calls)."""

import json
import random
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app, parse_api_keys
from gateway.monitoring.store import Store
from gateway.providers.base import OVERLOADED, QUOTA_EXHAUSTED, Completion, ProviderError, message_text
from gateway.providers.fake import FakeProvider
from gateway.routing import RulesRouter
from tests.conftest import make_config

EASY = "Translate 'good morning' into French."
HARD = ("Step by step, prove that the integral of x^2 from 0 to 1 is 1/3, then write ```python\ndef f(): ...``` "
        "code to check it. Compare both approaches.")


def chat(text, **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


async def no_sleep(seconds):
    return None


def build(tmp_path, config=None, groq=None, gemini=None):
    config = config or make_config()
    providers = {"groq": groq or FakeProvider("groq"), "gemini": gemini or FakeProvider("gemini")}
    store = Store(tmp_path / "gw.db")
    alerts = []
    app = create_app(config, providers=providers, store=store, router=RulesRouter(), rng=random.Random(0),
                     notify=alerts.append, sleep=no_sleep)
    return SimpleNamespace(client=TestClient(app), groq=providers["groq"], gemini=providers["gemini"], store=store,
                           alerts=alerts, config=config)


@pytest.fixture
def gw(tmp_path):
    return build(tmp_path)


def rows(store, sql="SELECT * FROM requests ORDER BY ts"):
    return store.query(sql)


def scripted(answers):
    """A fake script answering by provider model name; values may be exceptions."""

    def script(model, messages, params):
        answer = answers[model]
        if isinstance(answer, Exception):
            raise answer
        return Completion(answer, "stop", 100, 50, model)

    return script


# ── OpenAI-compatible chat completions ──────────────────────────────

def test_easy_request_is_routed_to_the_cheap_model(gw):
    response = gw.client.post("/v1/chat/completions", json=chat(EASY))
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    info = body["gateway"]
    assert info["tier"] == "cheap" and info["mode"] == "router" and info["router"] == "rules"
    assert info["release"] == "v1" and info["canary"] is False
    assert response.headers["x-gateway-model"] == "small"
    assert response.headers["x-gateway-request-id"] == body["id"]
    assert gw.groq.calls[0]["params"] == {"reasoning_effort": "low"}  # the model's default params


def test_hard_request_is_routed_to_the_strong_model(gw):
    body = gw.client.post("/v1/chat/completions", json=chat(HARD)).json()
    assert body["gateway"]["tier"] == "strong" and body["model"] == "openai/gpt-oss-120b"
    assert "maths" in body["gateway"]["reason"]


def test_threshold_header_trades_cost_for_quality(gw):
    always_strong = gw.client.post("/v1/chat/completions", json=chat(EASY), headers={"X-Gateway-Threshold": "1"})
    always_cheap = gw.client.post("/v1/chat/completions", json=chat(HARD), headers={"X-Gateway-Threshold": "0"})
    assert always_strong.json()["gateway"]["tier"] == "strong"
    assert always_cheap.json()["gateway"]["tier"] == "cheap"
    bad = gw.client.post("/v1/chat/completions", json=chat(EASY), headers={"X-Gateway-Threshold": "2"})
    assert bad.status_code == 400


@pytest.mark.parametrize("model, tier, provider_model", [
    ("cheap", "cheap", "openai/gpt-oss-20b"),
    ("strong", "strong", "openai/gpt-oss-120b"),
    ("large", "pinned", "openai/gpt-oss-120b"),  # a configured alias
    ("openai/gpt-oss-20b", "pinned", "openai/gpt-oss-20b"),  # a provider model name
    ("gemma", "pinned", "gemma-4-31b-it"),
])
def test_model_field_selects_the_mode(gw, model, tier, provider_model):
    body = gw.client.post("/v1/chat/completions", json=chat(HARD, model=model)).json()
    assert body["gateway"]["tier"] == tier and body["model"] == provider_model
    assert body["gateway"]["router"] is None


def test_unknown_model_is_an_openai_style_404(gw):
    response = gw.client.post("/v1/chat/completions", json=chat(EASY, model="gpt-9"))
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found" and error["param"] == "model" and "auto" in error["message"]


@pytest.mark.parametrize("payload, message", [
    ({"messages": []}, "messages"),
    (chat("hi", n=2), "n=1"),
    (chat("hi", tools=[{"type": "function", "function": {"name": "f"}}]), "tools"),
    (chat("hi", temperature=5), "temperature"),
    ({"messages": [{"role": "tool", "content": "x"}]}, "tool messages"),
])
def test_invalid_requests_get_400(gw, payload, message):
    response = gw.client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400 and message in response.json()["error"]["message"]


def test_extra_openai_fields_are_accepted(gw):
    response = gw.client.post("/v1/chat/completions", json=chat(EASY, logit_bias={"1": 1}, max_tokens=20, user="u1"))
    assert response.status_code == 200
    assert gw.groq.calls[0]["params"]["max_tokens"] == 20


# ── logging ─────────────────────────────────────────────────────────

def test_every_request_is_logged(gw):
    gw.client.post("/v1/chat/completions", json=chat(EASY))
    gw.client.post("/v1/chat/completions", json=chat(HARD), headers={"X-App-Id": "app-2"})
    gw.client.post("/v1/chat/completions", json=chat(EASY, model="nope"))
    logged = rows(gw.store)
    assert [r["status"] for r in logged] == ["ok", "ok", "rejected"]
    first = logged[0]
    assert first["app_id"] == "default" and first["model"] == "small" and first["tier"] == "cheap"
    assert first["mode"] == "router" and first["router"] == "rules" and first["route_score"] > 0.5
    assert first["prompt_tokens"] > 0 and first["cost_usd"] > 0 and first["latency_ms"] >= 0
    assert first["router_ms"] is not None and first["threshold"] == 0.5
    assert logged[1]["app_id"] == "app-2" and logged[1]["tier"] == "strong"
    assert logged[2]["http_status"] == 404 and logged[2]["error_type"] == "model_not_found"
    assert len(rows(gw.store, "SELECT * FROM calls")) == 2


def test_prompts_and_answers_are_not_stored(gw):
    gw.client.post("/v1/chat/completions", json=chat("my secret phrase is swordfish"))
    dump = json.dumps(rows(gw.store) + rows(gw.store, "SELECT * FROM calls"))
    assert "swordfish" not in dump


# ── cascade and fallback ────────────────────────────────────────────

def test_cascade_escalates_an_unsure_cheap_answer(tmp_path):
    groq = FakeProvider("groq", scripted({"openai/gpt-oss-20b": "I'm not sure, maybe 12?",
                                          "openai/gpt-oss-120b": "It is 42."}))
    gw = build(tmp_path, groq=groq)
    body = gw.client.post("/v1/chat/completions", json=chat("What is 6*7?", model="cascade")).json()
    assert body["choices"][0]["message"]["content"] == "It is 42."
    info = body["gateway"]
    assert info["escalated"] is True and info["tier"] == "strong" and "hedged" in info["reason"]
    assert body["usage"]["prompt_tokens"] == 200  # both calls are counted
    calls = rows(gw.store, "SELECT * FROM calls ORDER BY seq")
    assert [c["tier"] for c in calls] == ["cheap", "strong"]
    assert calls[0]["confidence"] == pytest.approx(0.3)
    expected = gw.config.models["small"].cost(100, 50) + gw.config.models["large"].cost(100, 50)
    assert rows(gw.store)[0]["cost_usd"] == pytest.approx(expected)


def test_cascade_keeps_a_confident_cheap_answer(tmp_path):
    groq = FakeProvider("groq", scripted({"openai/gpt-oss-20b": "6 times 7 is 42.", "openai/gpt-oss-120b": "42"}))
    gw = build(tmp_path, groq=groq)
    body = gw.client.post("/v1/chat/completions", json=chat("What is 6*7?", model="cascade")).json()
    assert body["gateway"]["escalated"] is False and body["gateway"]["tier"] == "cheap"
    assert len(groq.calls) == 1


def test_fallback_to_the_other_tier_when_a_model_fails(tmp_path):
    groq = FakeProvider("groq", scripted({"openai/gpt-oss-20b": ProviderError("busy", kind=OVERLOADED),
                                          "openai/gpt-oss-120b": "fine"}))
    gw = build(tmp_path, groq=groq)
    body = gw.client.post("/v1/chat/completions", json=chat(EASY)).json()
    assert body["gateway"]["fallback"] is True and body["gateway"]["tier"] == "strong"
    logged = rows(gw.store)[0]
    assert logged["status"] == "ok" and logged["fallback"] == 1
    calls = rows(gw.store, "SELECT * FROM calls ORDER BY seq")
    assert [(c["tier"], c["status"]) for c in calls] == [("cheap", "error"), ("strong", "ok")]
    assert calls[0]["error_type"] == "overloaded"
    assert len(groq.calls) == 4  # three cheap attempts (retried inside the call), then strong


def test_provider_errors_are_returned_and_logged(tmp_path):
    config = make_config(routing={"fallback_on_error": False})
    quota = ProviderError("tokens per day (TPD) reached", kind=QUOTA_EXHAUSTED, status=429, retry_after=120)
    groq = FakeProvider("groq", scripted({"openai/gpt-oss-20b": quota}))
    gw = build(tmp_path, config=config, groq=groq)
    response = gw.client.post("/v1/chat/completions", json=chat(EASY))
    assert response.status_code == 429 and response.headers["retry-after"] == "120"
    assert response.json()["error"]["code"] == "upstream_quota_exhausted"
    logged = rows(gw.store)[0]
    assert logged["status"] == "error" and logged["error_type"] == "upstream_quota_exhausted"


def test_pinned_models_never_fall_back(tmp_path):
    groq = FakeProvider("groq", scripted({"openai/gpt-oss-120b": ProviderError("down", kind=OVERLOADED)}))
    gw = build(tmp_path, groq=groq)
    response = gw.client.post("/v1/chat/completions", json=chat(EASY, model="large"))
    assert response.status_code == 503 and rows(gw.store)[0]["status"] == "error"


# ── streaming ───────────────────────────────────────────────────────

def test_streaming_returns_server_sent_events(gw):
    response = gw.client.post("/v1/chat/completions",
                              json=chat(EASY, stream=True, stream_options={"include_usage": True}))
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    events = [line[len("data: "):] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text.startswith("[openai/gpt-oss-20b] fake answer")
    assert chunks[-1]["usage"]["total_tokens"] > 0
    assert rows(gw.store)[0]["stream"] == 1


def test_openai_sdk_works_by_changing_the_base_url(gw):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=gw.client)
    completion = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": EASY}])
    assert completion.choices[0].message.content.startswith("[openai/gpt-oss-20b]")
    assert completion.usage.total_tokens > 0
    stream = client.chat.completions.create(model="strong", messages=[{"role": "user", "content": EASY}],
                                            stream=True)
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text.startswith("[openai/gpt-oss-120b]")
    assert [m.id for m in client.models.list()][:3] == ["auto", "router", "cascade"]


# ── releases: prompt versions and canary ────────────────────────────

def test_release_prompt_version_is_prepended(tmp_path):
    gw = build(tmp_path, config=make_config(deployment={"active": "v2"}))
    gw.client.post("/v1/chat/completions", json=chat(EASY, model="cheap"))
    sent = gw.gemini.calls[0]["messages"]  # v2's cheap model is Gemma on the Gemini API
    assert sent[0] == {"role": "system", "content": "Be concise."}
    assert message_text(sent[1]["content"]) == EASY
    assert rows(gw.store)[0]["prompt_version"] == "concise@1"


def test_canary_gets_its_share_of_traffic(tmp_path):
    config = make_config(deployment={"active": "v1", "canary": {"release": "v2", "share": 0.5}})
    gw = build(tmp_path, config=config)
    for _ in range(40):
        gw.client.post("/v1/chat/completions", json=chat(EASY, model="cheap"))
    logged = rows(gw.store)
    canary = [r for r in logged if r["canary"]]
    assert 8 < len(canary) < 32
    assert all(r["release"] == "v2" and r["model"] == "gemma" for r in canary)
    assert all(r["release"] == "v1" for r in logged if not r["canary"])


# ── budgets ─────────────────────────────────────────────────────────

def test_budget_soft_limit_then_hard_limit(tmp_path):
    groq = FakeProvider("groq", scripted({"openai/gpt-oss-20b": "ok", "openai/gpt-oss-120b": "ok"}))
    # Limit $0.00012, soft limit 80% = $0.000096. A strong call costs $0.0000525, a cheap one $0.000035.
    config = make_config(budgets={"apps": {"tiny": {"daily_usd": 0.00012}}})
    gw = build(tmp_path, config=config, groq=groq)
    headers = {"X-App-Id": "tiny"}
    tiers = [gw.client.post("/v1/chat/completions", json=chat(HARD), headers=headers).json()["gateway"]
             for _ in range(3)]
    assert [t["tier"] for t in tiers] == ["strong", "strong", "cheap"]  # $0.000105 spent: over the soft limit
    assert "daily budget" in tiers[2]["reason"]
    last = gw.client.post("/v1/chat/completions", json=chat(HARD), headers=headers)  # $0.00014 spent
    assert last.status_code == 429 and last.json()["error"]["code"] == "budget_exceeded"
    assert int(last.headers["retry-after"]) > 0
    assert rows(gw.store)[-1]["status"] == "rejected"
    other_app = gw.client.post("/v1/chat/completions", json=chat(HARD), headers={"X-App-Id": "other"})
    assert other_app.status_code == 200


# ── feedback ────────────────────────────────────────────────────────

def test_feedback_endpoint(gw):
    request_id = gw.client.post("/v1/chat/completions", json=chat(EASY)).json()["id"]
    up = gw.client.post("/v1/feedback", json={"request_id": request_id, "score": "up"})
    assert up.status_code == 200 and up.json()["score"] == 1
    down = gw.client.post("/v1/feedback", json={"request_id": request_id, "score": -1, "comment": "too long"})
    assert down.json()["score"] == -1
    stored = rows(gw.store, "SELECT score, comment FROM feedback ORDER BY id")
    assert stored == [{"score": 1, "comment": None}, {"score": -1, "comment": "too long"}]
    assert gw.client.post("/v1/feedback", json={"request_id": "chatcmpl-nope", "score": 1}).status_code == 404
    assert gw.client.post("/v1/feedback", json={"request_id": request_id, "score": 5}).status_code == 400


# ── auth ────────────────────────────────────────────────────────────

def test_parse_api_keys():
    assert parse_api_keys("a:k1, b:k2,broken,:x,c:") == {"k1": "a", "k2": "b"}


def test_api_keys_identify_the_app(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_GATEWAY_KEYS", "web:key-web,batch:key-batch")
    gw = build(tmp_path)
    assert gw.client.post("/v1/chat/completions", json=chat(EASY)).status_code == 401
    wrong = gw.client.post("/v1/chat/completions", json=chat(EASY), headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401 and wrong.json()["error"]["code"] == "invalid_api_key"
    ok = gw.client.post("/v1/chat/completions", json=chat(EASY),
                        headers={"Authorization": "Bearer key-batch", "X-App-Id": "spoofed"})
    assert ok.status_code == 200 and ok.json()["gateway"]["app_id"] == "batch"
    request_id = ok.json()["id"]
    other_app = gw.client.post("/v1/feedback", json={"request_id": request_id, "score": 1},
                               headers={"Authorization": "Bearer key-web"})
    assert other_app.status_code == 404  # an app can't rate another app's requests


def test_bad_app_id_header(gw):
    response = gw.client.post("/v1/chat/completions", json=chat(EASY), headers={"X-App-Id": "bad id!"})
    assert response.status_code == 400


# ── admin, health, models ───────────────────────────────────────────

def test_admin_is_off_without_a_key(gw):
    response = gw.client.get("/admin/status")
    assert response.status_code == 403 and response.json()["error"]["code"] == "admin_disabled"


def test_admin_release_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_GATEWAY_ADMIN_KEY", "admin-secret")
    gw = build(tmp_path)
    headers = {"X-Admin-Key": "admin-secret"}
    assert gw.client.get("/admin/status", headers={"X-Admin-Key": "wrong"}).status_code == 401
    canary = gw.client.post("/admin/canary", json={"release": "v2", "share": 0.2}, headers=headers).json()
    assert canary["canary"] == "v2" and canary["canary_share"] == 0.2
    assert gw.client.delete("/admin/canary", headers=headers).json()["canary"] is None
    assert gw.client.post("/admin/activate", json={"release": "v2"}, headers=headers).json()["active"] == "v2"
    body = gw.client.post("/v1/chat/completions", json=chat(EASY, model="cheap")).json()
    assert body["gateway"]["release"] == "v2"
    assert gw.client.post("/admin/rollback", headers=headers).json()["active"] == "v1"
    refused = gw.client.post("/admin/activate", json={"release": "v9"}, headers=headers)
    assert refused.status_code == 400 and "unknown release" in refused.json()["error"]["message"]
    status = gw.client.get("/admin/status", headers=headers).json()
    assert status["deployment"]["active"] == "v1" and status["last_24h"]["requests"] == 1
    assert status["spend_today"][0]["daily_limit_usd"] == 1.0
    actions = [d["action"] for d in gw.client.get("/admin/deployments", headers=headers).json()["deployments"]]
    assert actions == ["rollback", "activate", "canary-off", "canary"]


def test_alerts_fire_from_live_traffic(tmp_path):
    groq = FakeProvider("groq", scripted({"openai/gpt-oss-20b": ProviderError("down", kind=OVERLOADED),
                                          "openai/gpt-oss-120b": ProviderError("down", kind=OVERLOADED)}))
    gw = build(tmp_path, groq=groq)
    for _ in range(4):
        gw.client.post("/v1/chat/completions", json=chat(EASY))
    assert [a.kind for a in gw.alerts] == ["error_rate"]


def test_health_and_models(gw):
    health = gw.client.get("/health").json()
    assert health["status"] == "ok" and health["release"] == "v1" and health["router"].startswith("rule-based")
    ids = [m["id"] for m in gw.client.get("/v1/models").json()["data"]]
    assert ids == ["auto", "router", "cascade", "cheap", "strong", "small", "large", "gemma"]
    assert gw.client.get("/nope").json()["error"]["message"] == "Not Found"
