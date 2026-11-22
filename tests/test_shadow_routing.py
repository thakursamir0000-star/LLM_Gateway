"""Router shadow mode and controlled classifier activation (routing.shadow, routing.classifier_share)."""

from __future__ import annotations

import random

from fastapi.testclient import TestClient

import scripts.router_report as router_report
from gateway.app import create_app
from gateway.monitoring.store import Store, iso, utc_now
from gateway.providers.fake import FakeProvider
from gateway.routing.base import RouteScore
from gateway.service import in_share
from tests.conftest import make_config


class StubClassifier:
    """A trained classifier for the test config's v1 pair (small, large); fixed score."""

    name = "classifier"
    description = "stub classifier"

    def __init__(self, score: float = 0.9, aliases=("small", "large")):
        self.value, self.model_aliases, self.calls = score, aliases, 0

    def score(self, messages):
        self.calls += 1
        return RouteScore(self.value, self.name, f"P(cheap ok)={self.value}")


def build(tmp_path, router, **routing):
    config = make_config(routing={"default_mode": "strong", "router": "classifier", "classifier_threshold": 0.5,
                                  **routing})
    store = Store(tmp_path / "gateway.db")
    app = create_app(config, providers={"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini")},
                     store=store, router=router, rng=random.Random(0))
    return TestClient(app), store


def ask(client, model=None, stream=False):
    body = {"messages": [{"role": "user", "content": "What is 2 + 2?"}], "stream": stream}
    if model:
        body["model"] = model
    response = client.post("/v1/chat/completions", json=body, headers={"X-App-Id": "web"})
    assert response.status_code == 200
    return response


def test_in_share_is_stable_and_proportional():
    ids = [f"chatcmpl-{i}" for i in range(4000)]
    assert all(in_share(i, 1.0) for i in ids) and not any(in_share(i, 0.0) for i in ids)
    picked = sum(in_share(i, 0.1) for i in ids)
    assert 300 < picked < 500  # about 10%
    assert [in_share(i, 0.1) for i in ids[:50]] == [in_share(i, 0.1) for i in ids[:50]]
    assert all(in_share(i, 0.2) for i in ids if in_share(i, 0.1))  # raising the share keeps the earlier group


def test_shadow_mode_logs_the_classifier_decision_without_acting(tmp_path):
    router = StubClassifier(score=0.9)  # would choose cheap
    client, store = build(tmp_path, router, shadow=True)
    with client:
        for stream in (False, True):
            ask(client, stream=stream)  # default mode strong
        metrics = client.get("/metrics").text
    rows = store.query("SELECT tier, model, shadow_tier, shadow_score, router FROM requests")
    store.close()
    assert [(r["tier"], r["model"]) for r in rows] == [("strong", "large")] * 2  # still answered by strong
    assert all(r["shadow_tier"] == "cheap" and r["shadow_score"] == 0.9 for r in rows)
    assert 'gateway_shadow_decisions_total{tier="strong",shadow_tier="cheap"} 2' in metrics


def test_shadow_mode_off_or_mismatched_pair_scores_nothing(tmp_path):
    router = StubClassifier()
    client, store = build(tmp_path, router)  # shadow off
    with client:
        ask(client)
    mismatched = StubClassifier(aliases=("or-small", "or-large"))
    client2, store2 = build(tmp_path / "b", mismatched, shadow=True)
    with client2:
        ask(client2)
    rows = store.query("SELECT shadow_tier FROM requests") + store2.query("SELECT shadow_tier FROM requests")
    store.close()
    store2.close()
    assert router.calls == 0 and mismatched.calls == 0 and all(r["shadow_tier"] is None for r in rows)


def test_classifier_share_splits_router_traffic_into_classifier_and_control(tmp_path):
    router = StubClassifier(score=0.9)
    client, store = build(tmp_path, router, classifier_share=0.3, shadow=True)
    with client:
        for _ in range(60):
            ask(client, model="router")
    rows = store.query("SELECT id, router, tier, shadow_tier FROM requests")
    store.close()
    classifier = [r for r in rows if r["router"] == "classifier"]
    control = [r for r in rows if r["router"] == "control"]
    assert len(classifier) + len(control) == 60 and 5 < len(classifier) < 35
    assert all(r["tier"] == "cheap" and r["shadow_tier"] is None for r in classifier)
    assert all(r["tier"] == "strong" and r["shadow_tier"] == "cheap" for r in control)  # control is shadow-scored
    assert all(in_share(r["id"], 0.3) == (r["router"] == "classifier") for r in rows)


def test_router_report_reads_shadow_and_activation(tmp_path, monkeypatch):
    router = StubClassifier(score=0.9)
    client, store = build(tmp_path, router, classifier_share=0.5, shadow=True)
    with client:
        for _ in range(20):
            ask(client, model="router")
        for _ in range(5):
            ask(client)
    config = client.app.state.gateway.config
    first = store.query("SELECT id FROM requests WHERE router = 'control' LIMIT 1")[0]["id"]
    store.add_feedback(first, "web", 1, None, iso(utc_now()))
    result = router_report.report(store, config, iso(utc_now().replace(year=2000)))
    store.close()
    shadow, activation = result["shadow"], result["activation"]
    assert shadow["scored"] == 5 + activation["control"]["requests"]
    assert shadow["agreement"] == 0.0 and shadow["would_send_cheap"] == 1.0  # it would always pick cheap
    assert shadow["estimated_saving_usd"] > 0 and 0 < shadow["estimated_saving_share"] < 1
    assert activation["classifier"]["requests"] + activation["control"]["requests"] == 20
    assert activation["classifier"]["cheap_share"] == 1.0 and activation["control"]["cheap_share"] == 0.0
    assert activation["control"]["feedback"] == {"ratings": 1, "mean": 1.0}
