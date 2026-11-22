"""Prometheus metrics (/metrics), JSON access logs and the monitoring configs."""

from __future__ import annotations

import json
import logging
import random

import yaml
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import ROOT
from gateway.logs import JsonFormatter, log_access
from gateway.monitoring.prometheus import MAX_APPS, Metrics
from gateway.monitoring.store import CallRecord, RequestRecord, Store
from gateway.providers.fake import FakeProvider
from gateway.routing import RulesRouter
from tests.conftest import make_config

SECRET_PROMPT = "my secret prompt text 4711"


def build(monkeypatch, tmp_path, token=None):
    monkeypatch.setenv("TEST_GATEWAY_KEYS", "web:key-web")
    if token:
        monkeypatch.setenv("GATEWAY_METRICS_TOKEN", token)
    else:
        monkeypatch.delenv("GATEWAY_METRICS_TOKEN", raising=False)
    store = Store(tmp_path / "gateway.db")
    app = create_app(make_config(auth={"allow_anonymous": False}),
                     providers={"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini")}, store=store,
                     router=RulesRouter(), rng=random.Random(0))
    return app, store


def sample(text: str, name: str, **labels: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + "{") and all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name} {labels} not found")


def test_metrics_count_requests_tokens_cost_and_rejections(monkeypatch, tmp_path):
    app, store = build(monkeypatch, tmp_path)
    with TestClient(app) as client:
        for _ in range(2):
            ok = client.post("/v1/chat/completions", headers={"Authorization": "Bearer key-web"},
                             json={"messages": [{"role": "user", "content": SECRET_PROMPT}]})
            assert ok.status_code == 200
        bad = client.post("/v1/chat/completions", headers={"Authorization": "Bearer key-web"},
                          json={"model": "no-such-model", "messages": [{"role": "user", "content": "x"}]})
        assert bad.status_code == 404
        response = client.get("/metrics")
    store.close()
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = response.text
    assert sample(text, "gateway_requests_total", app="web", status="ok") == 2
    assert sample(text, "gateway_requests_total", app="web", status="rejected") == 1
    assert sample(text, "gateway_request_duration_seconds_count", status="ok") == 2
    assert sample(text, "gateway_tokens_total", app="web", kind="prompt") > 0
    assert sample(text, "gateway_provider_calls_total", status="ok") >= 2
    assert "gateway_build_info" in text and 'gateway_state_degraded{backend="memory"} 0' in text
    assert SECRET_PROMPT not in text and "key-web" not in text


def test_metrics_token_is_required_when_set(monkeypatch, tmp_path):
    app, store = build(monkeypatch, tmp_path, token="scrape-token")
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer scrape-token"}).status_code == 200
    store.close()


def test_histogram_buckets_are_cumulative_and_app_labels_are_capped():
    metrics = Metrics("test")
    for i in range(MAX_APPS + 5):
        metrics.observe(RequestRecord(id=f"r{i}", ts="2026-11-22T00:00:00", app_id=f"app{i}", status="ok",
                                      latency_ms=300.0, calls=[CallRecord(1, "cheap", "small", latency_ms=200.0,
                                                                          prompt_tokens=3, cost_usd=0.001)]))
    text = metrics.render()
    assert sample(text, "gateway_requests_total", app="other") == 5
    assert sample(text, "gateway_request_duration_seconds_bucket", le="0.25") == 0
    assert sample(text, "gateway_request_duration_seconds_bucket", le="0.5") == MAX_APPS + 5
    assert sample(text, "gateway_request_duration_seconds_bucket", le="+Inf") == MAX_APPS + 5
    assert sample(text, "gateway_cost_usd_total", app="app0", model="small") == 0.001


def test_json_access_log_has_outcome_and_no_content(caplog):
    record = RequestRecord(id="req-1", ts="2026-11-22T00:00:00", app_id="web", status="error", http_status=502,
                           error_type="upstream_error", error_message=f"provider said: {SECRET_PROMPT}",
                           model="small", cost_usd=0.002, latency_ms=12.5)
    with caplog.at_level(logging.INFO, logger="gateway.access"):
        log_access(record)
    line = JsonFormatter().format(caplog.records[-1])
    data = json.loads(line)
    assert (data["request_id"], data["app_id"], data["status"], data["http_status"]) == ("req-1", "web", "error", 502)
    assert data["logger"] == "gateway.access" and data["cost_usd"] == 0.002
    assert SECRET_PROMPT not in line  # provider messages stay in the request log


def test_monitoring_configs_are_consistent():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for name in ("prometheus", "grafana"):
        service = compose["services"][name]
        assert service["profiles"] == ["monitoring"]
        assert all(port.startswith("127.0.0.1:") for port in service["ports"])  # never published publicly
    for name in ("gateway", "gateway-replica"):
        assert compose["services"][name]["environment"]["GATEWAY_LOG_FORMAT"] == "json"
    prometheus = yaml.safe_load((ROOT / "config" / "prometheus" / "prometheus.yml").read_text(encoding="utf-8"))
    targets = prometheus["scrape_configs"][0]["static_configs"][0]["targets"]
    assert targets == ["gateway:8000", "gateway-replica:8001"]
    rules = yaml.safe_load((ROOT / "config" / "prometheus" / "alerts.yml").read_text(encoding="utf-8"))
    assert {r["alert"] for r in rules["groups"][0]["rules"]} >= {"GatewayHighErrorRate", "GatewayStateDegraded"}
    dashboard = json.loads((ROOT / "config" / "grafana" / "dashboards" / "gateway.json").read_text(encoding="utf-8"))
    exprs = " ".join(t["expr"] for p in dashboard["panels"] for t in p["targets"])
    for metric in ("gateway_requests_total", "gateway_request_duration_seconds_bucket", "gateway_cost_usd_total"):
        assert metric in exprs
    assert "location = /metrics" in (ROOT / "config" / "nginx.conf").read_text(encoding="utf-8")
