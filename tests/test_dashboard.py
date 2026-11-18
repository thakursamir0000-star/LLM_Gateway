"""The Streamlit dashboard renders from a request log (run headless with Streamlit's AppTest)."""

from datetime import timedelta

import pytest
import yaml

from gateway.config import ROOT
from gateway.monitoring.store import CallRecord, RequestRecord, Store, iso, utc_now

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
DASHBOARD = str(ROOT / "dashboard" / "app.py")


def test_dashboard_deployment_is_loopback_only_and_keeps_local_healthcheck():
    compose_path = ROOT / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    dashboard = compose["services"]["dashboard"]

    assert dashboard["ports"] == ["127.0.0.1:8501:8501"]
    assert "--server.address=0.0.0.0" in dashboard["command"]
    assert "http://127.0.0.1:8501/_stcore/health" in " ".join(dashboard["healthcheck"]["test"])

    # C2 changes dashboard exposure only. The gateway's existing host mapping remains unchanged.
    assert compose["services"]["gateway"]["ports"] == ["8000:8000"]

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "-p 127.0.0.1:8501:8501" in ci
    assert "-p 8501:8501" not in ci


def populate(path):
    store = Store(path)
    now = utc_now()
    for i in range(30):
        status = "error" if i % 3 == 0 else ("rejected" if i % 7 == 0 else "ok")
        tier = "strong" if i % 4 == 0 else "cheap"
        record = RequestRecord(id=f"r{i}", ts=iso(now - timedelta(hours=8 * i)), app_id="web" if i % 2 else "batch",
                               release="v2-gemma" if i % 5 == 0 else "v1", canary=i % 5 == 0, mode="router",
                               router="rules", route_score=0.7, tier=tier,
                               model="gpt-oss-120b" if tier == "strong" else "gpt-oss-20b", status=status,
                               cost_usd=0.002, latency_ms=100.0 + 10 * i, router_ms=0.3, prompt_tokens=20,
                               completion_tokens=40, calls=[CallRecord(1, tier, "gpt-oss-20b")])
        store.log_request(record)
    for i in range(20):  # a burst of recent failures: a live error-rate alert
        store.log_request(RequestRecord(id=f"burst{i}", ts=iso(now - timedelta(minutes=i % 5)), app_id="web",
                                        release="v1", status="error" if i % 2 else "ok", tier="cheap",
                                        model="gpt-oss-20b", latency_ms=90.0))
    store.add_feedback("r1", "web", 1, None, iso(now))
    store.add_alert("error_rate", 0.33, 0.2, "Error rate 33% is above 20%", iso(now))
    store.close()


def test_dashboard_without_a_database(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_DB", str(tmp_path / "missing.db"))
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    assert "No request log yet" in at.info[0].value


def test_dashboard_with_traffic(tmp_path, monkeypatch):
    db = tmp_path / "gw.db"
    populate(db)
    monkeypatch.setenv("GATEWAY_DB", str(db))
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Requests"] == "41"  # the default range is the last 7 days: the burst + 21 of 30
    assert labels["Error rate"].endswith("%") and labels["p95 latency"].endswith("ms")
    assert any("Alert (error rate)" in e.value for e in at.error)  # live alert: error rate above the limit
    at.sidebar.selectbox[0].set_value("All time").run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Requests"] == "50"
