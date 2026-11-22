"""scripts/load_test.py against an in-process gateway (fake providers)."""

from __future__ import annotations

import argparse
import asyncio
import random

import httpx

import scripts.load_test as load_test
from gateway.app import create_app
from gateway.monitoring.store import Store
from gateway.providers.fake import FakeProvider
from gateway.routing import RulesRouter
from tests.conftest import make_config


def test_percentile():
    assert load_test.percentile([], 0.5) is None
    assert load_test.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert load_test.percentile([5.0, 1.0], 1.0) == 5.0


def test_report_counts_match_the_gateway(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_GATEWAY_KEYS", "load:load-key")
    store = Store(tmp_path / "gateway.db")
    app = create_app(make_config(auth={"allow_anonymous": False}, client_limits={"default": {"rpm": 5}}),
                     providers={"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini")}, store=store,
                     router=RulesRouter(), rng=random.Random(0))
    args = argparse.Namespace(url=["http://gateway"], key="load-key", concurrency=3, duration=0.5, stream_share=0.5,
                              timeout=10.0, insecure=False, note="test")
    report = asyncio.run(load_test.run(args, transport=httpx.ASGITransport(app=app)))
    store.close()
    assert report["requests"] > 5 and report["statuses"]["200"] == 5  # the client limit admitted five
    assert report["statuses"]["429"] == report["requests"] - 5
    assert report["gateway_counted"] == report["requests"]
    assert report["latency_ms_ok"]["p50"] is not None
