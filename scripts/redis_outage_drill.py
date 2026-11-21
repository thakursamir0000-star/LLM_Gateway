"""Redis outage drill: stop a real Redis container under a running gateway, then restart it.

    python scripts/redis_outage_drill.py --redis-url redis://localhost:6379/15 --container gateway-redis

The gateway runs in-process with fake providers (no API calls), a throwaway SQLite request log and the Redis
state backend. The drill checks, and prints as JSON:

1. healthy: /ready 200, a request to a rate-limited model answers 200;
2. outage (``docker stop``): /ready 503, /health "degraded", the same request is refused with 429 under
   ``fail_closed`` (or answered within this replica's share under ``local_share``), and the provider is not
   called beyond that;
3. recovery (``docker start``): /ready returns to 200 and requests are answered again; the time it took.

Exits 1 if any expectation fails. Use a Redis database number you don't use for anything else (it is flushed).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import GatewayConfig, RedisConfig
from gateway.monitoring.store import Store
from gateway.providers.base import Completion
from gateway.providers.fake import FakeProvider
from gateway.state.redis_backend import RedisStateBackend

CONFIG = {
    "models": {
        "small": {"provider": "fake", "model": "small-model", "limits": {"rpm": 30}},
        "large": {"provider": "fake", "model": "large-model", "limits": {"rpm": 30}},
    },
    "providers": {"fake": {"base_url": "http://fake.invalid", "api_key_env": "DRILL_UNUSED_KEY"}},
    "prompts": {"plain@1": ""},
    "releases": {"v1": {"cheap": "small", "strong": "large", "prompt": "plain@1"}},
    "deployment": {"active": "v1"},
    "routing": {"default_mode": "strong", "router": "rules"},
    "auth": {"allow_anonymous": True},
}
BODY = {"model": "strong", "messages": [{"role": "user", "content": "drill"}]}


def docker(action: str, container: str) -> None:
    subprocess.run(["docker", action, container], check=True, capture_output=True, text=True)


def probe(client: TestClient) -> dict:
    chat = client.post("/v1/chat/completions", json=BODY, headers={"X-App-Id": "drill"})
    return {"ready": client.get("/ready").status_code, "health": client.get("/health").json().get("status"),
            "chat": chat.status_code, "chat_code": (chat.json().get("error") or {}).get("code")
            if chat.status_code != 200 else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redis-url", required=True, help="a Redis database used only by the drill (it is flushed)")
    parser.add_argument("--container", required=True, help="docker container running that Redis")
    parser.add_argument("--policy", choices=["fail_closed", "local_share"], default="fail_closed")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--recovery-timeout", type=float, default=60.0)
    args = parser.parse_args()

    config = GatewayConfig.model_validate(CONFIG)
    state = RedisStateBackend(RedisConfig(url=args.redis_url, socket_timeout_s=0.5, connect_timeout_s=0.5,
                                          outage_policy=args.policy, replicas=args.replicas))
    state.client.flushdb()
    calls = FakeProvider("fake", lambda model, messages, params: Completion("ok", "stop", 5, 5, model))
    report: dict = {"policy": args.policy, "replicas": args.replicas}
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "drill.db")
        app = create_app(config, providers={"fake": calls}, state=state, store=store)
        with TestClient(app) as client:
            report["healthy"] = probe(client)
            docker("stop", args.container)
            try:
                before = len(calls.calls)
                report["outage"] = probe(client)
                report["outage"]["provider_calls"] = len(calls.calls) - before
            finally:
                docker("start", args.container)
            started = time.monotonic()
            while client.get("/ready").status_code != 200:
                if time.monotonic() - started > args.recovery_timeout:
                    break
                time.sleep(0.5)
            report["recovery_s"] = round(time.monotonic() - started, 2)
            report["recovered"] = probe(client)
        store.close()
        state.close()

    expected_outage_chat = 429 if args.policy == "fail_closed" else 200
    checks = {
        "healthy: ready and answering": report["healthy"]["ready"] == 200 and report["healthy"]["chat"] == 200,
        "outage: /ready 503": report["outage"]["ready"] == 503,
        "outage: /health degraded": report["outage"]["health"] == "degraded",
        f"outage: chat {expected_outage_chat}": report["outage"]["chat"] == expected_outage_chat,
        "outage: no provider call when failing closed": args.policy != "fail_closed"
        or report["outage"]["provider_calls"] == 0,
        "recovered: ready and answering": report["recovered"]["ready"] == 200 and report["recovered"]["chat"] == 200,
    }
    report["checks"] = checks
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
