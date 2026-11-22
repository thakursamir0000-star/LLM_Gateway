"""Two-replica Docker Compose validation: both gateway replicas share PostgreSQL and Redis state.

    python scripts/multi_instance_check.py            # build, start, check, tear down
    python scripts/multi_instance_check.py --keep     # leave the stack running afterwards

Runs the repository's docker-compose.yml (PostgreSQL 16, Redis 7, migrations, two gateway replicas, the
dashboard and the nginx ``proxy`` profile) under its own Compose project (``llmgw-mi``) with fake providers,
so no provider is called and no existing Compose data is touched. The test config is the shipped
config/gateway.yaml with tiny budgets for two test apps, a low rate limit on one model and no rate-limit
waiting; it is written to var/multi-instance/ (git-ignored) with a Compose override that mounts it.

Checks (printed as JSON; exit 1 if any fails):
- both replicas and the proxy are healthy and ready;
- one daily budget: after one replica refuses an app, the other refuses it too, and a concurrent burst
  across both replicas overshoots by at most one request;
- one Redis rate limit: a model limited to 4 requests/minute gets at most 4 calls across both replicas;
- one provider cooldown and one circuit breaker: state written to Redis is honoured by both replicas;
- the HTTPS proxy: HTTP redirects to HTTPS, TLS 1.2/1.3 only, security headers, /admin/ blocked, the dashboard
  needs its password, and the gateway ports are published on loopback only;
- SSE streaming through the HTTPS proxy delivers several chunks and [DONE];
- a restarted replica comes back ready with the shared state intact;
- PostgreSQL connections stay bounded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import secrets
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_proxy_secrets import make_secrets  # noqa: E402

WORK = ROOT / "var" / "multi-instance"
PROJECT = "llmgw-mi"
REPLICAS = {"gateway": "http://127.0.0.1:8000", "gateway-replica": "http://127.0.0.1:8001"}
PROXY = "https://127.0.0.1"
PROXY_HTTP = "http://127.0.0.1"
TLS_DIR = WORK / "tls"
CERT = str(TLS_DIR / "server.crt")  # self-signed: trusted as its own CA
LIMITED_MODEL = "gpt-oss-120b"  # 4 requests per minute in the test config
BUDGET_MODEL = "gpt-oss-20b"
COOLDOWN_MODEL = "or-gpt-oss-20b"
BREAKER_MODEL, BREAKER_PROVIDER = "gemma-4-31b", "gemini"


def compose(*args: str, check: bool = True) -> str:
    command = ["docker", "compose", "-p", PROJECT, "-f", str(ROOT / "docker-compose.yml"),
               "-f", str(WORK / "compose.override.yml"), "--profile", "proxy", *args]
    env = {**os.environ, "GATEWAY_TLS_DIR": TLS_DIR.as_posix()}
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def psql(sql: str) -> str:
    return compose("exec", "-T", "postgres", "psql", "-U", "gateway", "-d", "gateway", "-tAc", sql)


def redis(*args: str) -> str:
    return compose("exec", "-T", "redis", "redis-cli", *args)


def write_test_setup(keys: dict[str, str], admin_key: str, dashboard_password: str) -> None:
    config = yaml.safe_load((ROOT / "config" / "gateway.yaml").read_text(encoding="utf-8"))
    for name, model in config["models"].items():  # only the rate-limit test model is limited
        model["limits"] = {"rpm": 4, "tpm": 0} if name == LIMITED_MODEL else {"rpm": 0, "tpm": 0}
    config["retries"]["max_wait_s"] = 0  # refuse at once instead of waiting for a rate-limit slot
    config["budgets"]["apps"].update({"mi-budget": {"daily_usd": 0.00005}, "mi-burst": {"daily_usd": 0.00002}})
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "gateway.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = {"GATEWAY_FAKE_PROVIDERS": "1", "GATEWAY_FAKE_PACING": "1",
                   "GATEWAY_CONFIG": "/app/test-config/gateway.yaml",
                   "GATEWAY_ADMIN_KEY": admin_key,
                   "GATEWAY_API_KEYS": ",".join(f"{app}:{key}" for app, key in keys.items())}
    service = {"environment": environment,
               "volumes": [{"type": "bind", "source": (WORK / "gateway.yaml").as_posix(),
                            "target": "/app/test-config/gateway.yaml", "read_only": True}]}
    override = {"services": {"gateway": service, "gateway-replica": service}}
    (WORK / "compose.override.yml").write_text(yaml.safe_dump(override, sort_keys=False), encoding="utf-8")
    make_secrets(TLS_DIR, "admin", dashboard_password, days=2)


def wait_ready(url: str, timeout: float = 120.0) -> bool:
    verify = CERT if url.startswith("https") else True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/ready", timeout=3, verify=verify).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


def chat(base: str, key: str, model: str, *, stream: bool = False) -> httpx.Response:
    return httpx.post(f"{base}/v1/chat/completions", timeout=30, headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": [{"role": "user", "content": "hello"}], "stream": stream})


def check_budget(keys: dict[str, str]) -> dict:
    bases = list(REPLICAS.values())
    def code(response: httpx.Response) -> str:
        return "ok" if response.status_code == 200 else response.json().get("error", {}).get("code", "")

    statuses = []
    for i in range(60):  # alternate replicas until the shared budget runs out
        statuses.append(code(chat(bases[i % 2], keys["mi-budget"], BUDGET_MODEL)))
        if statuses[-1] == "budget_exceeded":
            break
    other = code(chat(bases[len(statuses) % 2], keys["mi-budget"], BUDGET_MODEL))
    spent, worst = (float(v) for v in psql(
        "SELECT coalesce(sum(cost_usd),0), coalesce(max(cost_usd),0) FROM requests WHERE app_id='mi-budget'").split("|"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        burst = list(pool.map(lambda i: code(chat(bases[i % 2], keys["mi-burst"], BUDGET_MODEL)), range(20)))
    burst_spent, burst_worst = (float(v) for v in psql(
        "SELECT coalesce(sum(cost_usd),0), coalesce(max(cost_usd),0) FROM requests WHERE app_id='mi-burst'").split("|"))
    return {"sequential_answered_before_refusal": statuses.count("ok"), "refused_with": statuses[-1],
            "other_replica_after_refusal": other, "spent": spent, "limit": 0.00005,
            "burst_answered": burst.count("ok"), "burst_refused": burst.count("budget_exceeded"),
            "burst_other": len(burst) - burst.count("ok") - burst.count("budget_exceeded"),
            "burst_spent": burst_spent, "burst_limit": 0.00002, "burst_max_request_cost": burst_worst,
            "ok": statuses[-1] == "budget_exceeded" and other == "budget_exceeded" and spent <= 0.00005 + worst
            and burst_spent <= 0.00002 + burst_worst and burst.count("budget_exceeded") > 0}


def check_rate_limit(keys: dict[str, str]) -> dict:
    bases = list(REPLICAS.values())
    codes = [chat(bases[i % 2], keys["mi-rate"], LIMITED_MODEL).status_code for i in range(10)]
    attempts = int(psql(f"SELECT count(*) FROM calls WHERE model='{LIMITED_MODEL}' AND status = 'ok'") or 0)
    per_replica_429 = [codes[i::2].count(429) for i in range(2)]
    return {"codes": codes, "provider_calls": attempts, "replica_429s": per_replica_429,
            "ok": attempts <= 4 and all(per_replica_429) and codes.count(429) >= 6}


def check_cooldown_and_breaker(keys: dict[str, str]) -> dict:
    now = float(redis("TIME").splitlines()[0])
    redis("SET", f"gateway:cooldown:{COOLDOWN_MODEL}", str(now + 60), "EX", "60")
    redis("HSET", f"gateway:circuit:{BREAKER_PROVIDER}", "state", "OPEN", "opened_at", str(now), "failures", "5")
    cooled = {name: chat(base, keys["mi-rate"], COOLDOWN_MODEL).json().get("error", {}).get("code")
              for name, base in REPLICAS.items()}
    broken = {name: chat(base, keys["mi-rate"], BREAKER_MODEL).json().get("error", {}).get("code")
              for name, base in REPLICAS.items()}
    redis("DEL", f"gateway:cooldown:{COOLDOWN_MODEL}", f"gateway:circuit:{BREAKER_PROVIDER}")
    return {"cooldown": cooled, "breaker": broken,
            "ok": set(cooled.values()) == {"upstream_quota_exhausted"} and set(broken.values()) == {"upstream_unavailable"}}


def check_proxy_stream(keys: dict[str, str]) -> dict:
    events: list[str] = []
    with httpx.stream("POST", f"{PROXY}/v1/chat/completions", timeout=30, verify=CERT,
                      headers={"Authorization": f"Bearer {keys['mi-stream']}"},
                      json={"model": BUDGET_MODEL, "stream": True,
                            "messages": [{"role": "user", "content": "stream a few words please"}]}) as response:
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        events = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
    chunks = [e for e in events if e.startswith("{") and "chat.completion.chunk" in e]
    return {"status": status, "content_type": content_type, "chunks": len(chunks), "last": events[-1] if events else None,
            "ok": status == 200 and content_type.startswith("text/event-stream") and len(chunks) >= 2
            and events[-1] == "[DONE]"}


def tls_version(maximum: ssl.TLSVersion, minimum: ssl.TLSVersion | None = None) -> str:
    """The protocol the proxy negotiates for a client limited to ``maximum``, or why the handshake failed."""
    context = ssl.create_default_context(cafile=CERT)
    if minimum is not None:
        context.set_ciphers("DEFAULT:@SECLEVEL=0")  # let this client offer an old protocol at all
        context.minimum_version = minimum
    context.maximum_version = maximum
    try:
        with socket.create_connection(("127.0.0.1", 443), timeout=5) as raw,                 context.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
            return tls.version() or "unknown"
    except (ssl.SSLError, OSError) as error:
        return f"refused ({type(error).__name__})"


def check_https(keys: dict[str, str], admin_key: str, dashboard_password: str) -> dict:
    redirect = httpx.get(f"{PROXY_HTTP}/v1/models", follow_redirects=False, timeout=10)
    api = httpx.get(f"{PROXY}/v1/models", verify=CERT, timeout=10)
    headers = {name: api.headers.get(name) for name in ("strict-transport-security", "x-content-type-options",
                                                          "x-frame-options", "referrer-policy")}
    admin_via_proxy = httpx.get(f"{PROXY}/admin/status", headers={"X-Admin-Key": admin_key}, verify=CERT, timeout=10)
    admin_on_loopback = httpx.get(f"{REPLICAS['gateway']}/admin/status", headers={"X-Admin-Key": admin_key}, timeout=10)
    dashboard_anonymous = httpx.get(f"{PROXY}/dashboard/", verify=CERT, timeout=10)
    dashboard_wrong = httpx.get(f"{PROXY}/dashboard/", auth=("admin", "wrong-password"), verify=CERT, timeout=10)
    dashboard_ok = httpx.get(f"{PROXY}/dashboard/", auth=("admin", dashboard_password), verify=CERT, timeout=30)
    ports = {name: compose("port", name, port) for name, port in (("gateway", "8000"), ("gateway-replica", "8001"))}
    protocols = {"default": tls_version(ssl.TLSVersion.TLSv1_3), "max_tls1.2": tls_version(ssl.TLSVersion.TLSv1_2),
                 "max_tls1.1": tls_version(ssl.TLSVersion.TLSv1_1, ssl.TLSVersion.TLSv1)}
    report = {
        "http_redirect": [redirect.status_code, redirect.headers.get("location")], "headers": headers,
        "admin_via_proxy": admin_via_proxy.status_code, "admin_on_loopback": admin_on_loopback.status_code,
        "dashboard": {"no_password": dashboard_anonymous.status_code, "wrong_password": dashboard_wrong.status_code,
                      "right_password": dashboard_ok.status_code},
        "published_ports": ports, "tls": protocols,
    }
    report["ok"] = (redirect.status_code == 301 and (redirect.headers.get("location") or "").startswith("https://")
                    and api.status_code == 200 and all(headers.values())
                    and admin_via_proxy.status_code == 403 and admin_on_loopback.status_code == 200
                    and (dashboard_anonymous.status_code, dashboard_wrong.status_code, dashboard_ok.status_code)
                    == (401, 401, 200)
                    and all(port.startswith("127.0.0.1:") for port in ports.values())
                    and protocols["default"] == "TLSv1.3" and protocols["max_tls1.2"] == "TLSv1.2"
                    and protocols["max_tls1.1"].startswith("refused"))
    return report


def check_restart(keys: dict[str, str]) -> dict:
    compose("restart", "gateway-replica")
    ready = wait_ready(REPLICAS["gateway-replica"])
    still_refused = chat(REPLICAS["gateway-replica"], keys["mi-budget"], BUDGET_MODEL).status_code
    for _ in range(3):  # the demo fake provider fails 3% of calls on purpose
        answered = chat(REPLICAS["gateway-replica"], keys["mi-stream"], BUDGET_MODEL).status_code
        if answered == 200:
            break
    return {"ready_after_restart": ready, "exhausted_budget_still_refused": still_refused,
            "other_app_answered": answered, "ok": ready and still_refused == 429 and answered == 200}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    parser.add_argument("--no-build", action="store_true", help="reuse the existing llm-gateway:local image")
    args = parser.parse_args()

    keys = {app: secrets.token_hex(16) for app in ("mi-budget", "mi-burst", "mi-rate", "mi-stream")}
    admin_key, dashboard_password = secrets.token_hex(16), secrets.token_urlsafe(12)
    write_test_setup(keys, admin_key, dashboard_password)
    report: dict = {}
    try:
        compose("down", "-v", check=False)
        compose("up", "-d", *([] if args.no_build else ["--build"]))
        report["ready"] = {name: wait_ready(base) for name, base in {**REPLICAS, "proxy": PROXY}.items()}
        report["ready"]["ok"] = all(report["ready"].values())
        report["budget"] = check_budget(keys)
        report["rate_limit"] = check_rate_limit(keys)
        report["cooldown_and_breaker"] = check_cooldown_and_breaker(keys)
        report["https"] = check_https(keys, admin_key, dashboard_password)
        report["proxy_stream"] = check_proxy_stream(keys)
        report["restart"] = check_restart(keys)
        connections = int(psql("SELECT count(*) FROM pg_stat_activity WHERE datname = 'gateway'") or 0)
        report["postgres_connections"] = {"open": connections, "ok": connections <= 2 * 15 + 10}
    finally:
        if not args.keep:
            compose("down", "-v", check=False)
    report["passed"] = all(section.get("ok") for section in report.values() if isinstance(section, dict))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
