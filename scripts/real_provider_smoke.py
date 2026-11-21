"""Real-provider smoke test through the gateway's HTTP API (spends real money: run only when approved).

    python scripts/real_provider_smoke.py --dry-run          # the plan and its worst-case cost; no calls
    python scripts/real_provider_smoke.py --yes              # run it (keys from .env)
    python scripts/real_provider_smoke.py --yes --out results/real_provider_smoke.json

The gateway runs in-process with the shipped config/gateway.yaml, the real provider adapters, a throwaway SQLite
request log and a random client key. Four requests, each capped at ``--max-tokens`` output tokens:

1. Groq, interactive: ``model="auto"`` (strong tier, Groq for interactive traffic), JSON response.
2. OpenRouter, batch: ``X-Gateway-Latency: batch`` sends the strong tier to OpenRouter.
3. Forced fallback: a second gateway whose Groq adapter holds an invalid key, so Groq answers 401 (no charge) and the
   request falls back to the same tier on OpenRouter.
4. Streaming: the Groq request again with ``stream: true`` and usage, checking SSE chunks, ``[DONE]`` and timing.

It checks the OpenAI response shape, the answer, usage, cost and latency, the logged request and call rows, and that
no budget reservation is left behind. Before the first call it refuses to run when the worst case (every request
using all its output tokens at list price) is above ``--max-cost-usd``; it stops early if the measured spend passes it.
API keys are never printed. Costs for Groq are list prices (its free tier may not bill them); OpenRouter's are the
cost it reports.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import load_config
from gateway.monitoring.store import Store
from gateway.providers import build_providers
from gateway.providers.groq import GroqProvider

PROMPT = "In one short sentence: why is the sky blue?"
PLAN = [  # (name, latency class, stream, gateway, models the request may call)
    ("groq_interactive", "interactive", False, "main", ["gpt-oss-120b"]),
    ("openrouter_batch", "batch", False, "main", ["or-gpt-oss-120b"]),
    ("forced_fallback", "interactive", False, "broken_groq", ["gpt-oss-120b", "or-gpt-oss-120b"]),
    ("groq_streaming", "interactive", True, "main", ["gpt-oss-120b"]),
]


def worst_case(config, models: list[str], max_tokens: int) -> float:
    prompt_tokens = len(PROMPT) // 3 + 64
    return sum(config.models[m].cost(prompt_tokens, max_tokens) for m in models)


def run_request(client: TestClient, key: str, latency: str, stream: bool, max_tokens: int) -> dict:
    body = {"model": "auto", "max_tokens": max_tokens, "messages": [{"role": "user", "content": PROMPT}]}
    headers = {"Authorization": f"Bearer {key}", "X-Gateway-Latency": latency}
    started = time.perf_counter()
    if not stream:
        response = client.post("/v1/chat/completions", headers=headers, json=body)
        elapsed = time.perf_counter() - started
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        gateway = data.get("gateway") or {}
        return {
            "http_status": response.status_code,
            "shape_ok": data.get("object") == "chat.completion" and message.get("role") == "assistant"
            and set(data.get("usage") or {}) >= {"prompt_tokens", "completion_tokens", "total_tokens"},
            "answer": (message.get("content") or "")[:120], "finish_reason": choice.get("finish_reason"),
            "provider_model": data.get("model"), "usage": data.get("usage"),
            "gateway_model": gateway.get("model"), "tier": gateway.get("tier"), "fallback": gateway.get("fallback"),
            "cost_usd": gateway.get("cost_usd"), "client_latency_s": round(elapsed, 3),
            "error": data.get("error"),
        }
    body |= {"stream": True, "stream_options": {"include_usage": True}}
    first_chunk_s = None
    events: list[str] = []
    with client.stream("POST", "/v1/chat/completions", headers=headers, json=body) as response:
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        for line in response.iter_lines():
            if line.startswith("data: "):
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - started
                events.append(line[6:])
    elapsed = time.perf_counter() - started
    chunks = [json.loads(e) for e in events if e.startswith("{")]
    text = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in chunks if c.get("choices"))
    usage_chunk = next((c for c in chunks if c.get("usage")), {})
    return {
        "http_status": status, "content_type": content_type,
        "shape_ok": status == 200 and content_type.startswith("text/event-stream") and bool(chunks)
        and all(c.get("object") == "chat.completion.chunk" for c in chunks) and events[-1] == "[DONE]",
        "chunks": len(chunks), "ends_with_done": bool(events) and events[-1] == "[DONE]",
        "answer": text[:120], "usage": usage_chunk.get("usage"),
        "gateway_model": (usage_chunk.get("gateway") or {}).get("model"),
        "cost_usd": (usage_chunk.get("gateway") or {}).get("cost_usd"),
        "time_to_first_chunk_s": round(first_chunk_s or 0.0, 3), "client_latency_s": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and worst-case cost; no calls")
    parser.add_argument("--yes", action="store_true", help="really call the providers (spends money)")
    parser.add_argument("--max-cost-usd", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", help="write the JSON report here")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config()
    config.security.expose_internal_metadata = True  # the report reads the gateway's routing metadata
    worst = {name: worst_case(config, models, args.max_tokens) for name, _, _, _, models in PLAN}
    keys_present = {name: bool(os.environ.get(config.providers[name].api_key_env)) for name in ("groq", "openrouter")}
    plan = {"requests": [{"name": n, "latency_class": lc, "stream": s, "may_call": m, "worst_case_usd": round(worst[n], 6)}
                         for n, lc, s, _, m in PLAN],
            "worst_case_total_usd": round(sum(worst.values()), 6), "cap_usd": args.max_cost_usd,
            "max_tokens": args.max_tokens, "keys_present": keys_present}
    if args.dry_run or not args.yes:
        print(json.dumps({"dry_run": True, **plan}, indent=2))
        return 0
    if plan["worst_case_total_usd"] > args.max_cost_usd:
        print(f"refusing: worst case ${plan['worst_case_total_usd']} is above the cap ${args.max_cost_usd}")
        return 1
    if not all(keys_present.values()):
        print(f"refusing: missing provider keys {keys_present}")
        return 1

    client_key = secrets.token_hex(16)
    os.environ["GATEWAY_API_KEYS"] = f"smoke:{client_key}"
    os.environ.pop("GATEWAY_ALLOW_ANONYMOUS", None)
    report: dict = {"started_at": datetime.now(UTC).isoformat(timespec="seconds"), "plan": plan, "results": {}}
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "smoke.db")
        real = build_providers(config)
        broken = dict(build_providers(config))
        groq_spec = config.providers["groq"]
        broken["groq"] = GroqProvider("invalid-smoke-test-key", groq_spec.base_url, groq_spec.timeout_s)
        apps = {"main": create_app(config, providers=real, store=store),
                "broken_groq": create_app(config, providers=broken, store=store)}
        spent = 0.0
        with TestClient(apps["main"]) as main_client, TestClient(apps["broken_groq"]) as broken_client:
            clients = {"main": main_client, "broken_groq": broken_client}
            for name, latency, stream, app_name, _ in PLAN:
                if spent > args.max_cost_usd:
                    report["results"][name] = {"skipped": f"spend ${spent:.6f} passed the cap"}
                    continue
                result = run_request(clients[app_name], client_key, latency, stream, args.max_tokens)
                spent += float(result.get("cost_usd") or 0.0)
                report["results"][name] = result
        requests = store.query("SELECT id, latency_class, tier, model, provider_model, fallback, status, http_status, "
                               "error_type, prompt_tokens, completion_tokens, cost_usd, latency_ms, stream "
                               "FROM requests ORDER BY ts")
        calls = store.query("SELECT request_id, seq, model, status, error_type, prompt_tokens, completion_tokens, "
                            "cost_usd, latency_ms FROM calls ORDER BY ts, seq")
        reservations = store.reservations()
        store.close()

    results = report["results"]
    fallback = results.get("forced_fallback", {})
    checks = {
        "groq interactive answered by Groq": results["groq_interactive"].get("http_status") == 200
        and results["groq_interactive"].get("gateway_model") == "gpt-oss-120b",
        "openrouter batch answered by OpenRouter": results["openrouter_batch"].get("http_status") == 200
        and results["openrouter_batch"].get("gateway_model") == "or-gpt-oss-120b",
        "forced fallback reached OpenRouter": fallback.get("http_status") == 200 and fallback.get("fallback") is True
        and fallback.get("gateway_model") == "or-gpt-oss-120b",
        "streaming delivered chunks and [DONE]": bool(results["groq_streaming"].get("shape_ok")),
        "OpenAI response shape": all(r.get("shape_ok") for r in results.values() if "skipped" not in r),
        "every request logged once": len(requests) == len(PLAN),
        "failed Groq call logged for the fallback": any(c["status"] == "error" and c["model"] == "gpt-oss-120b"
                                                       for c in calls),
        "no budget reservation left": reservations == [],
        "spend within the cap": spent <= args.max_cost_usd,
    }
    report.update({"spent_usd": round(spent, 8), "logged_requests": requests, "logged_calls": calls,
                   "reservations_left": len(reservations), "checks": checks, "passed": all(checks.values())})
    rendered = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
