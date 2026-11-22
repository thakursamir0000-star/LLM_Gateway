"""Load test: many concurrent clients against one or more gateway URLs, for a fixed time.

    python scripts/load_test.py --url http://127.0.0.1:8000 --url http://127.0.0.1:8001 \\
        --key "$KEY" --concurrency 50 --duration 60 --out results/load_test.json

Each worker sends chat completions back to back (round-robin over the URLs; ``--stream-share`` of them streamed)
and records status and latency. The report: throughput, latency percentiles of successful requests, status codes,
and the gateway's own count from /metrics when reachable. Run it against fake providers
(GATEWAY_FAKE_PROVIDERS=1): with real providers every request is a paid call and the provider's rate limits, not
the gateway, set the result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROMPTS = ["What is 2 + 2?", "Name a primary colour.", "Summarise: the cat sat on the mat.",
           "Explain recursion in one sentence.", "Translate 'hello' to French."]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[index], 1)


async def worker(n: int, client: httpx.AsyncClient, urls: list[str], key: str, deadline: float,
                 stream_share: float, results: list[tuple[int, float, bool]]) -> None:
    i = n
    while time.perf_counter() < deadline:
        url = urls[i % len(urls)]
        stream = (i % 100) < stream_share * 100
        body = {"messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], "stream": stream}
        started = time.perf_counter()
        try:
            if stream:
                async with client.stream("POST", f"{url}/v1/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {key}"}) as response:
                    async for _ in response.aiter_bytes():
                        pass
                    status = response.status_code
            else:
                response = await client.post(f"{url}/v1/chat/completions", json=body,
                                             headers={"Authorization": f"Bearer {key}"})
                status = response.status_code
        except httpx.HTTPError:
            status = 0  # connection error or timeout
        results.append((status, (time.perf_counter() - started) * 1000, stream))
        i += 1


async def gateway_count(client: httpx.AsyncClient, urls: list[str]) -> int | None:
    total = 0
    for url in urls:
        try:
            text = (await client.get(f"{url}/metrics")).text
        except httpx.HTTPError:
            return None
        total += sum(int(float(line.rsplit(" ", 1)[1])) for line in text.splitlines()
                     if line.startswith("gateway_requests_total{"))
    return total


async def run(args: argparse.Namespace, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency * 2)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, verify=not args.insecure,
                                 transport=transport) as client:
        before = await gateway_count(client, args.url)
        results: list[tuple[int, float, bool]] = []
        started = time.perf_counter()
        deadline = started + args.duration
        await asyncio.gather(*(worker(n, client, args.url, args.key, deadline, args.stream_share, results)
                               for n in range(args.concurrency)))
        elapsed = time.perf_counter() - started
        after = await gateway_count(client, args.url)
    ok = [ms for status, ms, _ in results if status == 200]
    statuses = Counter(str(status) for status, _, _ in results)
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "setup": {"urls": args.url, "concurrency": args.concurrency, "duration_s": args.duration,
                  "stream_share": args.stream_share, "note": args.note,
                  "client": f"{platform.system()} {platform.release()}, Python {platform.python_version()}"},
        "requests": len(results),
        "elapsed_s": round(elapsed, 1),
        "throughput_rps": round(len(results) / elapsed, 1),
        "statuses": dict(sorted(statuses.items())),
        "success_rate": round(len(ok) / len(results), 4) if results else None,
        "latency_ms_ok": {"p50": percentile(ok, 0.5), "p95": percentile(ok, 0.95), "p99": percentile(ok, 0.99),
                          "mean": round(statistics.fmean(ok), 1) if ok else None, "max": percentile(ok, 1.0)},
        "gateway_counted": None if before is None or after is None else after - before,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", action="append", required=True, help="gateway base URL (repeat for replicas)")
    parser.add_argument("--key", default=os.environ.get("LOAD_TEST_KEY", ""), help="client API key")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--stream-share", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--insecure", action="store_true", help="accept a self-signed certificate")
    parser.add_argument("--note", default="", help="what was tested (stored with the results)")
    parser.add_argument("--out", help="write the report as JSON here")
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    print(json.dumps(report, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        path.write_text(json.dumps([*existing, report], indent=2) + "\n", encoding="utf-8")
    return 0 if report["requests"] and report["statuses"].get("0", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
