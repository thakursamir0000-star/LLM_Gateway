"""Send a mix of demo requests (and some feedback) to a running gateway.

    GATEWAY_FAKE_PROVIDERS=1 python -m gateway          # terminal 1: gateway with fake models
    python scripts/send_demo_traffic.py --n 300          # terminal 2
    streamlit run dashboard/app.py                       # terminal 3: watch it

Meant for the fake providers, so the dashboard has data without API keys. If
the gateway uses real providers (/health says so), every request spends real
quota, and the script refuses unless you pass --real.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys

import httpx

PROMPTS = [
    "Translate 'see you tomorrow' into Spanish.",
    "Summarise this in one sentence: the meeting moved to Thursday because the room was double-booked.",
    "What is the capital of Australia?",
    "Write a haiku about autumn rain.",
    "Hello! Can you help me plan a picnic menu for six people?",
    "Rephrase this politely: send me the report now.",
    "Step by step, prove that the sum of the first n odd numbers is n^2.",
    "Compare the trade-offs of SQL and NoSQL databases for an analytics workload.",
    "Debug this Python: ```def mean(xs): return sum(xs) / len(xs)``` fails on an empty list. Why, and how to fix it?",
    "A train travels 120 km in 1.5 hours, then 200 km in 2.5 hours. What is its average speed?",
    "Design an architecture for a rate limiter shared by 20 API servers. Explain why.",
    "If P(A)=0.3 and P(B)=0.5 and A and B are independent, what is the probability of A or B?",
]
APPS = ["web", "web", "web", "batch", "support-bot", "demo"]
MODELS = ["auto"] * 7 + ["cascade", "cheap", "strong"]


async def one(client: httpx.AsyncClient, rng: random.Random) -> int:
    payload = {"model": rng.choice(MODELS), "messages": [{"role": "user", "content": rng.choice(PROMPTS)}],
               "user": f"user-{rng.randint(1, 40)}"}
    response = await client.post("/v1/chat/completions", json=payload, headers={"X-App-Id": rng.choice(APPS)})
    if response.status_code == 200 and rng.random() < 0.3:
        request_id = response.json()["id"]
        await client.post("/v1/feedback", json={"request_id": request_id, "score": 1 if rng.random() < 0.8 else -1})
    return response.status_code


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real", action="store_true", help="allow sending to real providers (spends quota)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    async with httpx.AsyncClient(base_url=args.url, timeout=120) as client:
        health = (await client.get("/health")).json()
        if not health.get("fake_providers") and not args.real:
            print("the gateway is using real providers: every request would spend API quota. "
                  "Start it with GATEWAY_FAKE_PROVIDERS=1, or pass --real.", file=sys.stderr)
            return 1
        semaphore = asyncio.Semaphore(args.concurrency)
        codes: dict[int, int] = {}

        async def limited() -> None:
            async with semaphore:
                code = await one(client, rng)
                codes[code] = codes.get(code, 0) + 1

        await asyncio.gather(*(limited() for _ in range(args.n)))
    print("responses by HTTP status:", dict(sorted(codes.items())))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
