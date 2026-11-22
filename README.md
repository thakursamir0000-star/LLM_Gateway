# LLM Gateway

An OpenAI-compatible gateway between applications and LLM providers (Groq, OpenRouter, the Gemini API). Apps change
one URL; the gateway picks the model and provider for each request, falls back when a provider fails, enforces
per-client limits and daily budgets across replicas, and records every call's tokens, cost, latency and route
decision. It ships with PostgreSQL and Redis for multi-replica operation, Prometheus metrics and a Grafana dashboard,
and an offline lab that trains and evaluates an ML router.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<a gateway client key>")
reply = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "What is 17 * 23?"}])
```

> **Status:** a personal portfolio project, complete and tested locally; **not deployed**. The trained ML router is
> evaluated but **not active**. See [What is tested, and how](#what-is-tested-and-how) for exactly what each claim
> rests on.

## Contents

[Problem](#problem-and-motivation) · [Features](#key-features) · [Architecture](#architecture) ·
[Tech stack](#technology-stack) · [Request lifecycle](#request-lifecycle) · [Routing](#model-routing-and-fallback) ·
[Data stores](#postgresql-and-redis) · [Resilience](#rate-limiting-and-resilience) ·
[Monitoring](#monitoring-prometheus-and-grafana) · [Security](#security-and-api-key-management) ·
[Load test](#load-testing-results) · [Router evaluation](#router-evaluation-results) ·
[Testing status](#what-is-tested-and-how) · [Setup](#installation-and-local-setup) ·
[Configuration](#environment-configuration) · [API examples](#api-examples) · [Tests](#running-tests) ·
[Docker](#docker-setup) · [Limitations](#limitations-and-future-improvements) · [Docs](#documentation)

## Problem and motivation

Applications that call LLMs directly hard-code one provider and one model. That couples them to the provider's
outages, rate limits and prices; it sends easy questions to the most expensive model; and it leaves no shared record
of what was spent, how slow it was, or what failed. Running several app replicas makes it worse: each one keeps its
own idea of the provider's rate limit.

The gateway centralises those decisions behind the API the apps already use (OpenAI's), so model choice, fallback,
spend limits and observability are configuration and infrastructure rather than application code. The research
question behind it: **can a small classifier send easy questions to a cheaper model without losing accuracy?** The
answer, measured honestly below, is "a little, and not significantly".

## Key features

- **OpenAI-compatible API**: `POST /v1/chat/completions` (JSON and server-sent-event streaming), `GET /v1/models`,
  `POST /v1/feedback`; OpenAI's error format; works with the official OpenAI SDK by changing `base_url`.
- **Three providers**: Groq and OpenRouter (`gpt-oss-20b` cheap, `gpt-oss-120b` strong) and the Gemini API, behind
  one adapter interface, with provider error text stripped of account details.
- **Routing**: per-request modes (`auto`, `cheap`, `strong`, `router`, `cascade`, or a pinned model), a latency class
  that picks the provider (fast Groq for interactive traffic, cheaper OpenRouter for batch), releases with canaries and
  rollback.
- **Resilience**: bounded retries, cross-provider fallback, quota cooldowns, per-provider circuit breakers, and a
  defined policy for a Redis outage.
- **Multi-replica state**: PostgreSQL (request log, budgets, releases; Alembic migrations) and Redis (atomic Lua
  scripts for rate limits, client limits, cooldowns and circuit breakers).
- **Limits and spend control**: per-client requests/minute and requests-in-flight, per-model provider limits, and
  daily USD budgets with atomic cost reservations.
- **Observability**: Prometheus metrics, JSON access logs without prompt text, a provisioned Grafana dashboard, four
  alert rules, and a Streamlit request-log dashboard.
- **Security**: fail-closed authentication, SHA-256-hashed client keys with zero-downtime rotation, an HTTPS proxy
  (TLS 1.2/1.3, security headers), admin API and metrics not published, production start-up guards.
- **Operations**: retention, backup, restore and a restore drill; a load-test script; a deployment runbook.
- **ML router lab**: benchmark collection (1,200 GSM8K and MMLU questions answered by both models), a sentence-
  embedding + logistic-regression router, held-out evaluation with significance tests, and a staged rollout
  mechanism (shadow mode, then a traffic share with a control group).

## Architecture

The request path as implemented. Dashed boxes exist but are off in the shipped configuration. Source:
[`docs/architecture.mmd`](docs/architecture.mmd).

```mermaid
flowchart TB
    Client["Client application<br/>(OpenAI SDK or curl)"]

    subgraph Edge["Edge (Compose profile: proxy)"]
        Proxy["nginx HTTPS proxy<br/>TLS 1.2/1.3, security headers<br/>/admin and /metrics blocked"]
    end

    subgraph GW["FastAPI gateway (2 replicas, stateless)"]
        direction TB
        Limits["Body size limit<br/>+ request schema validation"]
        Auth["API key authentication<br/>hashed keys, constant-time"]
        ClientRL["Per-client limits<br/>requests/min, in flight"]
        Budget["Daily budget<br/>atomic cost reservation"]
        Release["Release + canary<br/>model pair, prompt version"]
        Route["Routing engine<br/>auto / cheap / strong / cascade / router<br/>latency class picks the provider"]
        Shadow["Shadow scoring +<br/>classifier share (control group)"]:::off
        Registry["Model registry<br/>cooldowns, circuit breakers,<br/>provider rate limits, retries"]
        Fallback["Fallback chain<br/>same tier other provider,<br/>then other tier"]
        Respond["Response<br/>JSON or SSE stream"]
        Finish["Finish: request log,<br/>metrics, JSON access log"]
    end

    subgraph Providers["LLM providers"]
        Groq["Groq<br/>gpt-oss-20b / 120b<br/>(interactive)"]
        OR["OpenRouter<br/>gpt-oss-20b / 120b<br/>(batch)"]
        Gemini["Gemini API<br/>Gemini / Gemma"]
    end

    subgraph State["Shared state"]
        Redis[("Redis 7<br/>client limits, rate limits,<br/>cooldowns, circuit breakers")]
        PG[("PostgreSQL 16<br/>requests, calls, budgets,<br/>releases, feedback, alerts")]
    end

    subgraph Obs["Observability (Compose profile: monitoring)"]
        Prom["Prometheus<br/>scrapes /metrics, 4 alert rules"]
        Grafana["Grafana<br/>LLM Gateway dashboard"]
        Dash["Streamlit dashboard<br/>(reads the request log)"]
    end

    subgraph Lab["Router lab (offline, no live traffic)"]
        Bench["GSM8K + MMLU<br/>1,200 questions"] --> Labels["collect_labels.py<br/>cheap vs strong answers"]
        Labels --> Train["train_router.py<br/>MiniLM embedding + logistic regression"]
        Train --> Artifact["router.json<br/>(not installed)"]:::off
        Train --> Eval["evaluate_router.py<br/>300 held-out questions"]
    end

    Client -->|"HTTPS"| Proxy --> Limits
    Client -.->|"loopback / development"| Limits
    Limits --> Auth --> ClientRL --> Budget --> Release --> Route
    Route -.-> Shadow
    Route --> Registry --> Groq & OR & Gemini
    Registry --> Fallback --> Registry
    Groq & OR & Gemini --> Respond --> Client
    Respond --> Finish

    ClientRL <--> Redis
    Registry <--> Redis
    Budget <--> PG
    Release <--> PG
    Finish --> PG
    Prom -->|"scrapes /metrics"| Finish
    Prom --> Grafana
    PG --> Dash
    Artifact -.->|"after approval only"| Route

    classDef off stroke-dasharray: 5 5
```

The gateway replicas hold no state of their own: anything that must agree across replicas (limits, breaker states,
budgets, the active release) lives in Redis or PostgreSQL, so replicas can be added behind the proxy.

## Technology stack

| Area | Technology |
|---|---|
| API | Python 3.11–3.13, FastAPI, Pydantic v2, Uvicorn, httpx (async provider clients, SSE) |
| Storage | PostgreSQL 16 (SQLAlchemy 2 Core, psycopg 3, Alembic); SQLite for single-process development |
| Shared state | Redis 7 (Lua scripts, server time) |
| Edge | nginx (TLS termination, round-robin over replicas, SSE pass-through) |
| Observability | Prometheus text format (hand-written exporter, no client library), Prometheus 2.55, Grafana 11, Streamlit, Altair |
| ML | sentence-transformers (`all-MiniLM-L6-v2`), NumPy logistic regression (Newton's method, checked against scikit-learn) |
| Packaging | Docker, Docker Compose profiles, hash-locked dependencies (uv), digest-pinned base image |
| Quality | pytest, Ruff, GitHub Actions (Python 3.11 and 3.12 with PostgreSQL and Redis services), pip-audit, Dependabot |

## Request lifecycle

One `POST /v1/chat/completions`, in code order (`gateway/app.py`, `gateway/service.py`):

1. **Edge**: the body-size limit (10 MB) and the OpenAI request schema are checked.
2. **Authenticate**: the bearer key is matched in constant time against every configured key (plain or SHA-256); it
   names the client app. Invalid keys get 401 and use none of the client's allowance.
3. **Client limits**: the app's requests/minute and requests-in-flight counters are updated atomically in Redis; over
   either limit → 429 `client_rate_limited` with `Retry-After`. A stream holds its slot until it ends.
4. **Budget**: the app's spend today is checked; at the daily limit → 429 `budget_exceeded`; past 80% routed requests
   go to the cheap model.
5. **Release**: the active release (or the canary, for its share of traffic) fixes the model pair and prompt version.
6. **Route**: the mode and latency class choose the tier and model (below). The maximum possible cost is reserved
   against the budget in one transaction before any provider call.
7. **Call**: the model registry checks the provider's cooldown and circuit breaker, waits for its rate limit (shared
   in Redis), and retries retryable errors with backoff. On failure the fallback chain tries the same tier on the
   other provider, then the other tier.
8. **Respond**: JSON, or SSE chunks relayed as they arrive, ending in `data: [DONE]`.
9. **Finish**: one row per request and per provider call in PostgreSQL (settling the budget reservation), Prometheus
   counters, one JSON access-log line, an alert check. Prompt and answer text are never stored.

## Model routing and fallback

| `model` value | What decides the tier |
|---|---|
| `auto` | the config's `routing.default_mode`: **`strong`** in the shipped config |
| `cheap` / `strong` | the request |
| `router` | a router's P(cheap model is good enough) against a threshold. Shipped: the **rule-based** router (signals such as code, maths, multi-step wording, length); the trained classifier only once installed |
| `cascade` | cheap model first; a heuristic answer check (empty, truncated, hedged, looping) escalates to the strong model |
| a configured model name | pinned: no routing, no fallback |

**Latency class** (`X-Gateway-Latency: interactive | batch`, or the app's default) picks the provider for the tier.
In the shipped release `v4-latency`:

| | Interactive | Batch |
|---|---|---|
| strong | Groq `gpt-oss-120b` | OpenRouter `gpt-oss-120b` |
| cheap | Groq `gpt-oss-20b` | OpenRouter `gpt-oss-20b` |

This choice is evidence-based: on 377 identical questions the provider made no measurable accuracy difference (exact
McNemar p = 0.75 for 20b, 0.38 for 120b), while OpenRouter was about 2–3.6× cheaper and 2.5–6.3× slower at the
median ([provider comparison](docs/ROUTING.md#provider-comparison-measured-2026-11-15-to-2026-11-16)).

**Fallback**: same tier on the other provider, then the other tier. A model that returns a quota error is skipped for
5 minutes (or as long as the provider says). Streams fall back only before their first chunk; after it, a failure
ends the stream with an SSE error event. Full detail: [docs/ROUTING.md](docs/ROUTING.md).

## PostgreSQL and Redis

| Store | Holds | Why there |
|---|---|---|
| **PostgreSQL 16** | requests, provider calls, feedback, deployments (release log), alerts, budget reservations | durable records and transactions: a budget reservation takes a per-app advisory lock so concurrent requests on different replicas can't spend the same remaining budget |
| **Redis 7** | per-client limits, per-model provider rate limits, quota cooldowns, circuit-breaker state | fast shared counters; each operation is one Lua script using Redis server time, so it is atomic and clock-skew free across replicas |

SQLite and an in-memory state backend are used for single-process development and most tests. Schema changes are
Alembic migrations (`0001`–`0004`); the gateway refuses to start on an unmigrated PostgreSQL database.
`GATEWAY_ENV=production` refuses SQLite, the in-memory backend, anonymous access and missing keys.

## Rate limiting and resilience

| Mechanism | Behaviour |
|---|---|
| Per-client limits | requests/minute (sliding window) and requests in flight per app, shared by all replicas; shipped default 600/min and 50 in flight |
| Provider rate limits | per-model requests and tokens per minute, reserved before each call and settled with real usage |
| Daily budgets | USD per app per UTC day at list prices, reserved atomically before the call |
| Retries | bounded attempts with exponential backoff, honouring `Retry-After` |
| Circuit breakers | one per provider; `CLOSED → OPEN` after 5 consecutive qualifying failures, half-open probes after 30 s; shared through Redis |
| Fallback | across providers and tiers (above); a quota error starts a 5-minute cooldown |
| Redis outage | `fail_closed` (default): rate-limited traffic gets 429 and `/ready` 503 rather than unmetered calls; `local_share`: each replica enforces `limit ÷ replicas` |
| PostgreSQL outage | budgeted requests are refused, not silently allowed |

Measured: with Redis stopped under load, both replicas reported not-ready and refused with `Retry-After: 5`; after
restart both answered again within 1 s ([final end-to-end run](docs/OPERATIONS.md#final-end-to-end-run-2026-11-22-this-laptop-fake-providers);
policy details in [docs/OPERATIONS.md](docs/OPERATIONS.md#8-outage-and-degraded-behavior)).

## Monitoring: Prometheus and Grafana

- **`GET /metrics`** per replica: requests by app/outcome/tier, a latency histogram, tokens and cost by model,
  provider calls and their latency, fallbacks, escalations, router and shadow decisions, Redis degradation. Labels
  never contain prompt text or keys. Not published by the proxy; optionally behind a bearer token.
- **Prometheus** (`--profile monitoring`) scrapes both replicas and evaluates four alert rules: high error rate, Redis
  degraded, replica down, slow p95.
- **Grafana** is provisioned with the "LLM Gateway" dashboard and refuses to start without an admin password.
- **Logs**: one JSON line per request (id, app, outcome, model, tokens, cost, latency).
- **Streamlit dashboard**: request log, cost per day, latency, error rate, model share, canary comparison.

![Grafana dashboard (fake providers)](docs/screenshots/grafana-dashboard.png)

*Grafana during a local load test with fake providers. The "rejected" series is the per-client limit returning 429.*

## Security and API key management

- Authentication fails closed: without configured keys every request gets 401 (anonymous access is an explicit
  development opt-in, refused in production).
- Client keys can be stored as `app:sha256:<hex>` in a mounted file (`GATEWAY_API_KEYS_FILE`), so the secret store
  never holds a usable key; `scripts/make_api_key.py` generates them. An app can hold two keys during rotation.
- The admin API needs `X-Admin-Key` (hashable too) and is off when unset; the HTTPS proxy blocks `/admin/` and
  `/metrics` and puts the dashboard behind a password. Gateway ports bind to `127.0.0.1` only.
- Responses hide internal routing metadata by default; provider error text (which can name accounts) is logged, not
  returned.
- Supply chain: hash-locked dependencies, a digest-pinned base image, `pip-audit` in CI, Dependabot.

Details: [SECURITY.md](SECURITY.md), [docs/OPERATIONS.md](docs/OPERATIONS.md#security).

## Load-testing results

`scripts/load_test.py` against the Docker Compose stack on one Windows laptop (two replicas, PostgreSQL 16, Redis 7,
client on the same machine, 20% of requests streamed), measured 2026-11-22. **Fake providers**: this measures the
gateway, database and Redis, not a model provider. Raw data: [`results/load_test.json`](results/load_test.json).

| Run | Clients | Requests | Throughput | Outcome | Latency p50 / p95 / p99 |
|---|---|---|---|---|---|
| Baseline | 1 | 24 in 22 s | 1.1/s | 24 × 200 | 815 / 1,985 / 2,646 ms |
| Client limits off | 50 | 3,370 in 61 s | 55/s | 3,370 × 200, **0 failures** | 842 / 1,396 / 2,815 ms |
| Shipped client limit (600/min) | 80 | 7,278 in 63 s | 116/s | 650 × 200, 6,628 × 429 | 1,498 / 4,047 / 5,232 ms |

- Most latency is the fake provider's simulated delay: the median moved only from 815 ms (1 client) to 842 ms (50).
- With the shipped limit, the busiest 60-second window in the request log held **exactly 600** successful requests:
  the limit held across both replicas.
- `/metrics` counted every request exactly once in all three runs.

## Router evaluation results

Both models answered 1,200 benchmark questions (400 GSM8K, 800 MMLU) on OpenRouter: a real, paid collection
($0.066 in total, including an earlier 377-question comparison set). A fixed 75/25 split was set before any model saw a question. Routers were trained on 900
questions and evaluated on the **300 held-out questions** ([report](results/router_eval.md),
[v2 report](results/router_v2.md)).

| Strategy | Accuracy | Cost per 1k requests | Sent to strong | vs always-strong (exact McNemar) |
|---|---|---|---|---|
| Always cheap (`gpt-oss-20b`) | 85.0% | $0.0234 | 0% | – |
| Always strong (`gpt-oss-120b`) | 88.3% | $0.0282 | 100% | – |
| **Router v1** (MiniLM embedding + logistic regression) | **88.0%** | $0.0259 (**8% saving**) | 67% | −0.3 pts, p = 1.0 |
| Router v2 (+ lexical features) | 87.0% | $0.0249 (12% saving) | 50% | −1.3 pts, p = 0.125 |
| Random routing at v1's share | 87.2% | $0.0266 | 67% | – |

**Reading:** v1 saves 8% with no accuracy difference the test can detect, but it is close to random routing, and on
this pair the strong model costs only 17% more than the cheap one, so routing can save little. v2 looked better in
cross-validation and worse on held-out data, so it was **not promoted**. The router is therefore shipped **off**:
`models/router.json` is not installed, and the staged rollout (shadow mode, then a traffic share with a control group,
[docs/ROUTING.md](docs/ROUTING.md#router-shadow-mode-and-controlled-activation)) is ready for when real traffic exists.

![Cost vs accuracy](results/cost_vs_accuracy.png)

## What is tested, and how

| Capability | Status | Evidence |
|---|---|---|
| Chat API, streaming, auth, routing modes, fallback, breakers, budgets, client limits, metrics, key hashing, shadow mode | **Tested with mocked/fake providers** | 437 tests ([below](#running-tests)) |
| PostgreSQL storage, migrations, concurrency of budgets and releases | **Tested with real infrastructure** | 26 integration tests on PostgreSQL 16 (CI and Docker) |
| Redis shared limits, breakers, cooldowns, outage policy | **Tested with real infrastructure** | Redis 7 tests; outage drill; two-replica Compose check |
| Two replicas behind the HTTPS proxy, Prometheus, Grafana, backup drill | **Tested with real infrastructure, fake providers** | production-mode Compose run, 2026-11-22 ([validation runs](docs/OPERATIONS.md#validation-runs)) |
| Groq and OpenRouter through the gateway | **Tested with real providers** (small) | one smoke run, 4 requests incl. fallback and streaming, $0.000089 ([results](results/real_provider_smoke.json)); 3,155 router-lab answers via the model registry |
| Gemini API adapter | **Tested with mocked HTTP only** | adapter tests |
| Trained ML router, shadow mode, gradual rollout | **Implemented but not activated** | offline evaluation above; unit tests |
| Retraining from user feedback; tool calling; images | **Not implemented** | see [Limitations](#limitations-and-future-improvements) |
| Public deployment | **Not done** | [runbook](docs/RUNBOOK.md) written; nothing deployed |

## Installation and local setup

Requirements: Python 3.11+ and, for the full stack, Docker with Compose v2. **No provider keys are needed**: fake
providers answer locally with placeholder text, and nothing leaves the machine.

```bash
git clone https://github.com/thakursamir0000-star/LLM_Gateway.git
cd LLM_Gateway
python -m venv .venv
source .venv/bin/activate               # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.lock    # exact, hash-checked versions
```

## Environment configuration

Copy `.env.example` to `.env` (git-ignored) and set what you need. The minimum for a local demo:

| Variable | Local demo value | Purpose |
|---|---|---|
| `GATEWAY_FAKE_PROVIDERS` | `1` | fake models instead of real providers (no keys, no cost) |
| `GATEWAY_API_KEYS` | `demo:<any long random string>` | client keys, `app:key` or `app:sha256:<hex>`; required (auth fails closed) |

For real providers set `GROQ_API_KEY`, `OPENROUTER_API_KEY` and/or `GEMINI_API_KEY` and leave
`GATEWAY_FAKE_PROVIDERS` unset (every request is then a paid call). Everything else (database URL, Redis, admin key,
metrics token, log format, outage policy) is listed in
[docs/OPERATIONS.md](docs/OPERATIONS.md#configuration); models, prices, limits and releases are in
[`config/gateway.yaml`](config/gateway.yaml), validated at start-up.

## Running the application

Single process (SQLite request log, in-memory state):

```bash
export GATEWAY_FAKE_PROVIDERS=1 GATEWAY_API_KEYS="demo:local-demo-key"   # PowerShell: $env:GATEWAY_FAKE_PROVIDERS="1"; $env:GATEWAY_API_KEYS="demo:local-demo-key"
python -m gateway                        # http://127.0.0.1:8000, Swagger UI at /docs
streamlit run dashboard/app.py           # optional: request-log dashboard
```

## API examples

```bash
# Health and readiness
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready

# A chat completion (model: auto | cheap | strong | router | cascade | a model name)
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer local-demo-key" -H "Content-Type: application/json" \
  -d '{"model": "router", "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'

# Streaming (server-sent events), on the cheaper batch provider
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer local-demo-key" -H "Content-Type: application/json" -H "X-Gateway-Latency: batch" \
  -d '{"model": "cheap", "stream": true, "messages": [{"role": "user", "content": "Name a primary colour."}]}'

# Thumbs up for an answer (the id is the response's "id")
curl -s http://127.0.0.1:8000/v1/feedback \
  -H "Authorization: Bearer local-demo-key" -H "Content-Type: application/json" \
  -d '{"request_id": "chatcmpl-...", "score": "up"}'

# Routing modes and configured models
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-demo-key"
```

Captured outputs of these, with screenshots of Swagger UI, Prometheus, Grafana and the dashboard:
[docs/DEMO.md](docs/DEMO.md).

## Running tests

```bash
pytest            # unit and SQLite tests; PostgreSQL and Redis tests skip without those servers
ruff check .
```

With real PostgreSQL and Redis (each PostgreSQL test creates and drops its own database):

```bash
docker run -d --name gw-pg -p 127.0.0.1:55432:5432 -e POSTGRES_USER=gw -e POSTGRES_PASSWORD=gwpass -e POSTGRES_DB=gateway_test postgres:16-alpine
docker run -d --name gw-redis -p 127.0.0.1:56379:6379 redis:7-alpine
TEST_POSTGRES_URL=postgresql+psycopg://gw:gwpass@127.0.0.1:55432/gateway_test pytest
```

**Results** (2026-11-23): **437 passed, 0 skipped** in a Linux container (Python 3.12) against PostgreSQL 16 and
Redis 7; on Windows without those servers, 399 passed and 38 skipped (the PostgreSQL and Redis tests). Ruff clean.
CI runs the same suite on Python 3.11 and 3.12 with PostgreSQL and Redis services, and builds and smoke-tests the
two-replica Compose stack. The tests never call a real provider.

## Docker setup

```bash
# .env: GATEWAY_FAKE_PROVIDERS=1 and GATEWAY_API_KEYS=demo:<key>
docker compose up -d --build                    # PostgreSQL, Redis, migrations, 2 gateway replicas, dashboard
docker compose --profile monitoring up -d       # + Prometheus :9090 and Grafana :3000 (needs GRAFANA_ADMIN_PASSWORD)
python scripts/make_proxy_secrets.py            # development TLS certificate + dashboard password
docker compose --profile proxy up -d            # + nginx HTTPS proxy on :443 (port 80 redirects)
```

| Service | Address (host) |
|---|---|
| gateway, gateway-replica | `127.0.0.1:8000`, `127.0.0.1:8001` |
| dashboard | `127.0.0.1:8501/dashboard/` |
| PostgreSQL, Redis | `127.0.0.1:54320`, `127.0.0.1:63790` |
| Prometheus, Grafana | `127.0.0.1:9090`, `127.0.0.1:3000` |
| proxy | `:443` (all interfaces) |

`python scripts/multi_instance_check.py` builds and checks the whole two-replica stack (shared budget, rate limit,
breaker, HTTPS, streaming through the proxy, restart) and removes it afterwards.

## Limitations and future improvements

- **Not deployed**: no public host, real certificate or real traffic. [docs/RUNBOOK.md](docs/RUNBOOK.md) is the plan.
- **Limited real-provider evidence**: one 4-request smoke test through the HTTP API plus the router lab's answers;
  real providers under load and Gemini are untested. Costs are list-price estimates.
- **The ML router is not active**, and its benefit on this model pair is small and not statistically significant;
  results are on benchmark questions, not real traffic.
- **No retraining from feedback**: the request log deliberately stores no prompt text, so ratings can't be joined to
  questions; this would need opt-in prompt capture.
- **Text chat only**: tool calling, images and `n > 1` are rejected with a clear 400.
- **Streams don't fail over after the first chunk**, and cascade mode doesn't check streamed answers.
- **Operations left to the deployer**: no Alertmanager routing, no scheduler for retention/backups/drills.
- **Synchronous Redis client** run in worker threads (off the event loop); throughput is bounded by the thread pool.
- A malformed unauthenticated request gets its 400 validation error before the 401 (FastAPI validates first).

## Documentation

| Document | Contents |
|---|---|
| [docs/DEMO.md](docs/DEMO.md) | walkthrough with captured outputs and screenshots |
| [docs/ROUTING.md](docs/ROUTING.md) | routing, the router lab, full evaluation and provider comparison |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | streaming, breakers, releases, budgets, metrics, configuration, multi-replica operation, troubleshooting |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | deploy, upgrade, incidents, key rotation |
| [docs/RESUME_PROJECT.md](docs/RESUME_PROJECT.md) | project summary for CVs and interviews |
| [SECURITY.md](SECURITY.md) | keys, secrets, supply-chain policy |
| [docs/architecture.mmd](docs/architecture.mmd) | editable architecture diagram |
