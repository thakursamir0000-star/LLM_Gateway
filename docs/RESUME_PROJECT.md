# LLM Gateway: resume and interview notes

Every number here comes from the repository's results files or recorded test runs. Context matters for each one,
so it is stated next to it.

## Two-line description

An OpenAI-compatible LLM gateway that routes each request across Groq, OpenRouter and Gemini, with fallback,
circuit breakers, Redis-backed distributed rate limits and PostgreSQL cost accounting, observable through Prometheus
and Grafana. It includes an ML router trained and evaluated on 1,200 benchmark questions, held behind a staged rollout.

## Technical summary

A FastAPI service that existing OpenAI clients use by changing one URL. Each request is authenticated with hashed
API keys, checked against per-client limits shared by all replicas through atomic Redis Lua scripts, and charged
against a daily budget reserved in a PostgreSQL transaction. The service then chooses a model tier (by mode, rules or
a trained classifier) and a provider (by latency class), and calls it through a registry with retries, quota
cooldowns, circuit breakers and cross-provider fallback. Answers return as JSON or relayed SSE streams; every request
is logged without its text and counted in Prometheus. Two stateless replicas run behind an nginx HTTPS proxy in
Docker Compose with Prometheus and Grafana. An offline lab collected 2,400 answers from two real models, trained a
sentence-embedding router and evaluated it on held-out questions with significance tests. The router ships disabled,
with shadow mode and a control-group rollout ready for real traffic.

## Resume bullets

- Built an OpenAI-compatible LLM gateway (FastAPI, httpx) routing requests across Groq, OpenRouter and Gemini by cost
  tier and latency class, with SSE streaming, retries, cross-provider fallback and per-provider circuit breakers.
- Made the service horizontally scalable: per-client and per-model rate limits, quota cooldowns and breaker state
  shared through atomic Redis Lua scripts, plus PostgreSQL budget reservations under advisory locks. The per-client
  limit held at exactly 600 requests/min across two replicas under an 80-client load test.
- Load-tested a two-replica Docker Compose stack (PostgreSQL 16, Redis 7, mock providers): 3,370 requests from
  50 concurrent clients with zero failures and a median latency of ~842 ms, versus 815 ms for a single client.
- Trained and evaluated an ML router (MiniLM embeddings + logistic regression) on 1,200 GSM8K/MMLU questions answered
  by two real models. It reached 88.0% on 300 held-out questions vs 88.3% always-strong, at 8% lower cost
  (McNemar p = 1.0). A second version scored 87.0%; I did not promote it, and shipped the router disabled behind
  shadow mode and a control-group rollout.
- Added production observability and operations: Prometheus metrics and alert rules, a provisioned Grafana dashboard,
  JSON access logs without prompt text, backup/restore with an automated restore drill, and a deployment runbook.
- Hardened security: fail-closed auth with SHA-256-hashed, rotatable API keys, an HTTPS proxy (TLS 1.2/1.3, security
  headers), unpublished admin and metrics endpoints, and hash-locked dependencies. 437 automated tests pass against
  real PostgreSQL 16 and Redis 7.

## Technology stack

Python, FastAPI, Pydantic, httpx, asyncio · PostgreSQL 16, SQLAlchemy 2 Core, Alembic · Redis 7 (Lua) · Docker,
Docker Compose, nginx · Prometheus, Grafana, Streamlit · sentence-transformers, NumPy · pytest, Ruff, GitHub Actions,
uv lock files, pip-audit.

## Verified numbers (with their context)

| Claim | Value | Context |
|---|---|---|
| Load test, no client limits | 3,370 requests, 50 clients, 0 failures, p50 842 ms (p95 1,396 ms) | 61 s, Docker Compose on one laptop, 2 replicas, **mock providers** (latency is mostly their simulated delay) |
| Single-client baseline | p50 815 ms | same stack |
| Client limit under load | busiest 60 s window: exactly 600 accepted | 80 clients, shipped 600/min limit, 2 replicas |
| Router v1 | 88.0% accuracy, 8% cheaper than always-strong (88.3%) | 300 held-out benchmark questions, real OpenRouter answers; difference not significant (p = 1.0) |
| Router v2 | 87.0%, 12% cheaper | same split; not promoted (p = 0.125) |
| Benchmark collection | 1,200 questions × 2 models, $0.066 | real OpenRouter calls (total includes an earlier 377-question set) |
| Tests | 437 passed, 0 skipped | Linux container, Python 3.12, real PostgreSQL 16 and Redis 7 |
| Redis outage recovery | both replicas ready within 1 s of Redis returning | local production-mode Compose run |

Do **not** claim: production deployment, real users or traffic, an active ML router, real-provider load testing,
or a significant accuracy-preserving saving.

## 60-second explanation

"I built an LLM gateway, a service that sits between applications and LLM providers and speaks the OpenAI API, so
an app switches to it by changing one URL. For every request it decides which model and which provider to use,
retries and falls back to another provider if one fails, and enforces limits and daily budgets. Because it runs as
several replicas, the shared state (rate limits, circuit breakers, budgets) lives in Redis and PostgreSQL with atomic
operations, so the limits hold across replicas; I verified that under load. It's observable through Prometheus and
Grafana, and I wrote backup, restore and a runbook for it. I also tried to cut cost with an ML router that sends easy
questions to a cheaper model. On 300 held-out questions it kept accuracy within the noise and saved 8%, but that
saving wasn't significant and it was close to random routing. So I shipped it disabled, behind a shadow-mode rollout,
rather than overclaim."

## Two-minute technical explanation

"The entry point is FastAPI. A request first passes a body-size limit and schema validation, then authentication:
the bearer key is compared in constant time against configured keys, which can be stored as SHA-256 hashes from a
mounted secret file, and it identifies the client app. Next comes a per-client limit: one Redis Lua script trims a
sorted-set sliding window, checks requests-per-minute and requests-in-flight, and records the request atomically,
using Redis server time so replicas with skewed clocks agree. A stream keeps its in-flight slot until it ends; a
lease expiry frees slots a crashed replica never released.

Then the budget: the request reserves its maximum possible cost (prompt plus max tokens at the dearest model it
might use) in one transaction. On PostgreSQL that transaction takes a per-app advisory lock, so two replicas can't
spend the same remaining budget; the reservation is settled with the real cost when the request is logged.

Routing has two parts. The tier, cheap or strong, comes from the requested mode, a rules router, or the trained
classifier. The latency class picks the provider: Groq is fast, OpenRouter cheaper and slower. I measured that the
host doesn't change accuracy, so that trade-off is safe. The model registry waits on the model's shared rate limit,
checks the provider's circuit breaker and quota cooldown, and retries with backoff. If the call still fails, the
fallback chain tries the same tier on the other provider, then the other tier. Streaming relays SSE chunks as they
arrive and only fails over before the first chunk.

For a Redis outage I chose fail-closed by default: rate-limited traffic gets a 429 with Retry-After instead of
unmetered provider calls. The alternative policy gives each replica its share of the limit, never the full limit.

The ML part was offline: both models answered 1,200 GSM8K and MMLU questions, split 900/300 before any model saw
them. The router embeds the prompt with MiniLM and runs a NumPy logistic regression; I chose its threshold on
out-of-fold training scores only, and compared it to always-strong with an exact McNemar test. It saved 8% at
−0.3 points, not significant. A second version with lexical features did worse on held-out data, so I kept v1 and
built shadow mode and a hashed traffic split with a control group, so it can be validated on real traffic before it
routes anything."

## Likely interview questions

1. **Why an OpenAI-compatible API?** Existing SDKs and apps work unchanged (only `base_url` changes), and the error
   format matches what clients already handle. A test drives the gateway with the official OpenAI SDK.
2. **How do rate limits stay correct across replicas?** Each check-and-record is one Redis Lua script, so it is
   atomic, and it uses Redis's clock rather than each replica's. Verified: exactly 600 accepted in the busiest
   minute with two replicas and 80 clients.
3. **What happens if Redis goes down?** `/ready` returns 503 and, by default, rate-limited requests get 429 with
   Retry-After rather than unmetered calls. Optionally each replica enforces `limit ÷ replicas` locally. Measured
   recovery: under 1 s after Redis returned.
4. **How do you stop concurrent requests overspending a budget?** Reserve the worst-case cost before calling, under
   a per-app PostgreSQL advisory lock (SQLite: `BEGIN IMMEDIATE`), then settle the real cost when the request is
   logged. A reservation from a crashed process expires.
5. **How do circuit breakers interact with retries and fallback?** One breaker per provider. Only timeouts, 5xx and
   overload count, and all retries of one call count once. When the breaker is open, calls fail fast, so the
   fallback chain moves to the other provider. Half-open probes use leases so a crashed probe can't block recovery.
6. **Why not fail over in the middle of a stream?** The client has already received tokens. Switching models would
   splice two different answers, so after the first chunk a failure ends the stream with an explicit error event.
7. **How did you evaluate the router, and why isn't it on?** A fixed held-out split, a threshold chosen only on
   training data, and comparison with always-cheap, always-strong and random routing at the same share, with exact
   McNemar tests. It saved 8% without a detectable accuracy loss, but it was close to random routing and not
   significant, so I'd validate it in shadow mode on real traffic first.
8. **What does shadow mode measure, and what can't it?** It logs the tier the classifier would have chosen, so you
   get agreement, the share it would send cheap, and the estimated saving. It can't tell whether those cheap answers
   would have been good enough; that needs feedback or sampled review, which is why the next stage has a control
   group.
9. **How do you handle secrets and API keys?** Keys can be stored as SHA-256 hashes in a mounted file and are compared
   in constant time. An app holds two keys during rotation. Admin and metrics endpoints aren't published by the proxy,
   and logs and metric labels never contain prompt text or keys.
10. **What would you do next for production?** Deploy following the runbook with managed PostgreSQL and Redis, add
    Alertmanager routing and scheduled backups and drills, load-test with real providers, then run the router in
    shadow mode and a 5% canary before any wider rollout.
