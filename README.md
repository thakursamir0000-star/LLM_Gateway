# LLM Gateway

An OpenAI-compatible service that sits between your apps and LLM providers (Groq, OpenRouter, the Gemini API). It
chooses the model for each request (a cheap or a strong tier, a fast or a cheap provider), falls back across providers
when one fails, enforces per-app budgets and shared rate limits, and records every call: tokens, cost, latency,
errors, route decisions and user feedback.

**What decides the model today:** the shipped config routes `model="auto"` to the strong tier, and the latency class
picks the provider. A trained classifier router exists and has been evaluated offline (8% cheaper than always-strong
on held-out benchmark questions, a difference in quality the test can't distinguish from zero; see
[Results](#results)), but it is **not installed or active**: `model="router"` uses the rule-based router.

An app switches to it by changing one URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="my-secret-key")  # a key from GATEWAY_API_KEYS
reply = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "What is 17 * 23?"}])
print(reply.choices[0].message.content)
# With GATEWAY_EXPOSE_METADATA=1 the response also has a "gateway" object: model, route, cost.
```

**Status:** Phases 1–3 (gateway, routing, monitoring, releases, security, streaming, circuit breakers, PostgreSQL,
Redis, Alembic, multiple replicas) are built and tested with fake providers and mocked HTTP. Phase 4 (verification) is
in progress: PostgreSQL 16 integration tests, Redis outage drills and a two-replica Docker validation pass, and a
real-provider smoke test through the gateway's HTTP API passed on 2026-11-22 (Groq, OpenRouter, a forced fallback and
streaming; $0.000089; [`results/real_provider_smoke.json`](results/real_provider_smoke.json)). Nothing is deployed.
See [Validation status](#validation-status).

## Validation status

| Capability | Implemented | Unit tested | Integration tested | Real-provider tested | Deployed |
|---|---|---|---|---|---|
| OpenAI-compatible chat API (JSON and SSE streaming) | yes | yes | yes: two replicas in Docker, SSE through nginx (fake providers) | yes: smoke test, JSON and streaming (one run each) | no |
| Groq and OpenRouter adapters | yes | yes (mocked HTTP) | – | yes: smoke test through the HTTP API, plus the router lab's 3,155 answers through the model registry | no |
| Gemini API adapter | yes | yes (mocked HTTP) | – | no | no |
| Retries, fallback, quota cooldowns, circuit breakers | yes | yes | yes: shared across two replicas through Redis | fallback only: Groq 401 → OpenRouter in the smoke test | no |
| Rules router, cascade | yes | yes | – | no | no |
| Trained classifier router (v1; v2 evaluated and not promoted) | yes, **not active** | yes | offline held-out evaluation | on real benchmark answers, offline | no |
| Router shadow mode and controlled activation (classifier share, control group) | yes, **off** | yes | – | – | no |
| Per-app budgets with atomic reservations | yes | yes | yes: PostgreSQL 16, two replicas | no | no |
| Request log, metrics, alerts | yes | yes | yes: PostgreSQL 16 | no | no |
| Releases, canary, rollback | yes | yes | yes: PostgreSQL 16, concurrent changes | – | no |
| PostgreSQL storage and Alembic migrations | yes | – | yes: 25 tests on PostgreSQL 16 in CI | – | no |
| Redis shared state and outage policy | yes | yes | yes: real Redis 7, outage drill | – | no |
| Authentication, admin key, body limits, CORS, generic upstream errors | yes | yes | yes: 401 without a key in the Docker smoke test | – | no |
| Locked, hash-checked dependencies; digest-pinned base image; vulnerability audit | yes | – | yes: CI installs the locks, audit workflow on every push and weekly | – | – |
| HTTPS proxy (TLS 1.2/1.3, redirect, security headers), dashboard password, admin API not published | yes | yes (password hashing) | yes: two-replica Docker check | – | no |
| Per-client request limits (requests per minute, requests in flight) | yes | yes | yes: real Redis 7, two backends sharing one allowance | – | no |
| Prometheus metrics, JSON access logs, Grafana dashboard, alert rules | yes | yes | – | – | no |
| Request-log retention, backup and restore (with a restore drill) | yes | yes (SQLite) | yes: retention on PostgreSQL 16, backup drill on the Compose database | – | no |

"Integration tested" means against real PostgreSQL, Redis or Docker Compose (`tests/test_postgres_integration.py`,
`tests/test_redis_*.py`, `scripts/redis_outage_drill.py`, `scripts/multi_instance_check.py`).

### Final end-to-end run (2026-11-22, this laptop, fake providers)

The merged code was run as a production-style stack: Docker Compose with `GATEWAY_ENV=production` (start-up guards
on), two replicas, PostgreSQL 16, Redis 7, the HTTPS proxy with a development certificate, Prometheus and Grafana;
client keys stored hashed in a mounted keys file, a hashed admin key, a metrics token given to Prometheus as
[docs/RUNBOOK.md](docs/RUNBOOK.md) describes. Fake providers, so no paid calls; nothing was published beyond
this machine, and the stack was removed afterwards.

| Check | Result |
|---|---|
| `scripts/multi_instance_check.py` (shared budget, rate limit, cooldown and breaker, HTTPS, SSE through the proxy, replica restart, bounded connections) | all 8 checks passed |
| Start-up in production mode; `/health` through HTTPS | every service healthy; `"auth": "api_keys"`, rule-based router (no classifier installed) |
| Chat through the proxy with a hashed key; wrong key | 6 × 200; 401 |
| Streaming through the proxy | 7 `data:` events ending in `[DONE]` |
| Admin API: through the proxy; on loopback with the hashed admin key; wrong key | 403; 200; 401 |
| `/metrics`: through the proxy; loopback without the token; with it | 403; 401; 200 |
| Dashboard without / with its password | 401; 200 |
| Client limit, `demo` app (60 per minute), 65 requests | 60 × 200, 5 × 429 |
| Prometheus | both replicas `up` with the token; counted 68 ok and 5 rejected, matching what was sent; no alerts firing |
| Grafana | "LLM Gateway" dashboard provisioned; admin/admin refused |
| JSON access logs | one line per request; none contains prompt text |
| Redis stopped under load (`fail_closed`) | both replicas `/ready` 503; requests 429 `client_rate_limited` with `Retry-After: 5`; Prometheus recorded `gateway_state_degraded` = 1 on both |
| Redis restarted | both replicas ready and answering within 1 s |
| Backup drill (`scripts/db_backup.py drill`) | 76 requests and 69 calls restored identically, Alembic revision `0004_shadow_routing` |
| Retention dry run | ran on PostgreSQL; nothing older than 90 days |

Not covered: real providers under load, a public host and real certificate, Gemini, the trained router (kept off).

## What it does

- **OpenAI-compatible API**: `POST /v1/chat/completions` (with `stream: true`), `GET /v1/models`, plus
  `POST /v1/feedback` for thumbs up/down. Errors use OpenAI's error format.
- **Providers**: Groq (`openai/gpt-oss-20b` as the cheap model, `openai/gpt-oss-120b` as the strong one), OpenRouter
  (the same two models, pinned to one full-precision host, with the real per-call cost it reports) and the Gemini API
  (Gemini and Gemma models). Calls are paced under each model's limits and retried within bounds; provider errors are
  stripped of account ids and keys before they are logged or returned.
- **Routing modes**, chosen per request with the `model` field:
  `cheap`, `strong`, `router` (the rule-based router; the trained classifier only once it is installed) and `cascade`
  (cheap first, escalate when the answer looks unsure). `auto` uses the configured default (`strong`).
- **Latency-aware provider routing**: `interactive` requests go to fast models (Groq), `batch` requests to cheaper,
  slower ones (OpenRouter), with fallback across providers and a cooldown for a model that is out of quota.
- **Audited cost/quality controls**: separate classifier and rules/cascade defaults in `[0, 1]`. Higher sends more
  traffic to the strong model, and the classifier default must match its trained artifact.
- **Monitoring**: every request is logged to SQLite (development) or PostgreSQL (model, route decision, tokens, cost,
  latency, error; never the prompt or the answer). A Streamlit dashboard shows requests, cost per day, p50/p95 latency, error rate and the share
  per model. Alerts fire when the error rate or daily cost passes a limit.
- **Release management**: model and prompt versions live in the config as releases; activate, roll back, or canary a
  share of traffic to a new release. Per-app daily budgets with atomic reservations.
- **Several replicas**: PostgreSQL holds the request log, budgets and release state; Redis holds rate limits, quota
  cooldowns and circuit breakers, so every replica enforces the same limits. Alembic manages the schema.
- **Router lab**: scripts to collect benchmark labels (GSM8K, MMLU) from both models, train the router, and compare it
  with always-cheap, always-strong and random routing on held-out questions.

## Architecture

```mermaid
flowchart LR
    App["App<br/>(OpenAI SDK)"] -->|"POST /v1/chat/completions"| API["FastAPI<br/>gateway/app.py"]
    API --> Svc["Gateway service<br/>gateway/service.py"]
    Svc --> Budget["Budget check<br/>per app, per UTC day"]
    Svc --> Rel["Release + canary<br/>models + prompt version"]
    Svc --> Router["Router<br/>rules | classifier"]
    Svc --> Reg["Model registry<br/>pacing, retries"]
    Reg --> Groq["Groq<br/>gpt-oss-20b / 120b"]
    Reg --> OpenRouter["OpenRouter<br/>gpt-oss-20b / 120b<br/>(pinned host)"]
    Reg --> Gemini["Gemini API<br/>Gemini / Gemma"]
    Svc --> Check["Cascade check<br/>escalate if unsure"]
    Svc --> DB[("SQLite (dev) or PostgreSQL<br/>requests, calls, feedback,<br/>deployments, alerts, budgets")]
    Reg --> Redis[("Redis (multi-replica)<br/>rate limits, cooldowns,<br/>circuit breakers")]
    Svc --> Alerts["Alerts<br/>log + webhook"]
    DB --> Dash["Streamlit dashboard"]
    Lab["Router lab scripts"] -->|"answers.jsonl"| Train["train_router.py"] -->|"models/router.json"| Router
```

What happens to one request (`gateway/service.py`):

1. **Validate**: text chat, `n=1`, no tools (not supported yet).
2. **Mode** from `model`: `auto`, `router`, `cascade`, `cheap`, `strong`, or a configured model (pinned, no routing).
3. **Budget**: at the app's daily limit the request gets HTTP 429; past the soft limit (80% by default) routed
   requests go to the cheap model.
4. **Release**: the active release, or the canary for its share of traffic (sticky per `user` when it is set). The
   release's prompt version is put before the app's messages.
5. **Latency class**: `interactive` or `batch`, from the `X-Gateway-Latency` header or the app's default; it picks
   which provider's models serve the request (see [Routing](#routing)).
6. **Route**: the tier is fixed by the mode (`strong` by default), or the router scores P(cheap model is good enough)
   and `score >= threshold` goes cheap, otherwise strong.
7. **Call** through the registry: default params, pacing under per-minute limits, bounded retries. If the model fails,
   the same tier on the other provider is tried, then the other tier (`fallback_on_error`).
8. **Cascade**: the cheap answer gets a heuristic confidence (empty, truncated, blocked, looping or hedged answers
   score low); below the threshold the request is escalated to the strong model of the same class.
9. **Log** one row per request (including its latency class) and one per provider call; check alerts (at most once a
   minute).

The response is a normal `chat.completion` with an `x-gateway-request-id` header. With `GATEWAY_EXPOSE_METADATA=1`
it also carries a `gateway` object (request id, release, canary, mode, router, score, threshold, latency class, tier,
model, escalation, fallback, cost, router time, reason) and more `x-gateway-*` headers; they are hidden by default.
`usage` counts every provider call the request made (a cascade escalation pays for two).

## Quick start

Python 3.11+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements-dev.lock   # exact, hash-checked versions (see SECURITY.md)
```

**Without API keys** (fake models, nothing leaves your machine; anonymous access so the demo script's `X-App-Id`
header identifies the app, which is refused when `GATEWAY_ENV=production`):

```bash
GATEWAY_FAKE_PROVIDERS=1 GATEWAY_ALLOW_ANONYMOUS=1 python -m gateway
```

```bash
python scripts/send_demo_traffic.py --n 300
```

```bash
streamlit run dashboard/app.py
```

**With real models**: copy `.env.example` to `.env`, set `GROQ_API_KEY` (plus `OPENROUTER_API_KEY` for the
`or-*` models and `GEMINI_API_KEY` for Gemini/Gemma), and configure `GATEWAY_API_KEYS="my-app:my-secret-key"`, then `python -m gateway`. The gateway listens on `http://127.0.0.1:8000` (`GATEWAY_HOST`, `GATEWAY_PORT`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Explain step by step why the sky is blue."}]}'
```

**Docker** (PostgreSQL, Redis, migrations, two gateway replicas on :8000 and :8001, dashboard on the host's loopback
interface; see [Docker & Multi-Instance Deployment](#6-docker--multi-instance-deployment)):

```bash
docker compose up --build
```

`WITH_EMBEDDINGS=true docker compose up --build` adds sentence-transformers (CPU torch) so the trained router can run
in the container; without it the gateway uses the rule-based router.

The dashboard shows app IDs, model and release information, costs and error details, and has no login of its own.
Docker Compose publishes it on `http://127.0.0.1:8501/dashboard/` only. Remote access goes through the HTTPS proxy
(`--profile proxy`), which serves it at `https://<host>/dashboard/` behind a password; or keep the loopback binding and
use an SSH tunnel (`ssh -L 8501:127.0.0.1:8501 user@gateway-host`).

**HTTPS proxy** (`docker compose --profile proxy up -d`): nginx terminates TLS 1.2/1.3 on port 443 in front of both
replicas and the dashboard; port 80 only redirects to HTTPS. It reads `server.crt`, `server.key` and
`dashboard.htpasswd` from `GATEWAY_TLS_DIR` (default `config/tls/`, git-ignored). For local use create them with:

```bash
python scripts/make_proxy_secrets.py    # self-signed localhost certificate; set DASHBOARD_PASSWORD or one is generated
```

In production put a certificate from a real CA in that directory. The proxy adds HSTS, `X-Content-Type-Options`,
`X-Frame-Options`, `Referrer-Policy` and `frame-ancestors 'none'`, refuses `/admin/` (use the gateway's loopback port
with `X-Admin-Key`, or the CLI) and asks for the dashboard password. The gateway ports themselves are published on
`127.0.0.1` only.

## Routing

Two choices are made per request: the **tier** (cheap or strong model) and the **latency class** (which provider
serves it). The shipped config routes by latency class: on the measured questions the strong model was the better buy
(see [Provider comparison](#provider-comparison-measured-2026-11-15)), and the bigger trade-off was speed against cost
across providers.

**Latency classes** (`interactive` | `batch`): a request sets its class with the `X-Gateway-Latency` header; otherwise
the app's default from `latency.apps` applies, otherwise `latency.default` (`interactive`). A release can name other
models for batch traffic; the active release `v4-latency` does:

| | Interactive (fast) | Batch (cheaper, slower) |
|---|---|---|
| strong | Groq `gpt-oss-120b` | OpenRouter `gpt-oss-120b` (DeepInfra bf16) |
| cheap | Groq `gpt-oss-20b` | OpenRouter `gpt-oss-20b` (DeepInfra bf16) |

Releases without a `batch` block (such as `v1`, Groq only) serve both classes with the same models. Pinned models
ignore the class.

**Fallback across providers**: when a model fails, the gateway tries the same tier on the other class's model (the
other provider), then both models in the other tier. Duplicate aliases are removed for releases that use the same
models for both classes. A model that hits a quota error (Groq's daily limit, OpenRouter out of credits) is skipped for
`retries.quota_cooldown_s` (300 s, or longer if the provider says so), so later requests go straight to the next
candidate. Interactive traffic that falls back to OpenRouter is slower, but answered.

| `model` value | What happens |
|---|---|
| `auto` | the configured `routing.default_mode` (`strong` in the shipped config) |
| `router` | the router's score against the threshold picks cheap or strong |
| `cascade` | cheap model first; escalate to strong when the answer check is below the threshold |
| `cheap` / `strong` | always that tier of the active (or canary) release |
| `gpt-oss-120b`, `openai/gpt-oss-120b`, ... | a configured model, pinned: no routing, no fallback |

**Thresholds**: `routing.classifier_threshold` is the classifier's authoritative runtime default. It must match the
training-selected threshold recorded in the artifact or the gateway refuses to load that classifier.
`routing.threshold` is the default for the rules router and cascade. Every router outputs P(cheap model is good
enough); requests with a score below its threshold go to the strong model. In cascade mode, cheap answers whose
confidence is below `routing.threshold` are escalated. `X-Gateway-Threshold` is an explicit, logged per-request
override. `0` = always cheap, `1` = (almost) always strong.

**Rule-based router** (`gateway/routing/rules.py`): hand-weighted signals of a hard request (long context, code,
maths, multi-step reasoning words, several questions, a long conversation). It is the baseline the trained router has
to beat.

**Classifier router** (`gateway/routing/classifier.py`): a sentence embedding of the last user message
(`sentence-transformers/all-MiniLM-L6-v2` by default) followed by logistic regression that predicts whether the cheap
model is good enough. The regression is fitted in numpy with Newton's method (a test checks it matches scikit-learn),
so serving it needs no scikit-learn. It is loaded from `models/router.json`; until that file exists, the gateway falls
back to the rules and says so in `/health`. The artifact names the cheap and strong aliases it was trained for. A
classifier request is allowed only when those aliases match the selected release and latency class; a mismatch uses
the strong model and records the reason.

The prepared OpenRouter canary keeps `routing.default_mode: strong`. After the audited artifact is deliberately
installed, only requests that explicitly use `model="router"` with `X-Gateway-Latency: batch` use it. Interactive
router requests stay on the strong model because the artifact is trained for the OpenRouter batch aliases.

**Cascade check** (`gateway/routing/cascade.py`): no extra model call, but it only catches answers that *look* wrong;
a confident wrong answer passes. The evaluation measures how much that matters.

## Streaming and circuit breakers

### Streaming

`stream: true` streams the answer token by token as server-sent events (`text/event-stream`): each provider chunk is
relayed as it arrives.

- Every event is an OpenAI `chat.completion.chunk` (`data: {...}`); the first one carries `delta.role: "assistant"`.
  With `stream_options: {"include_usage": true}` a last chunk with empty `choices` carries `usage` (and the `gateway`
  object when `GATEWAY_EXPOSE_METADATA=1`). The stream ends with `data: [DONE]`.
- Providers stream natively: Groq and OpenRouter through their OpenAI-compatible SSE (`stream: true`, usage
  requested; OpenRouter also reports the cost), the Gemini API through `streamGenerateContent?alt=sse`. The fake
  providers stream too.
- Until the first chunk arrives the request is handled like any other: cooldowns, rate limits and circuit breakers
  apply, retryable errors are retried with backoff, and the fallback chain can move to another model or provider. A
  failure at this point returns a normal HTTP error in OpenAI's format, because no response has started.
- Once the first chunk has been sent, the gateway never retries or switches provider. If the upstream stream fails
  after that, the client gets an SSE error event, `{"error": {"message": "upstream stream failed", "type":
  "upstream_error", "code": "stream_failed"}}`, then `data: [DONE]`; the HTTP status stays 200 and the provider's
  error text is not forwarded. A client disconnect closes the upstream stream and is logged as cancelled, with no
  error event.
- Tokens and cost come from the provider's usage when it reports them, otherwise from estimates (about 4 characters
  per token, at the config's list prices). The request is logged when the stream ends.
- In `cascade` mode a streamed answer comes from the cheap tier and is not checked or escalated.

### Circuit breakers

Each provider (`groq`, `openrouter`, `gemini`) has one circuit breaker (`gateway/circuit.py`), shared by its models.

| State | Behaviour |
|---|---|
| `CLOSED` | Calls pass. Each qualifying failure adds to a count of consecutive failures and a success resets it; at `failure_threshold` the breaker opens. |
| `OPEN` | Calls to the provider are refused at once with a 503 `overloaded` error and no upstream call, so the fallback chain moves on to the next model; with no model left the client gets HTTP 503. |
| `HALF_OPEN` | After `recovery_timeout_s`, up to `half_open_probes` requests at a time go through as probes while the rest are still refused. After `half_open_probes` successful probes the breaker closes; a qualifying probe failure opens it again and restarts the timer. |

- **Counts as a failure:** timeouts, connection and network errors, overloaded or unavailable responses, HTTP 5xx.
- **Does not count:** 4xx client errors (bad request, auth), rate limits and daily-quota errors (the quota cooldown
  handles those), and cancelled requests. Such an error during a probe frees the probe slot without penalty.
- All retries of one call count as at most one failure. A stream that fails after its first chunk still counts.

```yaml
circuit_breaker:
  failure_threshold: 5      # consecutive qualifying failures that open the breaker (default 5)
  recovery_timeout_s: 30.0  # seconds in OPEN before probes are allowed (default 30)
  half_open_probes: 1       # probes allowed at a time, and successful probes needed to close (default 1)
```

Breaker state lives in each gateway process's memory: it is not shown in `/health` and is not shared between
instances.

## Releases, canary and rollback

A release pins the cheap model, the strong model and the prompt version (`config/gateway.yaml`):

```yaml
releases:
  v1:       {cheap: gpt-oss-20b, strong: gpt-oss-120b, prompt: passthrough@1}
  v2-gemma: {cheap: gemma-4-31b, strong: gpt-oss-120b, prompt: concise@1}
deployment:
  active: v1
  canary: null
```

Changes are appended to a deployment log in the database, from the CLI or the admin API:

```bash
python -m gateway.manage canary v2-gemma 0.1 --note "try Gemma as the cheap model"
```

```bash
python -m gateway.manage activate v2-gemma
```

```bash
python -m gateway.manage rollback
```

```bash
python -m gateway.manage history
```

Rollback re-activates the release that was active before the current one and stops any canary. The dashboard compares
stable and canary traffic (requests, error rate, p50 latency, cost, thumbs up/down). The admin endpoints
(`/admin/status`, `/admin/activate`, `/admin/rollback`, `/admin/canary`, `/admin/deployments`, `/admin/alerts`) are off
unless `GATEWAY_ADMIN_KEY` is set, and then need it in the `X-Admin-Key` header.

The config's `deployment.active` is only the starting state: once a database has a deployment log, the log wins. To
move an existing install to the latency-aware release (or back to Groq only):

```bash
python -m gateway.manage activate v4-latency
```

```bash
python -m gateway.manage activate v1
```

## Security and Client Migration

Phase 1 security hardening enforces production-safe defaults:

- **Fail-Closed Client Authentication**: By default (`auth.allow_anonymous: false`), `/v1/chat/completions` and
  `/v1/feedback` require a valid bearer key configured in `GATEWAY_API_KEYS="app1:key1,app2:key2"`. Unauthenticated
  calls receive `HTTP 401 Unauthorized` (`auth_required`).
  - *Client Migration*: Any client calling the gateway must supply `Authorization: Bearer <key>`. For local development
    or offline demos where anonymous access via `X-App-Id` is desired, explicitly set `GATEWAY_ALLOW_ANONYMOUS=1` or
    `auth.allow_anonymous: true` in the configuration.
  - *Key storage and rotation*: keys may be stored as `app:sha256:<hex>` (from `scripts/make_api_key.py`) and read
    from a mounted file (`GATEWAY_API_KEYS_FILE`); an app may hold several keys while it rotates (see SECURITY.md).
- **Internal Metadata Protection**: By default (`security.expose_internal_metadata: false`), response payloads contain
  standard OpenAI-compatible fields (`id`, `object`, `created`, `model`, `choices`, `usage`). The custom `gateway`
  diagnostic dictionary and internal headers (`x-gateway-model`, `x-gateway-tier`, `x-gateway-release`,
  `x-gateway-cost-usd`) are omitted to prevent leaking internal routing strategies, prompt versions, and vendor costs.
  Only `x-gateway-request-id` is returned. Set `GATEWAY_EXPOSE_METADATA=1` in debug/test environments to re-enable them.
  `X-Admin-Key` is strictly reserved for `/admin/*` endpoints and never used to authorize diagnostic metadata on chat routes.
- **Request Body Size Limits**: Inbound payloads are capped by `security.max_request_body_bytes` (default `10485760` / 10MB;
  override with `GATEWAY_MAX_BODY_BYTES`). Payloads exceeding this limit (via `Content-Length` or streamed chunk accumulation)
  are rejected immediately with `HTTP 413 Payload Too Large` (`payload_too_large`).
- **Restrictive CORS**: Cross-origin browser access is disabled by default (`security.cors_origins: []`). Specify allowed
  domains via `GATEWAY_CORS_ORIGINS="https://app.example.com,https://dashboard.example.com"`.
- **Fail-Closed Admin API**: Admin endpoints (`/admin/*`) require `X-Admin-Key: <key>`. If `GATEWAY_ADMIN_KEY` is not set,
  the endpoints fail closed with `HTTP 403 Forbidden` (`admin_disabled`).

## Budgets, feedback and alerts

- **Budgets**: `budgets.default_daily_usd` and per-app `budgets.apps.<app>.daily_usd` (USD per UTC day, from the list
  prices in the config). An app is identified by its bearer key when `GATEWAY_API_KEYS="app:key,..."` is set, or
  by the `X-App-Id` header when anonymous access is explicitly enabled. Before its provider call a request reserves
  its estimated maximum cost (prompt plus `max_tokens`, or `budgets.reserve_output_tokens` output tokens, at the
  dearest model it may call), capped at what is left, in one SQLite transaction that counts the day's spend and the
  reservations of requests in flight; so concurrent requests can't share the same remaining budget. Logging the
  request records its real cost and drops its reservation in the same transaction; a reservation a crashed process
  never settled expires after `budgets.reservation_ttl_s`. A request is admitted while any budget is left, so the
  last one admitted can still overshoot by the part of its real cost its reservation didn't cover.
- **Per-client request limits**: `client_limits.default` and `client_limits.apps.<app>` set `rpm` (requests per
  minute) and `concurrency` (requests in flight, streams included until they end) per client; 0 means no limit. The
  shipped config allows 600 per minute and 50 at once (`demo`: 60 and 5). The counters live in Redis, so every
  replica shares one allowance (in-process with the memory backend). A refusal answers 429 `client_rate_limited`
  with `Retry-After`, is logged as rejected, and calls no provider. Only validated clients are counted: an invalid or
  revoked key gets 401 first and uses none of the allowance. A slot a crashed replica never released expires after
  `client_limits.lease_s`. While Redis is down the outage policy applies (`fail_closed`: 429; `local_share`: each
  replica allows `limit // replicas`). Token-per-minute limits per client are not implemented; daily USD budgets
  cap spend.
- **Feedback**: `POST /v1/feedback {"request_id": "<response id>", "score": "up" | "down" | 1 | -1, "comment": "..."}`.
- **Alerts**: error rate over the last `window_minutes` (with at least `min_requests`) above `alerts.error_rate`, or
  today's cost above `alerts.daily_cost_usd`. Each alert is logged, stored (the dashboard shows it), and POSTed as JSON
  to `ALERT_WEBHOOK_URL` when set; it fires at most once per window (error rate) or per day (cost).

## Metrics and logs

- **`GET /metrics`** (Prometheus text format, per replica): `gateway_requests_total{app,status,tier,error_type,stream}`,
  `gateway_request_duration_seconds` (histogram), `gateway_tokens_total{app,model,kind}`,
  `gateway_cost_usd_total{app,model}`, `gateway_provider_calls_total{model,status,error_type}`,
  `gateway_provider_call_duration_seconds`, `gateway_fallbacks_total`, `gateway_escalations_total`,
  `gateway_state_degraded` and `gateway_build_info`. Every finished request is counted once, from the same record
  the request log stores. Labels never hold prompt text, keys or provider messages; at most 100 distinct app ids
  are kept, later ones count as `other`. The HTTPS proxy answers 403 for `/metrics`; set `GATEWAY_METRICS_TOKEN`
  to require `Authorization: Bearer <token>` from scrapers as well.
- **Logs**: `GATEWAY_LOG_FORMAT=json` (set in Compose) writes one JSON object per line, and one `gateway.access`
  line per request (id, app, outcome, model, tokens, cost, latency; no content).
- **Prometheus and Grafana**: `docker compose --profile monitoring up -d` adds Prometheus (`127.0.0.1:9090`,
  scraping both replicas, alert rules in `config/prometheus/alerts.yml`: error rate, Redis degraded, replica down,
  slow p95) and Grafana (`127.0.0.1:3000`, admin password from `GRAFANA_ADMIN_PASSWORD`, the provisioned "LLM
  Gateway" dashboard). Routing the alerts to a pager needs an Alertmanager, which is not included.

## Retention, backup and restore

- **Retention**: `python -m gateway.manage retention --dry-run` counts, `--yes` deletes, in one transaction,
  requests, calls, feedback and alerts older than `storage.retention_days` (default 90, at least 2 so today's budget
  spend stays) or `--days N`, and budget reservations that have expired. Deployments are kept as the release audit
  trail. Run it daily (cron, a Kubernetes CronJob or `docker compose run --rm gateway python -m gateway.manage
  retention --yes`).
- **Backup**: `python scripts/db_backup.py backup` writes a custom-format `pg_dump` of the Compose database to
  `var/backups/` (`--local` uses `pg_dump` on this machine with `$GATEWAY_DATABASE_URL`; SQLite uses the online
  backup API). Backups contain app ids, costs and feedback comments: store them encrypted and off-site.
- **Restore**: `python scripts/db_backup.py restore <archive> --target <new database> --yes` restores next to the
  live database; overwriting the live one also needs `--overwrite-live`.
- **Drill**: `python scripts/db_backup.py drill` backs up, restores into a scratch database, compares the row
  count of every table and the Alembic revision, and drops the scratch database. Run it after every schema change
  and on a schedule; it passed on the Compose PostgreSQL 16 database at revision 0003 (5 requests) and again at 0004 in the final
  end-to-end run (76 requests).

## Router shadow mode and controlled activation

The trained classifier is never switched on in one step. Three stages, each read with `scripts/router_report.py`
(from the request log; `--json` to save it):

1. **Shadow** (`routing.shadow: true`): every routable request is still answered as configured, and the classifier
   is asked which tier it would have chosen; `requests.shadow_tier`/`shadow_score` and
   `gateway_shadow_decisions_total` record it. The report gives agreement with the tier used, the share it would
   send to the cheap model, and an estimated saving (strong answers it would have sent cheap, re-priced at the cheap
   model's list price). It cannot tell whether those cheap answers would have been good enough: compare user
   feedback per group, and sample requests for review.
2. **Canary** (`default_mode: router` or `model="router"`, with `routing.classifier_share: 0.05`): the classifier
   routes that share of router-mode requests, chosen by a hash of the request id; the rest are a control group
   answered by the strong tier (and shadow-scored). The report compares the groups' error rate, cost, latency and
   feedback. Raise the share in steps (0.05, 0.25, 1.0) only when the classifier group is no worse.
3. **Rollback** at any stage: `classifier_share: 0` (or `routing.router: rules`) and a rolling restart.

Only the model pair the classifier was trained for is ever routed or shadow-scored by it; with the shipped config
that is the OpenRouter batch pair. The shipped config keeps shadow off and `default_mode: strong`, and
`models/router.json` is not installed: turning any of this on is a deployment decision.

Retraining from feedback (Phase 6.6) is not built. The request log stores no prompt text, by design, so thumbs-up
or thumbs-down ratings can't be joined to the questions they rate. A retraining set would need an opt-in prompt
capture with its own retention and consent, and enough rated traffic to beat the 900 benchmark questions.

## Router lab: data, training, evaluation

The router is trained on which questions the cheap model gets right. The OpenRouter collection and held-out
evaluation are complete; the scripts use API keys and quota only when collecting new labels.

```bash
pip install -r requirements-lab.txt
```

1. **Questions** (no API calls): samples 400 GSM8K and 800 MMLU test questions and fixes a 75/25 train/test split
   before any model sees them. It refuses to overwrite the file, so the split can't change after answers exist.

   ```bash
   python scripts/prepare_questions.py
   ```

   If the download stops asking for `hf_xet`, set `HF_HUB_DISABLE_XET=1` (plain HTTP downloads) and rerun.

2. **Labels** (spends API quota): runs the cheap and the strong model on every question and grades the answers.
   Calls are paced under the per-minute limits in the config; every answer is appended to
   `router_data/answers.jsonl` the moment it arrives; rerunning skips everything already answered. It stops at a
   daily quota, a key problem or five errors in a row, and asks before spending quota (`--yes` skips the question).

   ```bash
   python scripts/collect_labels.py --dry-run
   ```

   ```bash
   python scripts/collect_labels.py --limit 20
   ```

   ```bash
   python scripts/collect_labels.py --yes
   ```

   Groq's free tier allows `openai/gpt-oss-120b` 8,000 tokens per minute and 200,000 per day. In a 20-question pilot
   a question took about 350-380 tokens per model (prompt plus answer, low reasoning effort), so all 1,200 questions
   need roughly 0.45M tokens per model: about three days of gpt-oss-120b's daily quota.

3. **Train** (no API calls; downloads the embedder once). Label: the cheap model is good enough when it is right, or
   when the strong model is wrong too. Regularisation is chosen by stratified 5-fold cross-validation.

   ```bash
   python scripts/train_router.py
   ```

4. **Evaluate** (no API calls): replays the held-out answers for always-cheap, always-strong, random routing (same
   share sent to strong as the router), the rules, the classifier and the cascade. It reports accuracy, cost per 1,000
   requests at list prices, the share sent to the strong model, the router's added latency, and the cheapest threshold
   within 1 percentage point of always-strong accuracy, and draws `results/cost_vs_accuracy.png`.

   ```bash
   python scripts/evaluate_router.py
   ```

5. **Compare providers** (no API calls). Another provider's models answer the same questions into their own file,
   with the provider, the host that served each answer, tokens, latency, the list-price estimate and the reported
   cost; a lock file stops two runs writing one file. For example, OpenRouter on 10 of the questions both Groq models
   answered, reading the key from another project's `.env` (never printed):

   ```bash
   python scripts/collect_labels.py --cheap or-gpt-oss-20b --strong or-gpt-oss-120b --answers router_data/answers_openrouter.jsonl --only-answered-in router_data/answers.jsonl --sample 10 --env-file ../Agentai/.env --yes
   ```

   `compare_providers.py` then compares every model on the questions they all answered: accuracy with 95%
   intervals, input/output tokens, cost per 1,000 questions, latency mean/p50/p95, failures and rate limits, and how
   often one model gives a different result on two hosts.

   ```bash
   python scripts/compare_providers.py
   ```

## Results

OpenRouter answered all 1,200 questions with both models. The collection completed with no final-call errors and
reported $0.06608 total cost, including the earlier 377-question comparison set. Dataset validation found no
integrity problems. The complete reports are in
[`results/dataset_quality.md`](results/dataset_quality.md),
[`results/provider_comparison_openrouter_1200.md`](results/provider_comparison_openrouter_1200.md), and
[`results/router_eval.md`](results/router_eval.md).

On the fixed 300-question held-out split, the trained classifier reached 88.0% accuracy versus 88.3% for always
using the strong model. It sent 67% of questions to the strong model and reduced estimated cost by 8% ($0.0259
versus $0.0282 per 1,000 requests). The one-answer difference was not significant (exact McNemar p = 1.000).
Classifier scoring added 42 ms p50 and 117 ms p95 on this CPU. Its threshold is 0.955, selected from out-of-fold
training scores and mirrored by `routing.classifier_threshold`. A mismatch prevents classifier loading.

The rules baseline selected a threshold that sent every held-out question to the strong model. The cascade also
escalated every question and cost 83% more than always-strong, so neither improved this OpenRouter pair.

### Router v2 (2026-11-22, offline)

A second classifier, MiniLM embeddings plus hashed word and word-pair features, trained and evaluated exactly like
v1, scored better in cross-validation (out-of-fold AUC 0.648 vs 0.638) but not on the held-out questions: 87.0%
accuracy with a 12% saving (v1: 88.0%, 8%), only 0.3 points above random routing at its strong share, and not
significantly different from always-strong (p = 0.125). v1 stays the candidate. Details:
[`results/router_v2.md`](results/router_v2.md).

### Load test (measured 2026-11-22, fake providers)

`scripts/load_test.py` against the Compose stack on one Windows laptop (two replicas, PostgreSQL 16, Redis 7, client
on the same machine), 20% of requests streamed. Fake providers: no paid calls, so this measures the gateway, its
database and Redis, not a provider. Raw results: [`results/load_test.json`](results/load_test.json).

| Run | Clients | Requests | Throughput | Status codes | p50 / p95 / p99 latency (successful) |
|---|---|---|---|---|---|
| Baseline, one client | 1 | 24 in 22 s | 1.1/s | 24 × 200 | 815 / 1,985 / 2,646 ms |
| Client limits off | 50 | 3,370 in 61 s | 55/s | 3,370 × 200 | 842 / 1,396 / 2,815 ms |
| Shipped client limits (600/min) | 80 | 7,278 in 63 s | 116/s | 650 × 200, 6,628 × 429 | 1,498 / 4,047 / 5,232 ms |

- Most of the latency is the fake provider's simulated delay: the median moved from 815 ms with one client to 842 ms
  with 50. Throughput at 50 clients was bound by that delay, not by a gateway error; no request failed.
- With the default limits, the busiest 60-second window in the request log held exactly 600 successful requests:
  the per-client limit held across both replicas through Redis. The 429s carry `Retry-After`.
- Every request was counted once by `/metrics` (`gateway_counted` equals `requests` in all three runs).
- Not measured: real providers under load, the HTTPS proxy in the path, hosts other than this laptop.

### Provider comparison (measured 2026-11-15 to 2026-11-16)

The same open-weight models on Groq and on OpenRouter (pinned to its DeepInfra bf16 host), on the 377 identical
GSM8K + MMLU questions, low reasoning effort. The consolidated table is in
[`results/provider_comparison.md`](results/provider_comparison.md). The full 1,200-question OpenRouter comparison is
in [`results/provider_comparison_openrouter_1200.md`](results/provider_comparison_openrouter_1200.md).

| Model | Accuracy (95% CI) | Cost per 1,000 questions | Latency p50 / p95 |
|---|---|---|---|
| Groq gpt-oss-20b | 83.6% (79-87%) | $0.052 | 0.57 / 0.91 s |
| Groq gpt-oss-120b | 87.3% (84-90%) | $0.107 | 0.69 / 1.12 s |
| OpenRouter gpt-oss-20b | 84.4% (80-88%) | $0.025 | 1.41 / 3.37 s |
| OpenRouter gpt-oss-120b | 88.6% (85-91%) | $0.030 | 4.34 / 15.48 s |

- The host made no measurable difference to accuracy (same model, Groq vs OpenRouter: exact McNemar p = 0.75 for 20b,
  0.38 for 120b). The same model gave the same final answer on both hosts for 88% (20b) and 94% (120b) of questions.
- 120b beat 20b on OpenRouter (p = 0.011); on Groq the gap was borderline (p = 0.059).
- OpenRouter's 120b was both more accurate than Groq's 20b (p = 0.005) and 43% cheaper, but its latency was 6-14x
  higher, with a long tail.
- On OpenRouter 120b costs only 17% more than 20b, so cheap/strong routing there can save little; the larger trade-off
  is latency against cost, across providers.
- Across all 1,200 OpenRouter questions, 120b scored 88.8% and 20b scored 84.9%. The difference was significant
  (exact McNemar p < 0.001), and the models produced the same final answer on 88% of questions.

Costs are list prices (OpenRouter's reported cost matched them on every call); Groq ran on its free tier. Latency was
measured from one machine on one day; Groq latency excludes 20 early answers recorded with local pacing waits. Two
OpenRouter calls hit the host's rate limit and succeeded on a rerun.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions (`stream: true` streams tokens as server-sent events, see [Streaming](#streaming)) |
| `GET /v1/models` | routing modes and configured models |
| `POST /v1/feedback` | thumbs up/down for a request id |
| `GET /health` | liveness: status (`ok` or `degraded`), active release, canary, router in use, state backend |
| `GET /ready` | readiness: 200 when the database and the state backend (Redis) answer, 503 otherwise |
| `/admin/*` | status, activate, rollback, canary (needs `GATEWAY_ADMIN_KEY` in `X-Admin-Key`; off when unset) |

Request headers: `Authorization: Bearer <key>` (a key from `GATEWAY_API_KEYS`; it names the app), `X-App-Id` (the app
name, only when anonymous access is enabled), `X-Gateway-Threshold` (per-request threshold), `X-Gateway-Latency`
(`interactive` or `batch`). Response headers: `x-gateway-request-id` always; `x-gateway-model`, `x-gateway-tier` and
`x-gateway-release` only with `GATEWAY_EXPOSE_METADATA=1`.

## Configuration

Everything is in [`config/gateway.yaml`](config/gateway.yaml), validated at startup (unknown keys and broken references
are errors): models with prices and per-minute limits, providers, prompt versions, releases, routing, retries, budgets,
alerts, auth, security, state and storage. Groq prices are on-demand list prices and change; check them before quoting
costs. The Gemini API models have no price set (free tier), so their cost shows as 0. Environment variables (also in
[`.env.example`](.env.example)):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | provider keys (names set by `providers.*.api_key_env`) |
| `GATEWAY_API_KEYS` | client keys, `app:key,app:key`; required unless anonymous access is enabled |
| `GATEWAY_ADMIN_KEY` | enables `/admin/*` (sent as `X-Admin-Key`) |
| `GATEWAY_ALLOW_ANONYMOUS` | `1` lets `X-App-Id` identify the app without a key (refused in production) |
| `GATEWAY_ENV` | `production` turns on the start-up guards: PostgreSQL, Redis, client and admin keys, no anonymous access |
| `GATEWAY_DATABASE_URL` | SQLAlchemy URL, e.g. `postgresql+psycopg://user:pass@host:5432/gateway`; unset: SQLite at `GATEWAY_DB` |
| `GATEWAY_DB` | SQLite path for development (default `var/gateway.db`) |
| `GATEWAY_STATE_BACKEND`, `GATEWAY_REDIS_URL` | `redis` and its URL for shared state across replicas (default `memory`) |
| `GATEWAY_REDIS_OUTAGE_POLICY`, `GATEWAY_REPLICAS` | `fail_closed` (default) or `local_share`, and the replica count it divides limits by |
| `GATEWAY_MAX_BODY_BYTES`, `GATEWAY_CORS_ORIGINS`, `GATEWAY_EXPOSE_METADATA` | request size limit, allowed CORS origins, internal metadata in responses |
| `GATEWAY_CONFIG`, `GATEWAY_HOST`, `GATEWAY_PORT`, `LOG_LEVEL` | config file, bind address and port, log level |
| `GATEWAY_FAKE_PROVIDERS`, `GATEWAY_FAKE_PACING` | fake models instead of real providers; make them honour rate limits (testing) |
| `ALERT_WEBHOOK_URL` | alerts are POSTed here as JSON |

## Development

```bash
pytest
```

```bash
ruff check .
```

With PostgreSQL and Redis (each test uses its own database; Redis tests flush the database number they use):

```bash
TEST_POSTGRES_URL=postgresql+psycopg://user:pass@localhost:5432/gateway_test GATEWAY_REDIS_URL=redis://localhost:56379/0 pytest
```

The tests never call a real API: providers are fakes, and the Groq, OpenRouter and Gemini adapters are tested against
httpx mock transports. One test drives the gateway with the OpenAI SDK by changing only `base_url`; another renders
the dashboard headless with Streamlit's `AppTest`. Without `TEST_POSTGRES_URL` the PostgreSQL tests skip, and without
Redis the Redis tests skip; in GitHub Actions the PostgreSQL tests fail instead of skipping.

**CI** (`.github/workflows/ci.yml`), on Python 3.11 and 3.12 with PostgreSQL 16 and Redis 7 services: Ruff, the
PostgreSQL 16 integration tests as their own step, then the rest of the suite (Redis tests included). A second job
builds the image and smoke-tests a Compose stack with two replicas and fake providers: health and readiness on both,
401 without a key, answers on both replicas, the dashboard.

## Production Architecture (Phase 3)

The LLM Gateway supports production-scale multi-instance deployments through SQLAlchemy 2.x Core, PostgreSQL 16, Redis 7, and Alembic migrations.

### 1. SQLite Development Setup
For local development, SQLite remains supported by default without external dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.lock   # exact, hash-checked versions (see SECURITY.md)
pytest
python -m gateway
```

### 2. PostgreSQL Production Setup
Production deployments require PostgreSQL 16 for persistent storage:
- Set `GATEWAY_DATABASE_URL=postgresql+psycopg://user:password@host:5432/gateway`.
- In production (`GATEWAY_ENV=production`), SQLite is rejected by fail-closed startup validation guards.
- Connection pooling parameters can be configured via `storage.pool_size`, `max_overflow`, `pool_timeout_s`, and `pool_recycle_s`.

### 3. Redis Distributed State Configuration
Production multi-instance deployments use Redis 7 for:
- **Distributed Rate Limiting**: Per-model RPM/TPM sliding windows computed atomically using Redis Lua scripts and Redis server time.
- **Shared Provider Cooldowns**: Model cooldowns following quota exhaustion (429) are synchronized across all gateway instances.
- **Distributed Circuit Breakers**: Shared `CLOSED`, `OPEN`, and `HALF_OPEN` states with atomic failure counting, recovery deadlines, and lease-expired half-open probes.
- Configure via `GATEWAY_STATE_BACKEND=redis` and `GATEWAY_REDIS_URL=redis://host:6379/0`.
- Memory policy must be set to `noeviction`.

### 4. Database Migrations (Alembic)
Schema evolution is managed using Alembic:
```bash
# Apply pending migrations to head
python -m gateway.migrations upgrade head

# Check that the database is at the head revision (the gateway refuses to start otherwise)
python -m gateway.migrations verify

# Adopt a database created before Alembic, then upgrade it
python -m gateway.migrations upgrade --adopt-existing

# Downgrade schema (destructive; requires explicit confirmation)
python -m gateway.migrations downgrade <revision> --confirm-destructive
```
The commands use `GATEWAY_DATABASE_URL` (or `--database-url`), else the SQLite file at `GATEWAY_DB`.

### 5. SQLite to PostgreSQL Migration
Migrate existing SQLite databases to PostgreSQL safely:
```bash
python scripts/migrate_sqlite_to_postgres.py --sqlite var/gateway.db --postgres-url postgresql+psycopg://user:password@host:5432/gateway
```
- Creates an automatic, verified backup of the SQLite database before running.
- Copies tables in batches with primary and foreign key preservation.
- Reconciles active budget reservations and skips expired ones.
- Resets PostgreSQL auto-increment sequences.
- Generates a reconciliation report with row count and cost validation.

### 6. Docker & Multi-Instance Deployment
Run the full production stack with two gateway replicas sharing PostgreSQL and Redis:
```bash
# Validate Docker Compose configuration
docker compose config

# Start all services (put GATEWAY_FAKE_PROVIDERS=1 in .env to run without provider keys)
docker compose up -d

# Also start the nginx load balancer on port 80 in front of both replicas
docker compose --profile proxy up -d
```
Services provided:
- `postgres`: PostgreSQL 16 on `127.0.0.1:54320` (container port 5432) with persistent volume `postgres-data`.
- `redis`: Redis 7 on `127.0.0.1:63790` (container port 6379) with `noeviction` policy.
- `migrations`: One-shot container that upgrades schema to `head` before gateways start.
- `gateway`: Primary gateway replica on `127.0.0.1:8000`.
- `gateway-replica`: Second gateway replica on `127.0.0.1:8001`.
- `dashboard`: Streamlit dashboard on `127.0.0.1:8501/dashboard/`.
- `proxy` (profile `proxy`): nginx with HTTPS on port 443 (port 80 redirects), round-robin over both replicas,
  response buffering off for SSE, `/admin/` blocked, the dashboard behind a password.

**Two-replica validation.** `python scripts/multi_instance_check.py` builds the image, starts this stack under its
own Compose project (`llmgw-mi`) with fake providers, and checks that both replicas share one daily budget, one
Redis rate limit, provider cooldowns and circuit breakers; that SSE streams through the proxy; that a restarted
replica comes back with the shared state; and that PostgreSQL connections stay bounded. It prints a JSON report,
exits 1 on any failure and removes the stack and its volumes afterwards (`--keep` leaves it running). Its test
config is the shipped one with tiny budgets, one rate-limited model and `GATEWAY_FAKE_PACING=1`, which makes fake
providers honour rate limits (they ignore them otherwise).

### 7. Health and Readiness Endpoints
- `GET /health`: Liveness probe. Returns HTTP 200 with service version, routing config, and state backend health (`"ok"` or `"degraded"`).
- `GET /ready`: Readiness probe. Returns HTTP 200 when PostgreSQL and Redis are both reachable, and HTTP 503 if any required infrastructure is disconnected.

### 8. Outage and Degraded Behavior
- **Redis Outage**: the gateway marks its state `degraded` in `/health` and answers HTTP 503 on `/ready`. Shared
  per-model RPM/TPM limits can't be enforced then, so `state.redis.outage_policy` decides (env
  `GATEWAY_REDIS_OUTAGE_POLICY`):
  - `fail_closed` (default): calls to models with RPM/TPM limits are refused with HTTP 429 and `Retry-After`
    (`state.redis.outage_retry_after_s`, default 5 s); models without limits keep working.
  - `local_share`: each replica enforces `limit // state.redis.replicas` locally (env `GATEWAY_REPLICAS`; set it to
    the number of gateway replicas).

  No replica ever falls back to the full limit, which would multiply the provider allowance by the replica count.
  Provider cooldowns and circuit breakers fall back to per-instance state, which only makes each replica more
  cautious. When Redis comes back, the next call reconnects and `/ready` returns 200.
- **PostgreSQL Outage**: Budget reservation and persistent request logging fail-closed; unreserved budgeted traffic is refused rather than silently allowed.

#### Redis tests and the outage drill (local)
```bash
docker run -d --name gateway-redis -p 127.0.0.1:56379:6379 redis:7-alpine   # the tests' default URL
pytest tests/test_redis_state.py tests/test_multi_instance.py tests/test_redis_outage.py
python scripts/redis_outage_drill.py --redis-url redis://127.0.0.1:56379/15 --container gateway-redis
python scripts/redis_outage_drill.py --redis-url redis://127.0.0.1:56379/15 --container gateway-redis --policy local_share
docker rm -f gateway-redis
```
The Redis tests skip when Redis isn't reachable (set `GATEWAY_REDIS_URL` to use another server; they flush the
database they use). The drill stops and restarts the container under a running gateway with fake providers and
prints the measured `/ready`, `/health` and request results for the healthy, outage and recovered phases; it exits
1 if any expectation fails. `tests/test_redis_outage.py` needs no Redis. Run everything with PostgreSQL too by
also setting `TEST_POSTGRES_URL` (see `tests/test_postgres_integration.py`).

### 9. Security Configuration
- `GATEWAY_API_KEYS="app1:key1,app2:key2"`: Client authentication is fail-closed by default.
- `GATEWAY_ADMIN_KEY="secret"`: Admin endpoints require `X-Admin-Key` header.
- `GATEWAY_MAX_BODY_BYTES=10485760`: Request body size limit (default: 10MB).
- `GATEWAY_CORS_ORIGINS`: Restrictive production CORS origins.
- `GATEWAY_EXPOSE_METADATA=0`: Sensitive internal gateway diagnostics and headers hidden by default.

## Directory layout

```
gateway/            the service
  app.py            FastAPI app and endpoints (/health, /ready, /v1)
  service.py        the request pipeline
  config.py         config schema and loading
  circuit.py        circuit breaker state machine
  ratelimit.py      token and request pacing
  state/            shared state backends (in-memory dev, Redis production)
  providers/        Groq, OpenRouter and Gemini adapters, fake provider, model registry
  routing/          rules, embeddings, logistic regression, classifier, cascade check
  monitoring/       SQLAlchemy Core store, metrics, alerts
  manage/           releases (activate, rollback, canary), budgets, CLI
migrations/         Alembic versioned migrations
dashboard/app.py    Streamlit dashboard
router_lab/         questions and grading, resumable collection, labels, evaluation
scripts/            router lab (prepare_questions, collect_labels, train_router, evaluate_router, ...),
                    migrate_sqlite_to_postgres, redis_outage_drill, multi_instance_check, send_demo_traffic
config/gateway.yaml configuration
tests/              pytest suite
```

## System requirements

- Python 3.11 or 3.12 (CI); 3.13 works locally. Docker with Compose v2 for the full stack.
- PostgreSQL 16 and Redis 7 for several replicas; SQLite and in-memory state are enough for one development process.
- Memory and CPU per replica haven't been measured yet (load testing is Phase 5); `WITH_EMBEDDINGS=true` adds CPU torch
  to the image.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `MigrationError: database revision is unversioned` at start-up | the PostgreSQL database isn't migrated: run `python -m gateway.migrations upgrade` (Compose runs the `migrations` service first) |
| Every request answers 401 `auth_required` | no `GATEWAY_API_KEYS` and anonymous access off: configure keys, or `GATEWAY_ALLOW_ANONYMOUS=1` for local demos |
| `/ready` answers 503 | the database or Redis is unreachable; `/health` shows which (`state.degraded`) |
| 429 `rate_limit_exceeded` while Redis is down | the `fail_closed` outage policy at work; restore Redis, or choose `local_share` with `GATEWAY_REPLICAS` |
| 429 `client_rate_limited` | the client's `client_limits` (requests per minute or in flight) are reached; wait `Retry-After` seconds or raise the limit |
| 429 `budget_exceeded` | the app's daily budget (UTC day) is spent or held by requests in progress |
| PostgreSQL tests skip | set `TEST_POSTGRES_URL` to a test database (its name must contain "test") |
| `psycopg` fails to import on Windows | a policy blocking native DLLs; run PostgreSQL work in Docker or WSL |

## Production deployment prerequisites

The step-by-step deployment, upgrade, incident and key-rotation procedures are in [docs/RUNBOOK.md](docs/RUNBOOK.md).

Not met by the repository alone: provide them before exposing the gateway.

- `GATEWAY_ENV=production` with PostgreSQL, Redis (`noeviction`), client keys and an admin key; start-up refuses less.
- A certificate from a real CA for the nginx `proxy` (the bundled script makes self-signed development ones), and a
  strong dashboard password (`DASHBOARD_PASSWORD`).
- Scheduled `retention --yes`, scheduled backups copied off the host (encrypted), and a periodic restore drill.
- An Alertmanager (or another receiver) for the Prometheus alert rules, and Grafana behind your SSO or VPN.

## Limitations and next steps

- An unauthenticated request with an invalid body gets the 400 validation error before the 401: FastAPI checks the
  body before the route runs. It reveals only the public request schema; a valid body without a key gets 401.
- Real providers through the gateway's HTTP API are verified by one smoke run (`scripts/real_provider_smoke.py`, four
  requests); the test suite itself uses fake providers or mocked HTTP. Gemini has not been called for real.
- Nothing is deployed. Retention and backups are commands to schedule; nothing schedules them yet.
- Redis is used through a synchronous client whose calls run in worker threads (never on the event loop); each
  call still holds a thread for one round trip, which bounds Redis throughput by the thread pool.
- The trained classifier is evaluated (see [Results](#results)) but not shipped: `models/router.json` is not
  committed, so `model="router"` requests use the rule-based router. On the held-out questions it saved 8% against
  always-strong with no significant accuracy difference, on benchmark questions only.
- Streaming has no retry or failover after the first chunk, so a provider failure mid-stream ends the answer early
  (with an SSE error event), and cascade mode does not check streamed answers.
- Text chat only: tools / function calling, images and `n > 1` are rejected with a clear 400.
- Costs are list-price estimates from the config, not provider invoices.
