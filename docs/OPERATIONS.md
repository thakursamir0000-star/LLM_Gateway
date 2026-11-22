# Operations and configuration reference

Detailed behaviour of streaming, circuit breakers, releases, security, budgets, metrics, storage and multi-replica
operation. Step-by-step procedures (deploy, upgrade, incidents, key rotation) are in [RUNBOOK.md](RUNBOOK.md);
the overview is in the [README](../README.md).

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

With the Redis state backend the breaker state is shared by every replica (Lua scripts, atomic counts and
probe leases; `scripts/multi_instance_check.py` verifies it); with the in-memory backend it is per process. While
Redis is unreachable each replica falls back to its own breaker, which only makes it more cautious.

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

## Security

Production-safe defaults:

- **Fail-Closed Client Authentication**: By default (`auth.allow_anonymous: false`), `/v1/chat/completions` and
  `/v1/feedback` require a valid bearer key configured in `GATEWAY_API_KEYS="app1:key1,app2:key2"`. Unauthenticated
  calls receive `HTTP 401 Unauthorized` (`auth_required`).
  - *Client Migration*: Any client calling the gateway must supply `Authorization: Bearer <key>`. For local development
    or offline demos where anonymous access via `X-App-Id` is desired, explicitly set `GATEWAY_ALLOW_ANONYMOUS=1` or
    `auth.allow_anonymous: true` in the configuration.
  - *Key storage and rotation*: keys may be stored as `app:sha256:<hex>` (from `scripts/make_api_key.py`) and read
    from a mounted file (`GATEWAY_API_KEYS_FILE`); an app may hold several keys while it rotates (see [SECURITY.md](../SECURITY.md)).
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
  dearest model it may call), capped at what is left, in one transaction (SQLite `BEGIN IMMEDIATE`, PostgreSQL an
  advisory lock) that counts the day's spend and the
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

Everything is in [`config/gateway.yaml`](../config/gateway.yaml), validated at startup (unknown keys and broken references
are errors): models with prices and per-minute limits, providers, prompt versions, releases, routing, retries, budgets,
alerts, auth, security, state and storage. Groq prices are on-demand list prices and change; check them before quoting
costs. The Gemini API models have no price set (free tier), so their cost shows as 0. Environment variables (also in
[`.env.example`](../.env.example)):

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
| `GATEWAY_API_KEYS_FILE` | file with `app:key` or `app:sha256:<hex>` entries (a mounted secret), read with `GATEWAY_API_KEYS` |
| `GATEWAY_METRICS_TOKEN` | when set, `/metrics` requires `Authorization: Bearer <token>` |
| `GATEWAY_LOG_FORMAT` | `json`: one JSON object per log line (set in Compose) |
| `GRAFANA_ADMIN_PASSWORD` | Grafana's admin password (Compose `monitoring` profile; Grafana refuses to start without it) |

## Multi-replica architecture

The gateway supports multi-replica deployments through SQLAlchemy 2.x Core, PostgreSQL 16, Redis 7, and Alembic migrations.

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

# Also start the nginx HTTPS proxy (443; port 80 redirects) in front of both replicas
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
gateway/              the service
  app.py              FastAPI app and endpoints (/v1, /health, /ready, /metrics, /admin)
  service.py          the request pipeline (limits, budget, release, routing, fallback, logging)
  config.py           config schema and loading
  circuit.py          circuit breaker state machine
  ratelimit.py        token and request pacing
  logs.py             JSON logging and access log
  state/              shared state: in-memory (development) and Redis (Lua scripts)
  providers/          Groq, OpenRouter and Gemini adapters, fake provider, model registry
  routing/            rules, embeddings, logistic regression, classifier, cascade check
  monitoring/         SQLAlchemy Core store, request-log metrics, Prometheus exporter, alerts
  manage/             releases (activate, rollback, canary), budgets, retention CLI
migrations/           Alembic migrations 0001-0004
dashboard/app.py      Streamlit request-log dashboard
router_lab/           questions and grading, resumable collection, labels, evaluation
router_data/          benchmark questions and the collected answers
scripts/              router lab, load test, backup/restore, key generation, proxy secrets,
                      multi-replica check, Redis outage drill, real-provider smoke test, router report
config/               gateway.yaml, nginx.conf, Prometheus and Grafana provisioning
results/              evaluation reports, load-test and smoke-test results
docs/                 demo, routing, operations, runbook, resume notes, architecture diagram, screenshots
tests/                pytest suite
```

## System requirements

- Python 3.11 or 3.12 (CI); 3.13 works locally. Docker with Compose v2 for the full stack.
- PostgreSQL 16 and Redis 7 for several replicas; SQLite and in-memory state are enough for one development process.
- Per-replica memory and CPU have not been profiled. The load test ran two replicas, PostgreSQL, Redis, Prometheus
  and Grafana on one laptop. `WITH_EMBEDDINGS=true` adds CPU torch to the image for the trained router.

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

The step-by-step deployment, upgrade, incident and key-rotation procedures are in [RUNBOOK.md](RUNBOOK.md).

Not met by the repository alone: provide them before exposing the gateway.

- `GATEWAY_ENV=production` with PostgreSQL, Redis (`noeviction`), client keys and an admin key; start-up refuses less.
- A certificate from a real CA for the nginx `proxy` (the bundled script makes self-signed development ones), and a
  strong dashboard password (`DASHBOARD_PASSWORD`).
- Scheduled `retention --yes`, scheduled backups copied off the host (encrypted), and a periodic restore drill.
- An Alertmanager (or another receiver) for the Prometheus alert rules, and Grafana behind your SSO or VPN.

## Validation runs

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
| PostgreSQL storage and Alembic migrations | yes | – | yes: 26 tests on PostgreSQL 16 in CI | – | no |
| Redis shared state and outage policy | yes | yes | yes: real Redis 7, outage drill | – | no |
| Authentication, admin key, body limits, CORS, generic upstream errors | yes | yes | yes: 401 without a key in the Docker smoke test | – | no |
| Locked, hash-checked dependencies; digest-pinned base image; vulnerability audit | yes | – | yes: CI installs the locks, audit workflow on every push and weekly | – | – |
| HTTPS proxy (TLS 1.2/1.3, redirect, security headers), dashboard password, admin API not published | yes | yes (password hashing) | yes: two-replica Docker check | – | no |
| Per-client request limits (requests per minute, requests in flight) | yes | yes | yes: real Redis 7, two backends sharing one allowance | – | no |
| Prometheus metrics, JSON access logs, Grafana dashboard, alert rules | yes | yes | yes: Compose run, both replicas scraped, counts matched traffic | – | no |
| Request-log retention, backup and restore (with a restore drill) | yes | yes (SQLite) | yes: retention on PostgreSQL 16, backup drill on the Compose database | – | no |

"Integration tested" means against real PostgreSQL, Redis or Docker Compose (`tests/test_postgres_integration.py`,
`tests/test_redis_*.py`, `scripts/redis_outage_drill.py`, `scripts/multi_instance_check.py`).

### Final end-to-end run (2026-11-22, this laptop, fake providers)

The merged code was run as a production-style stack: Docker Compose with `GATEWAY_ENV=production` (start-up guards
on), two replicas, PostgreSQL 16, Redis 7, the HTTPS proxy with a development certificate, Prometheus and Grafana;
client keys stored hashed in a mounted keys file, a hashed admin key, a metrics token given to Prometheus as
[RUNBOOK.md](RUNBOOK.md) describes. Fake providers, so no paid calls; nothing was published beyond
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
