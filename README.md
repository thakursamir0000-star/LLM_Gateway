# LLM Gateway

An OpenAI-compatible service that sits between your apps and LLM providers. For each request it picks a model with a
trained router (cheap model when it is likely good enough, strong model when it is not), and it records and monitors
every call: tokens, cost, latency, errors, route decisions and user feedback.

An app switches to it by changing one URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-without-gateway-keys")
reply = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "What is 17 * 23?"}])
print(reply.choices[0].message.content)
print(reply.model_extra["gateway"])  # which model answered, why, and what it cost
```

**Status:** phase 1. The gateway, routing, monitoring, release management and packaging are built and tested, with
fake providers or mocked HTTP. OpenRouter label collection is complete for all 1,200 questions, and the trained
router has been evaluated on the fixed 300-question held-out split. See [Results](#results).

## What it does

- **OpenAI-compatible API**: `POST /v1/chat/completions` (with `stream: true`), `GET /v1/models`, plus
  `POST /v1/feedback` for thumbs up/down. Errors use OpenAI's error format.
- **Providers**: Groq (`openai/gpt-oss-20b` as the cheap model, `openai/gpt-oss-120b` as the strong one), OpenRouter
  (the same two models, pinned to one full-precision host, with the real per-call cost it reports) and the Gemini API
  (Gemini and Gemma models). Calls are paced under each model's limits and retried within bounds; provider errors are
  stripped of account ids and keys before they are logged or returned.
- **Routing modes**, chosen per request with the `model` field:
  `cheap`, `strong`, `router` (trained classifier, or a rule-based baseline) and `cascade` (cheap first, escalate when
  the answer looks unsure). `auto` uses the configured default.
- **Latency-aware provider routing**: `interactive` requests go to fast models (Groq), `batch` requests to cheaper,
  slower ones (OpenRouter), with fallback across providers and a cooldown for a model that is out of quota.
- **Audited cost/quality controls**: separate classifier and rules/cascade defaults in `[0, 1]`. Higher sends more
  traffic to the strong model, and the classifier default must match its trained artifact.
- **Monitoring**: every request is logged to SQLite (model, route decision, tokens, cost, latency, error; never the
  prompt or the answer). A Streamlit dashboard shows requests, cost per day, p50/p95 latency, error rate and the share
  per model. Alerts fire when the error rate or daily cost passes a limit.
- **Release management**: model and prompt versions live in the config as releases; activate, roll back, or canary a
  share of traffic to a new release. Per-app daily budgets.
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
    Svc --> DB[("SQLite<br/>requests, calls, feedback,<br/>deployments, alerts")]
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

The response is a normal `chat.completion` with an extra `gateway` object (request id, release, canary, mode, router,
score, threshold, latency class, tier, model, escalation, fallback, cost, router time, reason) and `x-gateway-*`
headers. `usage`
counts every provider call the request made (a cascade escalation pays for two).

## Quick start

Python 3.11+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements-dev.txt
```

**Without API keys** (fake models, nothing leaves your machine):

```bash
GATEWAY_FAKE_PROVIDERS=1 python -m gateway
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

**Docker** (gateway on :8000, dashboard restricted to the host's loopback interface, sharing the request log in a
volume):

```bash
docker compose up --build
```

`WITH_EMBEDDINGS=true docker compose up --build` adds sentence-transformers (CPU torch) so the trained router can run
in the container; without it the gateway uses the rule-based router.

The dashboard is available locally at `http://127.0.0.1:8501`. It displays app IDs, model and release information,
costs, and error details, and it does not have built-in authentication. Docker Compose therefore publishes it only on
`127.0.0.1`. For remote administration, keep that binding and use an SSH tunnel:

```bash
ssh -L 8501:127.0.0.1:8501 user@gateway-host
```

Then open `http://127.0.0.1:8501` on the administrator's computer. An authenticated reverse proxy is also suitable
when managed remote access is required.

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
- **Feedback**: `POST /v1/feedback {"request_id": "<response id>", "score": "up" | "down" | 1 | -1, "comment": "..."}`.
- **Alerts**: error rate over the last `window_minutes` (with at least `min_requests`) above `alerts.error_rate`, or
  today's cost above `alerts.daily_cost_usd`. Each alert is logged, stored (the dashboard shows it), and POSTed as JSON
  to `ALERT_WEBHOOK_URL` when set; it fires at most once per window (error rate) or per day (cost).

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
| `GET /health` | status, active release, canary, router in use |
| `/admin/*` | status, activate, rollback, canary (needs `GATEWAY_ADMIN_KEY`) |

Request headers: `X-App-Id` (app name, when no gateway keys are configured), `X-Gateway-Threshold` (per-request
threshold), `X-Gateway-Latency` (`interactive` or `batch`). Response headers: `x-gateway-request-id`, `x-gateway-model`, `x-gateway-tier`, `x-gateway-release`,
`x-gateway-cost-usd`.

## Configuration

Everything is in [`config/gateway.yaml`](config/gateway.yaml), validated at startup (unknown keys and broken references
are errors): models with prices and per-minute limits, providers, prompt versions, releases, routing, retries, budgets,
alerts, auth and storage. Environment variables are listed in [`.env.example`](.env.example). Groq prices are on-demand
list prices and change; check them before quoting costs. The Gemini API models have no price set (free tier), so their
cost shows as 0.

## Development

```bash
pytest
```

```bash
ruff check .
```

The tests never call a real API: providers are fakes, and the Groq and Gemini adapters are tested against httpx mock
transports. One test drives the gateway with the OpenAI SDK by changing only `base_url`; another renders the dashboard
headless with Streamlit's `AppTest`. CI (`.github/workflows/ci.yml`) runs lint and tests on Python 3.11 and 3.12,
builds the Docker image, and smoke-tests the gateway and the dashboard containers with fake providers.

```
gateway/            the service
  app.py            FastAPI app and endpoints
  service.py        the request pipeline
  config.py         config schema and loading
  providers/        Groq, OpenRouter and Gemini adapters, fake provider, model registry (pacing, retries)
  routing/          rules, embeddings, logistic regression, classifier, cascade check
  monitoring/       SQLite store, metrics, alerts
  manage/           releases (activate, rollback, canary), budgets, CLI
dashboard/app.py    Streamlit dashboard
router_lab/         questions and grading, resumable collection, labels, evaluation, provider comparison
scripts/            prepare_questions, collect_labels, train_router, evaluate_router, compare_providers, send_demo_traffic
config/gateway.yaml configuration
tests/              pytest suite
```

## Production Architecture (Phase 3)

The LLM Gateway supports production-scale multi-instance deployments through SQLAlchemy 2.x Core, PostgreSQL 16, Redis 7, and Alembic migrations.

### 1. SQLite Development Setup
For local development, SQLite remains supported by default without external dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
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

# Check current revision
python -m gateway.migrations current

# Validate database revision matches head
python -m gateway.migrations check

# Downgrade schema (requires explicit confirmation)
python -m gateway.migrations downgrade <revision> --confirm-destructive
```

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

# Start all services with fake providers for evaluation
GATEWAY_FAKE_PROVIDERS=1 docker compose up -d
```
Services provided:
- `postgres`: PostgreSQL 16 on `127.0.0.1:5432` with persistent volume `postgres-data`.
- `redis`: Redis 7 on `127.0.0.1:6379` with `noeviction` policy.
- `migrations`: One-shot container that upgrades schema to `head` before gateways start.
- `gateway`: Primary gateway replica on port `8000`.
- `gateway-replica`: Second gateway replica on port `8001`.
- `dashboard`: Streamlit dashboard on `127.0.0.1:8501`.

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
scripts/            prepare_questions, train_router, migrate_sqlite_to_postgres
config/gateway.yaml configuration
tests/              pytest suite
```

## Limitations and next steps

- The trained classifier is evaluated (see [Results](#results)) but not shipped: `models/router.json` is not
  committed, so `model="router"` requests use the rule-based router. On the held-out questions it saved 8% against
  always-strong with no significant accuracy difference.
- Streaming has no retry or failover after the first chunk, so a provider failure mid-stream ends the answer early
  (with an SSE error event), and cascade mode does not check streamed answers.
- Text chat only: tools / function calling, images and `n > 1` are rejected with a clear 400.
- Costs are list-price estimates from the config, not provider invoices.
