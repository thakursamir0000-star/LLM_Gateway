# Deployment and operations runbook

How to deploy the gateway on one host with Docker Compose, operate it, and recover from the failures it has been
tested against. Nothing here has been run on a public host yet: the commands were checked on a local Compose stack
(see the README's validation table). Replace every `<...>`.

## 1. Prerequisites

| Need | Why |
|---|---|
| A Linux host with Docker Engine 24+ and Compose v2, ports 80 and 443 open | the stack and the HTTPS proxy |
| A DNS name and a certificate from a real CA (`server.crt`, `server.key`) | the bundled script only makes self-signed ones |
| A PostgreSQL 16 password other than the Compose default, or a managed PostgreSQL | the request log, budgets, releases |
| Provider keys (Groq, OpenRouter, Gemini) with spending limits set at the provider | the gateway caps spend per app, the provider caps the account |
| Somewhere off the host for encrypted backups | §6 |

## 2. Secrets (never in the repository)

Create on the host, readable only by the deploying user (`chmod 600`):

1. `.env` next to `docker-compose.yml`:
   ```bash
   GATEWAY_ENV=production
   GROQ_API_KEY=<key>
   OPENROUTER_API_KEY=<key>
   GEMINI_API_KEY=<key>
   GATEWAY_API_KEYS_FILE=/run/secrets/api-keys
   GATEWAY_ADMIN_KEY=sha256:<hex of the admin key>
   GATEWAY_METRICS_TOKEN=<random>                # optional; if set, also give it to Prometheus (step 5)
   GRAFANA_ADMIN_PASSWORD=<random>
   GATEWAY_REPLICAS=2
   ```
2. Client keys: `python scripts/make_api_key.py <app> --append secrets/api-keys` per client; give each client its key,
   mount `secrets/api-keys` at `/run/secrets/api-keys` in both gateway services (a Compose override).
3. TLS and the dashboard password: put the CA certificate and key in `config/tls/` (or `GATEWAY_TLS_DIR`), then
   `DASHBOARD_PASSWORD=<password> python scripts/make_proxy_secrets.py` writes `dashboard.htpasswd` (it also makes a
   self-signed certificate: overwrite it with the real one afterwards).
4. Change the PostgreSQL password in `docker-compose.yml` (`POSTGRES_PASSWORD` and every `GATEWAY_DATABASE_URL`) or
   point `GATEWAY_DATABASE_URL` at a managed database.
5. The metrics token. With `GATEWAY_METRICS_TOKEN` set, every replica answers `/metrics` with 401 unless the scrape
   sends the token, and the shipped `config/prometheus/prometheus.yml` sends none. Prometheus must get the token
   too, or every target shows as down and `GatewayReplicaDown` fires:
   - write the same token, with no trailing newline, to `secrets/metrics-token` (`printf '%s' '<token>' > ...`);
   - mount it in the `prometheus` service (in the same Compose override as step 2):
     `- ./secrets/metrics-token:/etc/prometheus/metrics-token:ro`;
   - in `config/prometheus/prometheus.yml`, uncomment the
     `authorization: {type: Bearer, credentials_file: /etc/prometheus/metrics-token}` line under the `gateway` job.

   Without the token, `/metrics` is still not public: the proxy answers 403 for it and the gateway ports are bound
   to loopback. In that case leave `GATEWAY_METRICS_TOKEN` out of `.env` and change nothing in Prometheus.

## 3. First deployment

```bash
git checkout <release tag>
docker compose build
docker compose run --rm migrations                       # Alembic upgrade to head
docker compose --profile proxy --profile monitoring up -d
```

Check, in order:

```bash
docker compose ps                                        # every service healthy
curl -s https://<host>/health                            # "status": "ok", "auth": "api_keys"
curl -s http://127.0.0.1:8000/ready                      # database and Redis ok (loopback only)
curl -s -o /dev/null -w "%{http_code}\n" https://<host>/admin/status      # 403: admin API not published
curl -s -o /dev/null -w "%{http_code}\n" https://<host>/metrics           # 403
curl -s -o /dev/null -w "%{http_code}\n" https://<host>/dashboard/        # 401 without the password
curl -s http://127.0.0.1:9090/api/v1/targets | grep -o '"health":"[a-z]*"'  # "up" for both replicas
```

A target showing `"down"` with a 401 in `lastError` means the metrics token is set on the gateway but not given to
Prometheus (§2 step 5).

Then the real-provider smoke test on the host (paid, capped at `--max-cost-usd`, default $0.01; it runs its own
in-process gateway with the same keys): `python scripts/real_provider_smoke.py --dry-run`, then `--yes`. Finally
one request through the proxy with a client key:
`curl -s https://<host>/v1/chat/completions -H "Authorization: Bearer <key>" -H "Content-Type: application/json"
-d '{"model": "auto", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 16}'`. Open Grafana (`ssh -L 3000:127.0.0.1:3000 <host>`)
and check that the "LLM Gateway" dashboard shows the requests.

## 4. Routine operations

| Task | How | When |
|---|---|---|
| Retention | `docker compose run --rm gateway python -m gateway.manage retention --yes` | daily (cron) |
| Backup | `python scripts/db_backup.py backup`, then copy `var/backups/` off the host, encrypted | daily |
| Restore drill | `python scripts/db_backup.py drill` | weekly and after every migration |
| Dependency audit | the `dependency-audit` workflow; Dependabot PRs for actions and images | weekly |
| Key rotation | §7 | on schedule or on suspicion |
| Release / canary / rollback | `python -m gateway.manage canary <release> 0.1`, `activate`, `rollback` (README) | per change |

## 5. Upgrades

1. `python scripts/db_backup.py backup`.
2. `git checkout <new tag> && docker compose build`.
3. `docker compose run --rm migrations` (migrations are additive; the gateway refuses to start on an unmigrated
   database, so an old replica keeps serving until it is replaced).
4. Replace replicas one at a time: `docker compose up -d --no-deps gateway`, wait for healthy, then
   `docker compose up -d --no-deps gateway-replica`. nginx sends traffic to the healthy one meanwhile.
5. Watch the error-rate and latency panels for 15 minutes.

Rolling back code: check out the previous tag and repeat step 4. Rolling back a schema change: restore the backup
from step 1 into a new database (`scripts/db_backup.py restore ... --target <new db> --yes`) and point
`GATEWAY_DATABASE_URL` at it; `python -m gateway.migrations downgrade` exists but drops data and needs confirmation.

## 6. Incidents

| Symptom (alert or user report) | Check | Action |
|---|---|---|
| `GatewayStateDegraded`; clients get 429 `rate_limit_exceeded` or `client_rate_limited` | `docker compose ps redis`, `docker compose logs redis` | restart Redis; with `fail_closed` (default) rate-limited traffic waits for Redis. To keep serving at reduced limits: `GATEWAY_REDIS_OUTAGE_POLICY=local_share` and a rolling restart. Drill: `scripts/redis_outage_drill.py` |
| `/ready` 503 with a database error; requests fail | `docker compose logs postgres`, disk space | restore service; budgeted requests are refused while PostgreSQL is down (no unmetered spend) |
| `GatewayHighErrorRate` with `upstream_*` errors for one provider | Grafana "provider call failures", `/health` circuit state | the circuit breaker and fallbacks move traffic to other models; if the provider is down for long, activate a release without it (`python -m gateway.manage activate <release>`) |
| 429 `budget_exceeded` for an app | the app's `cost_usd` today in the dashboard, or `SELECT sum(cost_usd) FROM requests WHERE app_id='<app>' AND day=current_date` | expected at the daily cap; raise `budgets.apps.<app>.daily_usd` and restart if intended |
| 401 for a client that worked | is its entry still in the keys file? | re-issue a key (§7) |
| `GatewayReplicaDown` | `docker compose ps`, `docker compose logs gateway` | `docker compose up -d gateway`; the other replica serves meanwhile |
| Suspected key leak | — | rotate the key (§7) at once; for provider keys revoke at the provider first |

## 7. Key rotation

- **Client key**: `python scripts/make_api_key.py <app> --append secrets/api-keys`, rolling restart, give the client
  the new key, delete the old line, rolling restart. The old key gets 401 afterwards.
- **Admin key / metrics token / Grafana password**: change in `.env`, rolling restart. For the metrics token, update `secrets/metrics-token` at the same time (Prometheus re-reads its `credentials_file` on each scrape).
- **Provider key**: create the new key at the provider, update `.env`, rolling restart, revoke the old key there.

## 8. Not covered yet

- Multi-host deployment (Kubernetes, managed Redis and PostgreSQL with replicas). The gateway is stateless apart from
  PostgreSQL and Redis, so it scales out behind any load balancer, but only the single-host Compose stack is tested.
- Alertmanager routing for the Prometheus alert rules; Grafana behind SSO.
- Automatic scheduling of retention, backups and drills (§4 lists them; a cron or CI schedule must run them).
