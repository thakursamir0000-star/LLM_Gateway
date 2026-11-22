"""Gateway configuration: one YAML file, validated with pydantic.

Environment overrides:
- ``GATEWAY_CONFIG``: path of the YAML file (default ``config/gateway.yaml``);
- ``GATEWAY_DB``: path of the SQLite request log (``storage.db_path``).
- ``GATEWAY_DATABASE_URL``: SQLAlchemy database URL (required in production).

Relative paths in the file are resolved against the project root, so the
gateway, the dashboard and the scripts find the same files from any directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "gateway.yaml"

Mode = Literal["cheap", "strong", "router", "cascade"]
MODES: tuple[str, ...] = ("cheap", "strong", "router", "cascade")
TIERS: tuple[str, ...] = ("cheap", "strong")
LatencyClass = Literal["interactive", "batch"]
LATENCY_CLASSES: tuple[str, ...] = ("interactive", "batch")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Price(_Strict):
    """USD per 1M tokens."""

    input: float = Field(0.0, ge=0)
    output: float = Field(0.0, ge=0)


class Limits(_Strict):
    """Client-side pacing (0 = no limit)."""

    rpm: int = Field(0, ge=0)
    tpm: int = Field(0, ge=0)


class ModelSpec(_Strict):
    provider: str
    model: str
    price_per_1m: Price = Field(default_factory=Price)
    limits: Limits = Field(default_factory=Limits)
    params: dict[str, Any] = {}

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """List-price cost of one call in USD."""
        return (prompt_tokens * self.price_per_1m.input + completion_tokens * self.price_per_1m.output) / 1e6


class ProviderSpec(_Strict):
    base_url: str
    api_key_env: str
    timeout_s: float = Field(60.0, gt=0)


class TierModels(_Strict):
    cheap: str
    strong: str


class Release(_Strict):
    """The models and prompt version that serve traffic.

    ``cheap``/``strong`` serve interactive traffic. ``batch`` can name other
    models for latency-tolerant traffic (e.g. a slower, cheaper provider);
    without it, batch traffic uses the same models.
    """

    cheap: str
    strong: str
    prompt: str
    batch: TierModels | None = None
    note: str = ""

    def model_for(self, tier: str, latency: str = "interactive") -> str:
        if latency == "batch" and self.batch is not None:
            return getattr(self.batch, tier)
        return getattr(self, tier)


class Canary(_Strict):
    release: str
    share: float = Field(ge=0, le=1)


class Deployment(_Strict):
    active: str
    canary: Canary | None = None


class RoutingConfig(_Strict):
    default_mode: Mode = "router"
    router: Literal["rules", "classifier"] = "classifier"
    classifier_path: str = "models/router.json"
    classifier_threshold: float = Field(0.5, ge=0, le=1)
    threshold: float = Field(0.5, ge=0, le=1)
    fallback_on_error: bool = True


class RetryConfig(_Strict):
    attempts: int = Field(3, ge=1)
    max_wait_s: float = Field(10.0, ge=0)
    # After a quota error (daily limit, no credits) a model is skipped this long, or longer if the provider says so.
    quota_cooldown_s: float = Field(300.0, ge=0)


class CircuitBreakerConfig(_Strict):
    failure_threshold: int = Field(5, ge=1)
    recovery_timeout_s: float = Field(30.0, ge=0)
    half_open_probes: int = Field(1, ge=1)


class LatencyConfig(_Strict):
    """Which latency class a request is in when it doesn't say (X-Gateway-Latency)."""

    default: LatencyClass = "interactive"
    apps: dict[str, LatencyClass] = {}

    def class_for(self, app_id: str, requested: str | None = None) -> str:
        return requested or self.apps.get(app_id) or self.default


class AppBudget(_Strict):
    daily_usd: float | None = Field(None, ge=0)


class BudgetConfig(_Strict):
    default_daily_usd: float | None = Field(None, ge=0)
    soft_limit_fraction: float = Field(0.8, ge=0, le=1)
    apps: dict[str, AppBudget] = {}
    # A request reserves its estimated maximum cost before the provider call; without max_tokens the
    # output is assumed to be this many tokens. Reservations a crashed request never settled expire.
    reserve_output_tokens: int = Field(4096, ge=1)
    reservation_ttl_s: float = Field(900.0, gt=0)


class ClientLimit(_Strict):
    rpm: int = Field(0, ge=0)  # requests per minute per client (0 = no limit)
    concurrency: int = Field(0, ge=0)  # requests in flight at once per client (0 = no limit)


class ClientLimitsConfig(_Strict):
    """Per-client request limits, enforced across replicas through the state backend (Redis).

    A client is the app its validated API key names (or its X-App-Id when anonymous access is enabled). Daily
    USD budgets are separate (``budgets``). A request in flight holds a lease that expires after ``lease_s``, so
    a crashed replica can't hold a client's concurrency slots forever.
    """

    default: ClientLimit = Field(default_factory=ClientLimit)
    apps: dict[str, ClientLimit] = {}
    lease_s: float = Field(600.0, gt=0)

    def for_app(self, app_id: str) -> ClientLimit:
        return self.apps.get(app_id, self.default)


class AlertConfig(_Strict):
    error_rate: float = Field(0.2, ge=0, le=1)
    window_minutes: int = Field(15, ge=1)
    min_requests: int = Field(20, ge=1)
    daily_cost_usd: float | None = Field(5.0, ge=0)
    check_interval_s: float = Field(60.0, ge=0)
    webhook_env: str = "ALERT_WEBHOOK_URL"


class AuthConfig(_Strict):
    keys_env: str = "GATEWAY_API_KEYS"
    keys_file_env: str = "GATEWAY_API_KEYS_FILE"
    admin_key_env: str = "GATEWAY_ADMIN_KEY"
    allow_anonymous: bool = False


class SecurityConfig(_Strict):
    max_request_body_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    cors_origins: list[str] = Field(default_factory=list)
    expose_internal_metadata: bool = False


class StorageConfig(_Strict):
    db_path: str = "var/gateway.db"
    pool_size: int = Field(5, ge=1)
    max_overflow: int = Field(10, ge=0)
    pool_timeout_s: float = Field(30, gt=0)
    pool_recycle_s: int = Field(1800, ge=0)
    # Request log retention for `python -m gateway.manage retention` (the budget needs today's rows: at least 2).
    retention_days: int = Field(90, ge=2)


class RedisConfig(_Strict):
    url: str = "redis://localhost:6379/0"
    socket_timeout_s: float = Field(2.0, gt=0)
    connect_timeout_s: float = Field(2.0, gt=0)
    max_connections: int = Field(50, ge=1)
    # While Redis is unreachable the shared per-model RPM/TPM limits can't be enforced. fail_closed refuses
    # calls to rate-limited models (HTTP 429); local_share gives each replica limit // replicas locally.
    # Never every replica the full limit, which would multiply the provider allowance by the replica count.
    outage_policy: Literal["fail_closed", "local_share"] = "fail_closed"
    replicas: int = Field(1, ge=1)
    outage_retry_after_s: float = Field(5.0, gt=0)


class StateConfig(_Strict):
    backend: Literal["memory", "redis"] = "memory"
    redis: RedisConfig = Field(default_factory=RedisConfig)


class GatewayConfig(_Strict):
    models: dict[str, ModelSpec]
    providers: dict[str, ProviderSpec]
    prompts: dict[str, str]
    releases: dict[str, Release]
    deployment: Deployment
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    client_limits: ClientLimitsConfig = Field(default_factory=ClientLimitsConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    state: StateConfig = Field(default_factory=StateConfig)

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        for alias, spec in self.models.items():
            if spec.provider not in self.providers:
                raise ValueError(f"model {alias!r}: unknown provider {spec.provider!r}")
        for name, release in self.releases.items():
            for tier in TIERS:
                for latency in LATENCY_CLASSES:
                    alias = release.model_for(tier, latency)
                    if alias not in self.models:
                        where = f"batch.{tier}" if latency == "batch" and release.batch else tier
                        raise ValueError(f"release {name!r}: unknown {where} model {alias!r}")
            if release.prompt not in self.prompts:
                raise ValueError(f"release {name!r}: unknown prompt {release.prompt!r}")
        if self.deployment.active not in self.releases:
            raise ValueError(f"deployment.active: unknown release {self.deployment.active!r}")
        canary = self.deployment.canary
        if canary is not None:
            if canary.release not in self.releases:
                raise ValueError(f"deployment.canary: unknown release {canary.release!r}")
            if canary.release == self.deployment.active:
                raise ValueError("deployment.canary: the canary must differ from the active release")
        return self

    def resolve_model(self, name: str) -> str | None:
        """The alias for ``name``, given as an alias or as a provider model name."""
        if name in self.models:
            return name
        for alias, spec in self.models.items():
            if spec.model == name:
                return alias
        return None


def resolve_path(path: str | Path) -> Path:
    """``path`` itself if absolute, else relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def database_target(config: GatewayConfig) -> str | Path:
    """Configured SQLAlchemy URL, or the development SQLite path."""
    return os.environ.get("GATEWAY_DATABASE_URL") or resolve_path(config.storage.db_path)


def load_config(path: str | Path | None = None) -> GatewayConfig:
    """Read and validate the config (``path``, else $GATEWAY_CONFIG, else the default)."""
    path = Path(path or os.environ.get("GATEWAY_CONFIG") or DEFAULT_CONFIG)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = GatewayConfig.model_validate(data)
    db = os.environ.get("GATEWAY_DB")
    if db:
        config.storage.db_path = db
    allow_anon = os.environ.get("GATEWAY_ALLOW_ANONYMOUS")
    if allow_anon is not None:
        config.auth.allow_anonymous = allow_anon.strip().lower() in ("1", "true", "yes", "on")
    max_bytes = os.environ.get("GATEWAY_MAX_BODY_BYTES")
    if max_bytes:
        try:
            config.security.max_request_body_bytes = int(max_bytes)
        except ValueError:
            pass
    cors = os.environ.get("GATEWAY_CORS_ORIGINS")
    if cors is not None:
        config.security.cors_origins = [o.strip() for o in cors.split(",") if o.strip()]
    expose_meta = os.environ.get("GATEWAY_EXPOSE_METADATA")
    if expose_meta is not None:
        config.security.expose_internal_metadata = expose_meta.strip().lower() in ("1", "true", "yes", "on")
    state_backend = os.environ.get("GATEWAY_STATE_BACKEND")
    if state_backend:
        backend_val = state_backend.strip().lower()
        if backend_val in ("memory", "redis"):
            config.state.backend = backend_val
    redis_url = os.environ.get("GATEWAY_REDIS_URL") or os.environ.get("REDIS_URL")
    if redis_url:
        config.state.redis.url = redis_url.strip()
    outage_policy = os.environ.get("GATEWAY_REDIS_OUTAGE_POLICY")
    if outage_policy:
        config.state.redis.outage_policy = outage_policy.strip().lower()  # type: ignore[assignment]
        RedisConfig.model_validate(config.state.redis.model_dump())  # reject unknown policies
    replicas = os.environ.get("GATEWAY_REPLICAS")
    if replicas:
        config.state.redis.replicas = max(1, int(replicas))
    return config
