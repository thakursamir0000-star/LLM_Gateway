"""Gateway configuration: one YAML file, validated with pydantic.

Environment overrides:
- ``GATEWAY_CONFIG``: path of the YAML file (default ``config/gateway.yaml``);
- ``GATEWAY_DB``: path of the SQLite request log (``storage.db_path``).

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


class Release(_Strict):
    cheap: str
    strong: str
    prompt: str
    note: str = ""


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
    threshold: float = Field(0.5, ge=0, le=1)
    fallback_on_error: bool = True


class RetryConfig(_Strict):
    attempts: int = Field(3, ge=1)
    max_wait_s: float = Field(10.0, ge=0)


class AppBudget(_Strict):
    daily_usd: float | None = Field(None, ge=0)


class BudgetConfig(_Strict):
    default_daily_usd: float | None = Field(None, ge=0)
    soft_limit_fraction: float = Field(0.8, ge=0, le=1)
    apps: dict[str, AppBudget] = {}


class AlertConfig(_Strict):
    error_rate: float = Field(0.2, ge=0, le=1)
    window_minutes: int = Field(15, ge=1)
    min_requests: int = Field(20, ge=1)
    daily_cost_usd: float | None = Field(5.0, ge=0)
    check_interval_s: float = Field(60.0, ge=0)
    webhook_env: str = "ALERT_WEBHOOK_URL"


class AuthConfig(_Strict):
    keys_env: str = "GATEWAY_API_KEYS"
    admin_key_env: str = "GATEWAY_ADMIN_KEY"


class StorageConfig(_Strict):
    db_path: str = "var/gateway.db"


class GatewayConfig(_Strict):
    models: dict[str, ModelSpec]
    providers: dict[str, ProviderSpec]
    prompts: dict[str, str]
    releases: dict[str, Release]
    deployment: Deployment
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        for alias, spec in self.models.items():
            if spec.provider not in self.providers:
                raise ValueError(f"model {alias!r}: unknown provider {spec.provider!r}")
        for name, release in self.releases.items():
            for tier in TIERS:
                if getattr(release, tier) not in self.models:
                    raise ValueError(f"release {name!r}: unknown {tier} model {getattr(release, tier)!r}")
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


def load_config(path: str | Path | None = None) -> GatewayConfig:
    """Read and validate the config (``path``, else $GATEWAY_CONFIG, else the default)."""
    path = Path(path or os.environ.get("GATEWAY_CONFIG") or DEFAULT_CONFIG)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = GatewayConfig.model_validate(data)
    db = os.environ.get("GATEWAY_DB")
    if db:
        config.storage.db_path = db
    return config
