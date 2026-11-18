"""FastAPI app: OpenAI-compatible endpoints, feedback, health and admin.

Run it with ``python -m gateway`` or
``uvicorn gateway.app:create_app --factory``. An OpenAI client switches to the
gateway by changing its base URL to ``http://<host>:8000/v1``.
"""

from __future__ import annotations

import hmac
import math
import os
import random
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from gateway import __version__
from gateway.config import LATENCY_CLASSES, ROOT, GatewayConfig, load_config, resolve_path
from gateway.manage.budgets import BudgetPolicy
from gateway.manage.releases import ReleaseError, ReleaseManager
from gateway.monitoring import metrics
from gateway.monitoring.alerts import Alert, AlertMonitor, log_notifier, webhook_notifier
from gateway.monitoring.store import Store, iso, utc_now
from gateway.providers import ModelRegistry, Provider, build_providers
from gateway.routing import Router, build_router
from gateway.schemas import ActivateRequest, CanaryRequest, ChatRequest, FeedbackRequest, NoteRequest
from gateway.service import MODE_ALIASES, Gateway, GatewayError, RequestContext

_APP_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def parse_api_keys(raw: str) -> dict[str, str]:
    """``"app1:key1,app2:key2"`` -> {key: app}."""
    keys = {}
    for item in raw.split(","):
        app_id, sep, key = item.strip().partition(":")
        if sep and app_id.strip() and key.strip():
            keys[key.strip()] = app_id.strip()
    return keys


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def create_app(config: GatewayConfig | None = None, *, providers: dict[str, Provider] | None = None,
               router: Router | None = None, store: Store | None = None, rng: random.Random | None = None,
               clock: Callable[[], datetime] = utc_now, notify: Callable[[Alert], None] | None = None,
               sleep: Callable[[float], Any] | None = None) -> FastAPI:
    """Build the app. Every dependency can be injected (tests pass fakes)."""
    if config is None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")  # does not override variables that are already set
        config = load_config()
    fake = providers is None and _truthy(os.environ.get("GATEWAY_FAKE_PROVIDERS"))
    providers = providers if providers is not None else build_providers(config, fake=fake)
    registry = ModelRegistry(config, providers, pacing=not fake, **({"sleep": sleep} if sleep else {}))
    owns_store = store is None
    store = store if store is not None else Store(resolve_path(config.storage.db_path))
    router = router if router is not None else build_router(config)
    webhook = os.environ.get(config.alerts.webhook_env)
    notify = notify or (webhook_notifier(webhook) if webhook else log_notifier)
    releases = ReleaseManager(config, store)
    gateway = Gateway(config, registry, store, router, releases, BudgetPolicy(config.budgets),
                      AlertMonitor(store, config.alerts, notify, clock), rng=rng, clock=clock)
    api_keys = parse_api_keys(os.environ.get(config.auth.keys_env, ""))
    admin_key = os.environ.get(config.auth.admin_key_env) or None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await registry.aclose()
        if owns_store:
            store.close()

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan,
                  description="OpenAI-compatible gateway with cost/quality routing and request monitoring.")
    app.state.gateway = gateway

    if config.security.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        allow_all = "*" in config.security.cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.security.cors_origins,
            allow_credentials=not allow_all,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def enforce_body_size_limit(request: Request, call_next):
        max_bytes = config.security.max_request_body_bytes
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    err = GatewayError(413, f"request payload too large ({content_length} bytes > {max_bytes} bytes limit)",
                                       code="payload_too_large")
                    return JSONResponse(err.body(), status_code=413)
            except ValueError:
                pass

        bytes_received = 0
        original_receive = request._receive

        async def limited_receive():
            nonlocal bytes_received
            message = await original_receive()
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > max_bytes:
                    request.state.body_size_exceeded = True
                    raise GatewayError(413, f"request body exceeded limit of {max_bytes} bytes",
                                       code="payload_too_large")
            return message

        request._receive = limited_receive
        try:
            response = await call_next(request)
        except GatewayError as exc:
            return JSONResponse(exc.body(), status_code=exc.status)

        if getattr(request.state, "body_size_exceeded", False):
            err = GatewayError(413, f"request body exceeded limit of {max_bytes} bytes",
                               code="payload_too_large")
            return JSONResponse(err.body(), status_code=413)
        return response

    # ── errors in OpenAI's format ───────────────────────────────────
    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        headers = {"Retry-After": str(math.ceil(exc.retry_after))} if exc.retry_after else None
        return JSONResponse(exc.body(), status_code=exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
        return JSONResponse(GatewayError(400, problems).body(), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(GatewayError(exc.status_code, str(exc.detail)).body(), status_code=exc.status_code)

    # ── auth ────────────────────────────────────────────────────────
    def app_id_for(request: Request) -> str:
        raw_app_id = request.headers.get("x-app-id", "").strip()
        if raw_app_id and not _APP_ID.fullmatch(raw_app_id):
            raise GatewayError(400, "X-App-Id must be 1-64 letters, digits, '.', '_' or '-'", param="X-App-Id")

        if api_keys:
            header = request.headers.get("authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else ""
            for key, app_id in api_keys.items():
                if token and hmac.compare_digest(token, key):
                    return app_id
            raise GatewayError(401, "invalid or missing API key", code="invalid_api_key", counts_as_error=False)

        if not config.auth.allow_anonymous:
            raise GatewayError(401, f"authentication required: configure {config.auth.keys_env}",
                               code="auth_required", counts_as_error=False)

        return raw_app_id or "default"

    def require_admin(request: Request) -> None:
        if admin_key is None:
            raise GatewayError(403, f"admin endpoints are off: set {config.auth.admin_key_env} to enable them",
                               code="admin_disabled")
        if not hmac.compare_digest(request.headers.get("x-admin-key", ""), admin_key):
            raise GatewayError(401, "invalid admin key", code="invalid_admin_key")

    def threshold_for(request: Request) -> float | None:
        raw = request.headers.get("x-gateway-threshold")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not 0.0 <= value <= 1.0:
            raise GatewayError(400, "X-Gateway-Threshold must be a number from 0 to 1", param="X-Gateway-Threshold")
        return value

    def latency_for(request: Request) -> str | None:
        raw = request.headers.get("x-gateway-latency")
        if raw is None:
            return None
        value = raw.strip().lower()
        if value not in LATENCY_CLASSES:
            raise GatewayError(400, "X-Gateway-Latency must be 'interactive' or 'batch'", param="X-Gateway-Latency")
        return value

    # ── OpenAI-compatible API ───────────────────────────────────────
    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request):
        ctx = RequestContext(app_id=app_id_for(request), threshold=threshold_for(request),
                             latency=latency_for(request))
        expose_meta = config.security.expose_internal_metadata
        if not body.stream:
            result = await gateway.chat(body, ctx)
            return JSONResponse(result.to_openai(expose_metadata=expose_meta),
                                headers=result.headers(expose_metadata=expose_meta))

        generator, headers = await gateway.stream(body, ctx, expose_metadata=expose_meta)
        return StreamingResponse(generator, media_type="text/event-stream", headers=headers)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        modes = [m for m in MODE_ALIASES if m and m != "gateway"]
        data = [{"id": m, "object": "model", "created": 0, "owned_by": "llm-gateway"} for m in modes]
        data += [{"id": alias, "object": "model", "created": 0, "owned_by": spec.provider}
                 for alias, spec in config.models.items()]
        return {"object": "list", "data": data}

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackRequest, request: Request) -> dict[str, Any]:
        app_id = app_id_for(request)
        owner = store.request_app(body.request_id)
        if owner is None or (api_keys and owner != app_id):
            raise GatewayError(404, f"no request {body.request_id!r}", param="request_id", code="request_not_found")
        feedback_id = store.add_feedback(body.request_id, app_id, body.value, body.comment, iso(clock()))
        return {"ok": True, "feedback_id": feedback_id, "request_id": body.request_id, "score": body.value}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        state = releases.state()
        return {"status": "ok", "version": __version__, "release": state.active, "canary": state.canary,
                "canary_share": state.canary_share, "default_mode": config.routing.default_mode,
                "threshold": config.routing.threshold,
                "classifier_threshold": config.routing.classifier_threshold,
                "latency_default": config.latency.default,
                "router": router.description, "fake_providers": fake,
                "auth": "api_keys" if api_keys else ("x-app-id" if config.auth.allow_anonymous else "fail_closed")}

    # ── admin ───────────────────────────────────────────────────────
    def deployment_view() -> dict[str, Any]:
        state = releases.state()
        return {"active": state.active, "canary": state.canary, "canary_share": state.canary_share,
                "source": state.source, "since": state.ts,
                "releases": {name: r.model_dump() for name, r in config.releases.items()}}

    def change(action: Callable[[], Any]) -> dict[str, Any]:
        try:
            action()
        except ReleaseError as e:
            raise GatewayError(400, str(e), code="invalid_release_change") from e
        return deployment_view()

    @app.get("/admin/status")
    async def admin_status(request: Request) -> dict[str, Any]:
        require_admin(request)
        now = clock()
        day = iso(now)[:10]
        budgets = gateway.budgets
        spend = [{**row, "daily_limit_usd": budgets.limit(row["app_id"])} for row in metrics.app_spend(store, day)]
        return {"deployment": deployment_view(), "router": router.description,
                "last_24h": metrics.summary(store, since=iso(now - timedelta(hours=24))),
                "spend_today": spend, "alerts": store.alerts(10)}

    @app.post("/admin/activate")
    async def admin_activate(body: ActivateRequest, request: Request) -> dict[str, Any]:
        require_admin(request)
        return change(lambda: releases.activate(body.release, body.note))

    @app.post("/admin/rollback")
    async def admin_rollback(request: Request, body: NoteRequest | None = None) -> dict[str, Any]:
        require_admin(request)
        return change(lambda: releases.rollback(body.note if body else ""))

    @app.post("/admin/canary")
    async def admin_canary(body: CanaryRequest, request: Request) -> dict[str, Any]:
        require_admin(request)
        return change(lambda: releases.set_canary(body.release, body.share, body.note))

    @app.delete("/admin/canary")
    async def admin_canary_off(request: Request) -> dict[str, Any]:
        require_admin(request)
        return change(lambda: releases.clear_canary())

    @app.get("/admin/deployments")
    async def admin_deployments(request: Request, limit: int = 50) -> dict[str, Any]:
        require_admin(request)
        return {"deployments": releases.history(limit)}

    @app.get("/admin/alerts")
    async def admin_alerts(request: Request, limit: int = 50) -> dict[str, Any]:
        require_admin(request)
        return {"alerts": store.alerts(limit)}

    return app
