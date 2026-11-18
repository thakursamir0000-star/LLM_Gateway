"""The request pipeline behind ``POST /v1/chat/completions``.

1. Validate (text chat only: no tools, n=1).
2. Mode from the ``model`` field: ``auto`` (the configured default), ``router``,
   ``cascade``, ``cheap``, ``strong``, or a configured model (pinned, no routing).
3. Budget: refuse at the app's daily limit; past the soft limit use the cheap model.
4. Release: the active one, or the canary for its share of traffic; its prompt
   version is put before the app's messages.
5. Latency class: interactive or batch (X-Gateway-Latency header, else the
   app's default, else the global one). A release can serve batch traffic with
   other models, e.g. a slower, cheaper provider.
6. Route: the router's score against the threshold picks cheap or strong;
   cascade starts with cheap. The tier and the latency class pick the model.
7. Call the model (paced, retried). If it fails, try the same tier on the other
   latency class's model, then both models in the other tier.
8. Cascade: check the cheap answer and escalate to strong when unsure.
9. Log one row per request and one per provider call, then check alerts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from gateway.config import GatewayConfig, Release
from gateway.manage.budgets import BudgetPolicy, seconds_until_utc_midnight
from gateway.manage.releases import ReleaseManager
from gateway.monitoring.alerts import AlertMonitor
from gateway.monitoring.store import CallRecord, RequestRecord, Store, iso, utc_now
from gateway.providers import ModelRegistry
from gateway.providers.base import (
    AUTH,
    BAD_REQUEST,
    OVERLOADED,
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    TIMEOUT,
    Completion,
    ProviderError,
    message_text,
    redact,
)
from gateway.routing import Router, choose_tier, routing_text
from gateway.routing.cascade import check_answer, should_escalate
from gateway.schemas import ChatRequest

log = logging.getLogger("gateway")

# `model` values that select a routing mode (None = the configured default).
MODE_ALIASES: dict[str, str | None] = {"auto": None, "gateway": None, "": None, "router": "router",
                                       "cascade": "cascade", "cheap": "cheap", "strong": "strong"}


class GatewayError(Exception):
    """An error returned to the client in OpenAI's error format."""

    def __init__(self, status: int, message: str, *, type_: str = "invalid_request_error", code: str | None = None,
                 param: str | None = None, retry_after: float | None = None, counts_as_error: bool | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.param = param
        self.retry_after = retry_after
        # Client mistakes and budget refusals are "rejected", not failures of the gateway.
        self.counts_as_error = status >= 500 if counts_as_error is None else counts_as_error

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


# What a client sees when the upstream call fails: status, type, code and a generic message per kind.
# The provider's own message can name the provider, model, account, service tier or quota figures, so
# it stays server-side (the request log and the gateway's log; see upstream_detail).
_PROVIDER_ERRORS = {
    RATE_LIMIT: (429, "rate_limit_error", "rate_limit_exceeded", "the upstream model is rate limited; retry later"),
    QUOTA_EXHAUSTED: (429, "rate_limit_error", "upstream_quota_exhausted",
                      "the upstream model's quota is used up; retry later"),
    AUTH: (502, "upstream_error", "upstream_auth_failed", "the gateway could not authenticate with the upstream provider"),
    BAD_REQUEST: (400, "invalid_request_error", "upstream_bad_request", "the upstream provider rejected the request"),
    TIMEOUT: (504, "upstream_error", "upstream_timeout", "the upstream model timed out"),
    OVERLOADED: (503, "upstream_error", "upstream_unavailable", "the upstream model is unavailable"),
}
_OTHER_PROVIDER_ERROR = (502, "upstream_error", "upstream_error", "the upstream model call failed")


def from_provider_error(error: ProviderError) -> GatewayError:
    """The client-facing error for a failed upstream call: generic message, useful status and code."""
    status, type_, code, message = _PROVIDER_ERRORS.get(error.kind, _OTHER_PROVIDER_ERROR)
    return GatewayError(status, message, type_=type_, code=code, retry_after=error.retry_after,
                        counts_as_error=error.kind != BAD_REQUEST)


def upstream_detail(request_id: str, error: ProviderError) -> str:
    """The provider's own error (redacted), logged server-side and kept in the request log."""
    detail = redact(str(error))[:500]
    log.warning("request %s: upstream call failed: %s", request_id, detail)
    return detail


def new_request_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


@dataclass
class RequestContext:
    app_id: str = "default"
    threshold: float | None = None  # per-request override (X-Gateway-Threshold)
    latency: str | None = None  # per-request latency class (X-Gateway-Latency)
    request_id: str = field(default_factory=new_request_id)


@dataclass
class ChatResult:
    request_id: str
    created: int
    model: str  # provider model name that produced the answer
    text: str
    finish_reason: str
    prompt_tokens: int  # summed over every provider call the request made
    completion_tokens: int
    gateway: dict[str, Any]

    @property
    def usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}

    def to_openai(self, expose_metadata: bool = True) -> dict[str, Any]:
        data = {
            "id": self.request_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.text},
                         "finish_reason": self.finish_reason, "logprobs": None}],
            "usage": self.usage,
        }
        if expose_metadata:
            data["gateway"] = self.gateway
        return data

    def to_chunks(self, include_usage: bool = False, expose_metadata: bool = True) -> list[dict[str, Any]]:
        """The answer as ``chat.completion.chunk`` objects (sent once the model has finished)."""
        base = {"id": self.request_id, "object": "chat.completion.chunk", "created": self.created,
                "model": self.model}
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {"content": self.text}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": self.finish_reason}]},
        ]
        if include_usage:
            chunk = {**base, "choices": [], "usage": self.usage}
            if expose_metadata:
                chunk["gateway"] = self.gateway
            chunks.append(chunk)
        return chunks

    def headers(self, expose_metadata: bool = True) -> dict[str, str]:
        base = {"x-gateway-request-id": self.request_id}
        if not expose_metadata:
            return base
        return {
            **base,
            "x-gateway-model": str(self.gateway.get("model")),
            "x-gateway-tier": str(self.gateway.get("tier")),
            "x-gateway-release": str(self.gateway.get("release")),
            "x-gateway-cost-usd": f"{self.gateway.get('cost_usd', 0.0):.8f}",
        }


class Gateway:
    def __init__(self, config: GatewayConfig, registry: ModelRegistry, store: Store, router: Router,
                 releases: ReleaseManager, budgets: BudgetPolicy, alerts: AlertMonitor | None = None, *,
                 rng: random.Random | None = None, clock: Callable[[], datetime] = utc_now):
        self.config = config
        self.registry = registry
        self.store = store
        self.router = router
        self.releases = releases
        self.budgets = budgets
        self.alerts = alerts
        self.rng = rng or random.Random()
        self.clock = clock

    def resolve_mode(self, requested: str) -> tuple[str, str | None]:
        """(mode, pinned model alias) for the request's ``model`` field."""
        key = (requested or "").strip()
        if key.lower() in MODE_ALIASES:
            return MODE_ALIASES[key.lower()] or self.config.routing.default_mode, None
        alias = self.config.resolve_model(key)
        if alias is None:
            choices = ", ".join([*[m for m in MODE_ALIASES if m and m != "gateway"], *self.config.models])
            raise GatewayError(404, f"unknown model {requested!r}; use one of: {choices}", param="model",
                               code="model_not_found")
        return "pinned", alias

    async def chat(self, request: ChatRequest, ctx: RequestContext) -> ChatResult:
        started = time.perf_counter()
        now = self.clock()
        record = RequestRecord(id=ctx.request_id, ts=iso(now), app_id=ctx.app_id, requested_model=request.model,
                               stream=request.stream)
        try:
            result = await self._run(request, ctx, record, now)
            record.status, record.http_status = "ok", 200
            return result
        except GatewayError as e:
            self._fail(record, e)
            raise
        except ProviderError as e:
            error = from_provider_error(e)
            self._fail(record, error, detail=upstream_detail(record.id, e))
            raise error from e
        except Exception as e:
            log.exception("request %s failed", ctx.request_id)
            error = GatewayError(500, "internal gateway error", type_="server_error", code="internal_error")
            self._fail(record, error, detail=f"{type(e).__name__}: {e}")
            raise error from e
        finally:
            self._finish(record, started)

    async def stream(
        self, request: ChatRequest, ctx: RequestContext, expose_metadata: bool = True
    ) -> tuple[AsyncIterator[str], dict[str, str]]:
        """Start a streamed answer: (SSE events, headers).

        A request that fails before its first chunk (invalid parameters, unknown model, budget, every
        provider failing, storage errors) raises a GatewayError and is logged here, once. A stream that
        started is logged once by its event generator when it ends.
        """
        started = time.perf_counter()
        now = self.clock()
        record = RequestRecord(id=ctx.request_id, ts=iso(now), app_id=ctx.app_id, requested_model=request.model,
                               stream=True)
        try:
            return await self._open_stream(request, ctx, record, now, started, expose_metadata)
        except GatewayError as e:
            if record.http_status is None:  # not recorded yet by the provider-failure paths
                self._fail(record, e)
            self._finish(record, started)
            raise
        except ProviderError as e:
            error = from_provider_error(e)
            self._fail(record, error, detail=upstream_detail(record.id, e))
            self._finish(record, started)
            raise error from e
        except asyncio.CancelledError:
            record.status, record.error_type = "error", "cancelled"
            record.error_message = "client disconnected or request cancelled before the first chunk"
            if record.calls and record.calls[-1].status == "ok" and record.calls[-1].latency_ms is None:
                record.calls[-1].status, record.calls[-1].error_type = "error", "cancelled"
            self._finish(record, started)
            raise
        except Exception as e:
            log.exception("streaming request %s failed", ctx.request_id)
            error = GatewayError(500, "internal gateway error", type_="server_error", code="internal_error")
            self._fail(record, error, detail=f"{type(e).__name__}: {e}")
            self._finish(record, started)
            raise error from e

    async def _open_stream(self, request: ChatRequest, ctx: RequestContext, record: RequestRecord, now: datetime,
                           started: float, expose_metadata: bool) -> tuple[AsyncIterator[str], dict[str, str]]:
        """Everything up to the first chunk; failures are raised for ``stream`` to log."""
        if request.n not in (None, 1):
            raise GatewayError(400, "only n=1 is supported", param="n")
        if request.tools or request.functions:
            raise GatewayError(400, "tools / function calling is not supported yet", param="tools")
        app_messages = [m.to_dict() for m in request.messages]
        if any(m["role"] == "tool" for m in app_messages):
            raise GatewayError(400, "tool messages are not supported yet", param="messages")
        mode, pinned = self.resolve_mode(request.model)
        notes: list[str] = []

        budget = self.budgets.status(ctx.app_id, self.store.app_spend(ctx.app_id, record.day))
        if budget.exceeded:
            err = GatewayError(
                429, f"daily budget for app {ctx.app_id!r} is used up: ${budget.spent:.4f} of "
                     f"${budget.limit:.2f} (resets at 00:00 UTC)",
                type_="budget_exceeded", code="budget_exceeded", retry_after=seconds_until_utc_midnight(now),
                counts_as_error=False)
            raise err
        if budget.soft and mode in ("router", "cascade", "strong"):
            notes.append(f"over {self.budgets.config.soft_limit_fraction:.0%} of the daily budget: cheap model")
            mode = "cheap"

        release_name, canary = self.releases.pick(self.releases.state(), request.user, self.rng)
        release = self.config.releases[release_name]
        record.release, record.canary, record.prompt_version = release_name, canary, release.prompt
        system_prompt = self.config.prompts[release.prompt]
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + app_messages

        latency = self.config.latency.class_for(ctx.app_id, ctx.latency)
        classifier = mode == "router" and self.router.name == "classifier"
        default_threshold = (self.config.routing.classifier_threshold if classifier
                             else self.config.routing.threshold)
        threshold = default_threshold if ctx.threshold is None else ctx.threshold
        record.mode, record.threshold, record.latency_class = ("pinned" if pinned else mode), threshold, latency
        if pinned:
            tier = "pinned"
        elif mode == "router":
            expected = (release.model_for("cheap", latency), release.model_for("strong", latency))
            trained = getattr(self.router, "model_aliases", None)
            if classifier and trained != expected:
                record.router = self.router.name
                notes.append(
                    f"classifier trained for {trained}; release {release_name} {latency} uses {expected}: strong model"
                )
                tier = "strong"
            else:
                t = time.perf_counter()
                score = self.router.score(app_messages)
                record.router_ms = round((time.perf_counter() - t) * 1000, 3)
                record.router, record.route_score = score.router, score.score
                notes.append(score.reason)
                tier = choose_tier(score.score, threshold)
        else:
            tier = "cheap" if mode == "cascade" else mode
        alias = pinned or release.model_for(tier, latency)
        params = request.params()

        allow_fallback = not pinned and (self.config.routing.fallback_on_error or mode == "cascade")
        chain = self.fallback_chain(tier, alias, release, latency) if allow_fallback else [(tier, alias)]

        active_gen = None
        active_tier = None
        active_alias = None
        first_chunk = None

        for i, (step_tier, step_alias) in enumerate(chain):
            spec = self.config.models[step_alias]
            call = CallRecord(seq=len(record.calls) + 1, tier=step_tier, model=step_alias, provider_model=spec.model)
            record.calls.append(call)
            t_call = time.perf_counter()
            try:
                upstream_gen = self.registry.stream(step_alias, messages, params)
                first_chunk = await anext(upstream_gen)
                call.latency_ms = round((time.perf_counter() - t_call) * 1000, 2)
                active_gen = upstream_gen
                active_tier = step_tier
                active_alias = step_alias
                break
            except ProviderError as e:
                call.latency_ms = round((time.perf_counter() - t_call) * 1000, 2)
                call.status = "error"
                call.error_type = e.kind
                call.error_message = str(e)[:500]
                if e.kind == BAD_REQUEST or i == len(chain) - 1:
                    err = from_provider_error(e)
                    self._fail(record, err, detail=upstream_detail(record.id, e))
                    raise err from e
                record.fallback = True
                continue
            except httpx.TransportError as e:
                call.latency_ms = round((time.perf_counter() - t_call) * 1000, 2)
                call.status = "error"
                call.error_type = OVERLOADED
                call.error_message = f"{type(e).__name__}: {e}"[:500]
                if i == len(chain) - 1:
                    err = GatewayError(503, "upstream model stream is unavailable", type_="upstream_error",
                                       code="upstream_unavailable")
                    self._fail(record, err, detail=f"{type(e).__name__}: {e}")
                    log.warning("request %s: upstream stream unreachable: %s", record.id, type(e).__name__)
                    raise err from e
                record.fallback = True
                continue
            except StopAsyncIteration:
                call.latency_ms = round((time.perf_counter() - t_call) * 1000, 2)
                active_gen = upstream_gen
                active_tier = step_tier
                active_alias = step_alias
                first_chunk = None
                break
            except Exception as e:
                call.latency_ms = round((time.perf_counter() - t_call) * 1000, 2)
                call.status = "error"
                call.error_type = "internal"
                call.error_message = str(e)[:500]
                if i == len(chain) - 1:
                    err = GatewayError(500, "internal gateway error", type_="server_error", code="internal_error")
                    self._fail(record, err, detail=f"{type(e).__name__}: {e}")
                    log.error("request %s: internal error before the first chunk: %s: %s", record.id,
                              type(e).__name__, e)
                    raise err from e
                record.fallback = True
                continue

        if active_gen is None:
            raise GatewayError(503, "all providers failed to stream", type_="upstream_error",
                               code="upstream_unavailable")

        if record.fallback:
            notes.append(f"fallback to {active_alias} ({active_tier}) after an error")

        provider_model = (first_chunk and first_chunk.model) or self.config.models[active_alias].model
        record.tier, record.model, record.provider_model = active_tier, active_alias, provider_model
        record.route_reason = "; ".join(notes)[:500] or None

        headers = {"x-gateway-request-id": record.id}
        if expose_metadata:
            headers["x-gateway-model"] = str(active_alias)
            headers["x-gateway-tier"] = str(active_tier)
            headers["x-gateway-release"] = str(record.release or "")

        spec = self.config.models[active_alias]
        prompt_chars = sum(len(message_text(m.get("content"))) for m in messages)

        async def event_generator() -> AsyncIterator[str]:
            created_ts = int(now.timestamp())
            accumulated_text: list[str] = []
            prompt_tokens = (first_chunk and first_chunk.prompt_tokens) or 0
            completion_tokens = (first_chunk and first_chunk.completion_tokens) or 0
            cost_usd = first_chunk and first_chunk.provider_cost_usd
            last_call = record.calls[-1]

            def account(finished: bool) -> None:
                """Tokens and cost of this call: the provider's usage when it sent any, otherwise estimates
                (about 4 characters per token). A stream that stops early is charged for what it produced."""
                nonlocal prompt_tokens, completion_tokens, cost_usd
                produced = "".join(accumulated_text)
                if prompt_tokens == 0:
                    prompt_tokens = max(1, prompt_chars // 4)
                if completion_tokens == 0 and (finished or produced):
                    completion_tokens = max(1, len(produced) // 4)
                if cost_usd is None:
                    cost_usd = spec.cost(prompt_tokens, completion_tokens)
                last_call.prompt_tokens = prompt_tokens
                last_call.completion_tokens = completion_tokens
                last_call.cost_usd = cost_usd

            try:
                if first_chunk is not None:
                    if first_chunk.text:
                        accumulated_text.append(first_chunk.text)
                    first_payload = {
                        "id": record.id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": provider_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": first_chunk.text},
                                "finish_reason": first_chunk.finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(first_payload)}\n\n"

                async for chunk in active_gen:
                    if chunk.text:
                        accumulated_text.append(chunk.text)
                    if chunk.prompt_tokens is not None:
                        prompt_tokens = chunk.prompt_tokens
                    if chunk.completion_tokens is not None:
                        completion_tokens = chunk.completion_tokens
                    if chunk.provider_cost_usd is not None:
                        cost_usd = chunk.provider_cost_usd

                    payload = {
                        "id": record.id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": provider_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.text} if chunk.text else {},
                                "finish_reason": chunk.finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                account(finished=True)
                last_call.status = "ok"

                record.prompt_tokens = prompt_tokens
                record.completion_tokens = completion_tokens
                record.cost_usd = cost_usd
                record.status = "ok"
                record.http_status = 200

                include_usage = bool(request.stream_options and request.stream_options.include_usage)
                if include_usage:
                    usage_payload = {
                        "id": record.id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": provider_model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                    if expose_metadata:
                        usage_payload["gateway"] = {
                            "request_id": record.id, "app_id": record.app_id, "release": record.release,
                            "canary": record.canary, "prompt_version": record.prompt_version, "mode": record.mode,
                            "router": record.router, "route_score": record.route_score, "threshold": threshold,
                            "latency_class": latency, "tier": active_tier, "model": active_alias,
                            "escalated": record.escalated, "fallback": record.fallback, "cost_usd": round(cost_usd, 8),
                            "router_ms": record.router_ms, "reason": record.route_reason,
                        }
                    yield f"data: {json.dumps(usage_payload)}\n\n"

                yield "data: [DONE]\n\n"

            except (asyncio.CancelledError, GeneratorExit):
                account(finished=False)
                last_call.status = "error"
                last_call.error_type = "cancelled"
                last_call.error_message = "client disconnected or request cancelled"
                record.status = "error"
                record.error_type = "cancelled"
                record.error_message = "client disconnected or request cancelled"
                raise
            except Exception as e:
                account(finished=False)
                detail = redact(str(e))[:500]
                log.warning("request %s: stream failed after the first chunk: %s", record.id, detail)
                last_call.status = "error"
                last_call.error_type = e.kind if isinstance(e, ProviderError) else type(e).__name__
                last_call.error_message = detail
                record.status = "error"
                record.http_status = 200
                record.error_type = "stream_failed"
                record.error_message = detail

                safe_err = {
                    "error": {
                        "message": "upstream stream failed",
                        "type": "upstream_error",
                        "code": "stream_failed",
                    }
                }
                yield f"data: {json.dumps(safe_err)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                self._finish(record, started)
                await active_gen.aclose()

        return event_generator(), headers

    async def _run(self, request: ChatRequest, ctx: RequestContext, record: RequestRecord,
                   now: datetime) -> ChatResult:
        if request.n not in (None, 1):
            raise GatewayError(400, "only n=1 is supported", param="n")
        if request.tools or request.functions:
            raise GatewayError(400, "tools / function calling is not supported yet", param="tools")
        app_messages = [m.to_dict() for m in request.messages]
        if any(m["role"] == "tool" for m in app_messages):
            raise GatewayError(400, "tool messages are not supported yet", param="messages")
        mode, pinned = self.resolve_mode(request.model)
        notes: list[str] = []

        budget = self.budgets.status(ctx.app_id, self.store.app_spend(ctx.app_id, record.day))
        if budget.exceeded:
            raise GatewayError(
                429, f"daily budget for app {ctx.app_id!r} is used up: ${budget.spent:.4f} of "
                     f"${budget.limit:.2f} (resets at 00:00 UTC)",
                type_="budget_exceeded", code="budget_exceeded", retry_after=seconds_until_utc_midnight(now),
                counts_as_error=False)
        if budget.soft and mode in ("router", "cascade", "strong"):
            notes.append(f"over {self.budgets.config.soft_limit_fraction:.0%} of the daily budget: cheap model")
            mode = "cheap"

        release_name, canary = self.releases.pick(self.releases.state(), request.user, self.rng)
        release = self.config.releases[release_name]
        record.release, record.canary, record.prompt_version = release_name, canary, release.prompt
        system_prompt = self.config.prompts[release.prompt]
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + app_messages

        latency = self.config.latency.class_for(ctx.app_id, ctx.latency)
        classifier = mode == "router" and self.router.name == "classifier"
        default_threshold = (self.config.routing.classifier_threshold if classifier
                             else self.config.routing.threshold)
        threshold = default_threshold if ctx.threshold is None else ctx.threshold
        record.mode, record.threshold, record.latency_class = ("pinned" if pinned else mode), threshold, latency
        if pinned:
            tier = "pinned"
        elif mode == "router":
            expected = (release.model_for("cheap", latency), release.model_for("strong", latency))
            trained = getattr(self.router, "model_aliases", None)
            if classifier and trained != expected:
                record.router = self.router.name
                notes.append(
                    f"classifier trained for {trained}; release {release_name} {latency} uses {expected}: strong model"
                )
                tier = "strong"
            else:
                t = time.perf_counter()
                score = self.router.score(app_messages)
                record.router_ms = round((time.perf_counter() - t) * 1000, 3)
                record.router, record.route_score = score.router, score.score
                notes.append(score.reason)
                tier = choose_tier(score.score, threshold)
        else:
            tier = "cheap" if mode == "cascade" else mode
        alias = pinned or release.model_for(tier, latency)
        params = request.params()

        allow_fallback = not pinned and (self.config.routing.fallback_on_error or mode == "cascade")
        completion, tier, alias = await self._call_with_fallback(record, tier, alias, release, latency, messages,
                                                                 params, allow_fallback)
        if record.fallback:
            notes.append(f"fallback to {alias} ({tier}) after an error")

        if mode == "cascade" and not pinned and tier == "cheap":
            check = check_answer(routing_text(app_messages), completion.text, completion.finish_reason)
            record.calls[-1].confidence, record.calls[-1].check_reason = check.confidence, check.reason
            notes.append(f"cheap answer: {check.reason} ({check.confidence:.2f})")
            if should_escalate(check, threshold):
                strong = release.model_for("strong", latency)
                try:
                    completion = await self._call(record, "strong", strong, messages, params)
                    tier, alias, record.escalated = "strong", strong, True
                except ProviderError as e:
                    notes.append(f"escalation failed ({e.kind}); kept the cheap answer")

        provider_model = completion.model or self.config.models[alias].model
        record.tier, record.model, record.provider_model = tier, alias, provider_model
        record.route_reason = "; ".join(notes)[:500] or None
        self._totals(record)
        gateway_info = {
            "request_id": record.id, "app_id": record.app_id, "release": record.release, "canary": record.canary,
            "prompt_version": record.prompt_version, "mode": record.mode, "router": record.router,
            "route_score": record.route_score, "threshold": threshold, "latency_class": latency, "tier": tier,
            "model": alias,
            "escalated": record.escalated, "fallback": record.fallback, "cost_usd": round(record.cost_usd, 8),
            "router_ms": record.router_ms, "reason": record.route_reason,
        }
        return ChatResult(record.id, int(now.timestamp()), provider_model, completion.text, completion.finish_reason,
                          record.prompt_tokens, record.completion_tokens, gateway_info)

    @staticmethod
    def fallback_chain(tier: str, alias: str, release: Release, latency: str) -> list[tuple[str, str]]:
        """(tier, model) to try in order: the chosen model, the same tier for the other latency
        class (another provider, same quality), then the other tier on both latency classes."""
        other_class = "batch" if latency == "interactive" else "interactive"
        other_tier = "strong" if tier == "cheap" else "cheap"
        chain = [(tier, alias), (tier, release.model_for(tier, other_class)),
                 (other_tier, release.model_for(other_tier, latency)),
                 (other_tier, release.model_for(other_tier, other_class))]
        unique: list[tuple[str, str]] = []
        for step in chain:
            if step[1] not in [a for _, a in unique]:
                unique.append(step)
        return unique

    async def _call_with_fallback(self, record: RequestRecord, tier: str, alias: str, release: Release,
                                  latency: str, messages: list[dict[str, Any]], params: dict[str, Any],
                                  allow_fallback: bool) -> tuple[Completion, str, str]:
        chain = self.fallback_chain(tier, alias, release, latency) if allow_fallback else [(tier, alias)]
        for i, (step_tier, step_alias) in enumerate(chain):
            try:
                return await self._call(record, step_tier, step_alias, messages, params), step_tier, step_alias
            except ProviderError as e:
                if e.kind == BAD_REQUEST or i == len(chain) - 1:
                    raise
                record.fallback = True
        raise AssertionError("unreachable")  # pragma: no cover

    async def _call(self, record: RequestRecord, tier: str, alias: str, messages: list[dict[str, Any]],
                    params: dict[str, Any]) -> Completion:
        spec = self.config.models[alias]
        call = CallRecord(seq=len(record.calls) + 1, tier=tier, model=alias, provider_model=spec.model)
        record.calls.append(call)
        started = time.perf_counter()
        try:
            completion = await self.registry.complete(alias, messages, params)
        except Exception as e:
            call.status = "error"
            call.error_type = e.kind if isinstance(e, ProviderError) else "internal"
            call.error_message = str(e)[:500]
            raise
        finally:
            call.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        call.prompt_tokens, call.completion_tokens = completion.prompt_tokens, completion.completion_tokens
        # The provider's own figure when it reports one (OpenRouter), else list prices from the config.
        call.cost_usd = (completion.provider_cost_usd if completion.provider_cost_usd is not None
                         else spec.cost(completion.prompt_tokens, completion.completion_tokens))
        call.provider_model = completion.model or spec.model
        return completion

    @staticmethod
    def _totals(record: RequestRecord) -> None:
        record.prompt_tokens = sum(c.prompt_tokens for c in record.calls)
        record.completion_tokens = sum(c.completion_tokens for c in record.calls)
        record.cost_usd = sum(c.cost_usd for c in record.calls)

    def _finish(self, record: RequestRecord, started: float) -> None:
        """Log a finished request (once) and check alerts. Neither may break the response: a storage or
        alert failure goes to the gateway's log, with what the lost row would have said."""
        record.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self._totals(record)
        try:
            self.store.log_request(record)
        except Exception:
            log.exception("could not log request %s (status %s, http %s, error %s, cost $%.8f)", record.id,
                          record.status, record.http_status, record.error_type, record.cost_usd)
        if self.alerts is not None:
            try:
                self.alerts.maybe_check()
            except Exception:
                log.exception("alert check failed")

    @staticmethod
    def _fail(record: RequestRecord, error: GatewayError, detail: str | None = None) -> None:
        record.status = "error" if error.counts_as_error else "rejected"
        record.http_status = error.status
        record.error_type = error.code or error.type
        record.error_message = (detail or error.message)[:500]
