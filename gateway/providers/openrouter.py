"""OpenRouter adapter (OpenAI-compatible API that forwards to many model hosts).

Differences from a plain OpenAI-compatible API:
- reasoning effort is sent as ``reasoning: {"effort": ...}``;
- ``provider`` routing preferences (which hosts, quantizations, fallbacks) pass
  through from the model's config params, so a benchmark can pin one host;
- every request asks for usage accounting, so the response carries the real
  cost of the call, and the host that served it is recorded;
- 402 means the account is out of credits and the free models' daily cap is a
  per-day 429: neither is retried. 403 is a moderation refusal (a bad request);
- an error can also come back inside an HTTP 200 body.
"""

from __future__ import annotations

from typing import Any

from gateway.providers.base import (
    AUTH,
    BAD_REQUEST,
    OTHER,
    OVERLOADED,
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    TIMEOUT,
    Completion,
    ProviderError,
    redact,
)
from gateway.providers.openai_compat import OpenAICompatibleProvider, parse_response

# Error codes OpenRouter puts in a response body
_BODY_ERRORS = {400: BAD_REQUEST, 401: AUTH, 402: QUOTA_EXHAUSTED, 403: BAD_REQUEST, 408: TIMEOUT,
                429: RATE_LIMIT, 502: OVERLOADED, 503: OVERLOADED}


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    label = "OpenRouter"
    key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"
    passthrough = ("temperature", "top_p", "stop", "seed", "presence_penalty", "frequency_penalty",
                   "response_format", "provider")

    def headers(self) -> dict[str, str]:
        return {**super().headers(), "X-Title": "llm-gateway"}

    def build_body(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        body = super().build_body(model, messages, params)
        if params.get("reasoning_effort"):
            body["reasoning"] = {"effort": params["reasoning_effort"]}
        body["usage"] = {"include": True}
        return body

    def classify(self, status: int, body: str, retry_after_header: str | None) -> tuple[str, float | None]:
        kind, retry_after = super().classify(status, body, retry_after_header)
        if status == 402:
            return QUOTA_EXHAUSTED, None
        if status == 403:
            return BAD_REQUEST, None
        return kind, retry_after

    def parse(self, data: dict[str, Any], model: str) -> Completion:
        error = data.get("error")
        choices = data.get("choices") or []
        if not error and choices and isinstance(choices[0], dict):
            error = choices[0].get("error")
        if isinstance(error, dict):
            code = error.get("code")
            kind = _BODY_ERRORS.get(code, OTHER) if isinstance(code, int) else OTHER
            raise ProviderError(f"OpenRouter error {code}: {redact(str(error.get('message', ''))[:500])}", kind=kind,
                                status=code if isinstance(code, int) else None)
        completion = parse_response(data, model)
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        completion.provider_cost_usd = float(cost) if isinstance(cost, int | float) else None
        completion.served_by = data.get("provider")
        return completion
