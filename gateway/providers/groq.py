"""Groq adapter (OpenAI-compatible chat completions API)."""

from __future__ import annotations

from typing import Any

import httpx

from gateway.providers.base import (
    AUTH,
    BAD_REQUEST,
    OTHER,
    OVERLOADED,
    TIMEOUT,
    Completion,
    Provider,
    ProviderError,
    classify_http_error,
    error_message,
    message_text,
)

# Request fields passed through to Groq; anything else is dropped.
_PASSTHROUGH = ("temperature", "top_p", "stop", "seed", "presence_penalty", "frequency_penalty",
                "reasoning_effort", "response_format")


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, api_key: str | None, base_url: str = "https://api.groq.com/openai/v1",
                 timeout_s: float = 60.0, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport)

    def build_body(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m["role"], "content": message_text(m.get("content"))} for m in messages],
        }
        for key in _PASSTHROUGH:
            if params.get(key) is not None:
                body[key] = params[key]
        max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")
        if max_tokens:
            body["max_completion_tokens"] = int(max_tokens)
        return body

    async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        if not self._api_key:
            raise ProviderError("GROQ_API_KEY is not set", kind=AUTH)
        body = self.build_body(model, messages, params)
        try:
            response = await self._client.post(
                "/chat/completions", json=body, headers={"Authorization": f"Bearer {self._api_key}"})
        except httpx.TimeoutException as e:
            raise ProviderError(f"Groq timed out: {type(e).__name__}", kind=TIMEOUT) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Groq unreachable: {type(e).__name__}: {e}", kind=OVERLOADED) from e
        if response.status_code >= 400:
            kind, retry_after = classify_http_error(
                response.status_code, response.text, response.headers.get("retry-after"))
            raise ProviderError(f"Groq HTTP {response.status_code}: {error_message(response.text)}",
                                kind=kind, status=response.status_code, retry_after=retry_after)
        return parse_response(response.json(), model)

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_response(data: dict[str, Any], model: str) -> Completion:
    """A Completion from an OpenAI-style chat completion response."""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"unexpected response shape: {str(data)[:200]}", kind=OTHER) from e
    message = choice.get("message") or {}
    if message.get("tool_calls"):
        raise ProviderError("tool calls are not supported by the gateway yet", kind=BAD_REQUEST, status=400)
    usage = data.get("usage") or {}
    return Completion(
        text=message.get("content") or "",
        finish_reason=choice.get("finish_reason") or "stop",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        model=data.get("model") or model,
    )
