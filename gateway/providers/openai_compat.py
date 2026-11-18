"""Shared adapter for OpenAI-compatible chat completions APIs (Groq, OpenRouter).

Subclasses set their name, the key's environment variable, the default base
URL and the request fields they pass through, and can adjust the request
body, the error classification and the response parsing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
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
    StreamChunk,
    classify_http_error,
    error_message,
    message_text,
)


class OpenAICompatibleProvider(Provider):
    name = "openai-compatible"
    label = "OpenAI-compatible API"  # used in error messages
    key_env = "API key"  # named in the "not set" error
    default_base_url = ""
    # Request fields passed through; anything else is dropped.
    passthrough: tuple[str, ...] = ("temperature", "top_p", "stop", "seed", "presence_penalty", "frequency_penalty",
                                    "reasoning_effort", "response_format")

    def __init__(self, api_key: str | None, base_url: str | None = None, timeout_s: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=(base_url or self.default_base_url).rstrip("/"),
                                         timeout=timeout_s, transport=transport)

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def build_body(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m["role"], "content": message_text(m.get("content"))} for m in messages],
        }
        for key in self.passthrough:
            if params.get(key) is not None:
                body[key] = params[key]
        max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")
        if max_tokens:
            body["max_completion_tokens"] = int(max_tokens)
        return body

    def classify(self, status: int, body: str, retry_after_header: str | None) -> tuple[str, float | None]:
        return classify_http_error(status, body, retry_after_header)

    def parse(self, data: dict[str, Any], model: str) -> Completion:
        return parse_response(data, model)

    def parse_chunk(self, data: dict[str, Any], model: str) -> StreamChunk | None:
        return parse_chunk_response(data, model)

    async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        if not self._api_key:
            raise ProviderError(f"{self.key_env} is not set", kind=AUTH)
        body = self.build_body(model, messages, params)
        try:
            response = await self._client.post("/chat/completions", json=body, headers=self.headers())
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.label} timed out: {type(e).__name__}", kind=TIMEOUT) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.label} unreachable: {type(e).__name__}", kind=OVERLOADED) from e
        if response.status_code >= 400:
            kind, retry_after = self.classify(response.status_code, response.text, response.headers.get("retry-after"))
            raise ProviderError(f"{self.label} HTTP {response.status_code}: {error_message(response.text)}",
                                kind=kind, status=response.status_code, retry_after=retry_after)
        return self.parse(response.json(), model)

    async def stream(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        if not self._api_key:
            raise ProviderError(f"{self.key_env} is not set", kind=AUTH)
        body = self.build_body(model, messages, params)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        req = self._client.build_request("POST", "/chat/completions", json=body, headers=self.headers())
        try:
            response = await self._client.send(req, stream=True)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.label} timed out: {type(e).__name__}", kind=TIMEOUT) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.label} unreachable: {type(e).__name__}", kind=OVERLOADED) from e

        try:
            if response.status_code >= 400:
                body_bytes = await response.aread()
                text = body_bytes.decode("utf-8", errors="replace")
                kind, retry_after = self.classify(response.status_code, text, response.headers.get("retry-after"))
                raise ProviderError(f"{self.label} HTTP {response.status_code}: {error_message(text)}",
                                    kind=kind, status=response.status_code, retry_after=retry_after)

            async for line in response.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    raw_data = line[len("data: "):].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
                    chunk = self.parse_chunk(data, model)
                    if chunk is not None:
                        yield chunk
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_chunk_response(data: dict[str, Any], model: str) -> StreamChunk | None:
    """A StreamChunk from an OpenAI-style chunk."""
    choices = data.get("choices") or []
    usage = data.get("usage")
    prompt_tokens = int(usage["prompt_tokens"]) if usage and usage.get("prompt_tokens") is not None else None
    completion_tokens = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") is not None else None
    resp_model = data.get("model") or model
    if not choices:
        if usage:
            return StreamChunk(model=resp_model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return None
    choice = choices[0]
    delta = choice.get("delta") or {}
    if delta.get("tool_calls"):
        raise ProviderError("tool calls are not supported by the gateway yet", kind=BAD_REQUEST, status=400)
    text = delta.get("content") or ""
    finish = choice.get("finish_reason")
    return StreamChunk(text=text, finish_reason=finish, model=resp_model,
                       prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


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


__all__ = ["OpenAICompatibleProvider", "parse_chunk_response", "parse_response"]
