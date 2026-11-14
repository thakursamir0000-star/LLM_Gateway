"""Gemini API adapter (native ``generateContent`` REST API), for Gemini and Gemma models.

OpenAI messages are converted to Gemini ``contents``:
- ``assistant`` becomes role ``model``; consecutive turns of one role are merged;
- system/developer messages become ``systemInstruction``, except for Gemma
  models, which reject system instructions: there the text is put before the
  first user message instead;
- reasoning ("thought") parts are dropped from the answer, and their tokens
  count as output tokens (that is how they are billed).
"""

from __future__ import annotations

from typing import Any

import httpx

from gateway.providers.base import (
    AUTH,
    BAD_REQUEST,
    OVERLOADED,
    TIMEOUT,
    Completion,
    Provider,
    ProviderError,
    classify_http_error,
    error_message,
    message_text,
)

_FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter",
           "BLOCKLIST": "content_filter", "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter"}


def supports_system_instruction(model: str) -> bool:
    return not model.lower().startswith("gemma")


def build_request(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """The generateContent body for OpenAI-style messages and parameters."""
    system: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        text = message_text(message.get("content"))
        if role in ("system", "developer"):
            if text:
                system.append(text)
            continue
        if role not in ("user", "assistant"):
            raise ProviderError(f"unsupported message role for Gemini: {role!r}", kind=BAD_REQUEST, status=400)
        gemini_role = "model" if role == "assistant" else "user"
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})
    if not contents:
        raise ProviderError("the request has no user or assistant messages", kind=BAD_REQUEST, status=400)

    body: dict[str, Any] = {"contents": contents}
    if system:
        system_text = "\n\n".join(system)
        if supports_system_instruction(model):
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        else:
            first_user = next((c for c in contents if c["role"] == "user"), None)
            if first_user is None:
                contents.insert(0, {"role": "user", "parts": [{"text": system_text}]})
            else:
                first_user["parts"][0]["text"] = system_text + "\n\n" + first_user["parts"][0]["text"]

    config: dict[str, Any] = {}
    if params.get("temperature") is not None:
        config["temperature"] = params["temperature"]
    if params.get("top_p") is not None:
        config["topP"] = params["top_p"]
    max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")
    if max_tokens:
        config["maxOutputTokens"] = int(max_tokens)
    stop = params.get("stop")
    if stop:
        config["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    if params.get("seed") is not None:
        config["seed"] = params["seed"]
    response_format = params.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


def parse_response(data: dict[str, Any], model: str) -> Completion:
    usage = data.get("usageMetadata") or {}
    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    completion_tokens = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
    candidates = data.get("candidates") or []
    if not candidates:  # the prompt itself was blocked
        return Completion("", "content_filter", prompt_tokens, completion_tokens, data.get("modelVersion") or model)
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought"))
    finish = _FINISH.get(candidate.get("finishReason") or "STOP", "stop")
    return Completion(text, finish, prompt_tokens, completion_tokens, data.get("modelVersion") or model)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str | None, base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 timeout_s: float = 120.0, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport)

    async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        if not self._api_key:
            raise ProviderError("GEMINI_API_KEY is not set", kind=AUTH)
        body = build_request(model, messages, params)
        try:
            response = await self._client.post(
                f"/models/{model}:generateContent", json=body, headers={"x-goog-api-key": self._api_key})
        except httpx.TimeoutException as e:
            raise ProviderError(f"Gemini timed out: {type(e).__name__}", kind=TIMEOUT) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Gemini unreachable: {type(e).__name__}: {e}", kind=OVERLOADED) from e
        if response.status_code >= 400:
            kind, retry_after = classify_http_error(
                response.status_code, response.text, response.headers.get("retry-after"))
            raise ProviderError(f"Gemini HTTP {response.status_code}: {error_message(response.text)}",
                                kind=kind, status=response.status_code, retry_after=retry_after)
        return parse_response(response.json(), model)

    async def aclose(self) -> None:
        await self._client.aclose()
