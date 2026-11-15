"""Provider interface, the error type every adapter raises, and error parsing."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ProviderError.kind values
RATE_LIMIT = "rate_limit"  # per-minute limit: retry after a wait
QUOTA_EXHAUSTED = "quota_exhausted"  # daily quota: no retry until it resets
OVERLOADED = "overloaded"  # 5xx / network: retry with backoff
TIMEOUT = "timeout"
AUTH = "auth"  # missing or rejected API key
BAD_REQUEST = "bad_request"  # the request itself is wrong; never retried
OTHER = "other"

RETRYABLE = (RATE_LIMIT, OVERLOADED, TIMEOUT)


@dataclass
class Completion:
    """One model answer, normalised across providers."""

    text: str
    finish_reason: str = "stop"  # stop | length | content_filter
    prompt_tokens: int = 0
    completion_tokens: int = 0  # includes reasoning ("thinking") tokens, which are billed as output
    model: str = ""  # provider model name
    latency_ms: float | None = None  # the successful provider call only (no pacing waits or retries)


class ProviderError(Exception):
    """A failed provider call, classified so callers know whether to retry."""

    def __init__(self, message: str, *, kind: str = OTHER, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after

    def __str__(self) -> str:
        return f"{self.kind}: {super().__str__()}"


class Provider(ABC):
    """An LLM API. ``complete`` takes OpenAI-style messages and returns a Completion."""

    name: str = "provider"

    @abstractmethod
    async def complete(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> Completion:
        """Run one chat completion; raise ProviderError on failure."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""


# ── messages ────────────────────────────────────────────────────────

def message_text(content: Any) -> str:
    """The text of an OpenAI message ``content`` (a string or a list of parts).

    Raises ProviderError(bad_request) for non-text parts (images, audio):
    the gateway's providers are used for text only.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            else:
                kind = part.get("type") if isinstance(part, dict) else type(part).__name__
                raise ProviderError(f"unsupported message content part: {kind!r} (text only)", kind=BAD_REQUEST,
                                    status=400)
        return "\n".join(texts)
    raise ProviderError(f"unsupported message content: {type(content).__name__}", kind=BAD_REQUEST, status=400)


# ── errors ──────────────────────────────────────────────────────────

_RETRY_AFTER = (
    re.compile(r"try again in (?:(\d+)m)?([\d.]+)s"),  # Groq: "Please try again in 1m43.2s"
    re.compile(r"retry in ([\d.]+)s"),  # Gemini: "Please retry in 33.6s."
    re.compile(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s"),  # Gemini RetryInfo: "retryDelay": "33s"
)
# Groq: "tokens per day (TPD)"; Gemini quota IDs: "GenerateRequestsPerDayPerProjectPerModel"
_PER_DAY = re.compile(r"per ?day|\bTPD\b|\bRPD\b", flags=re.IGNORECASE)


def retry_after_seconds(text: str) -> float | None:
    """The wait a provider suggests in an error message, in seconds."""
    groq = _RETRY_AFTER[0].search(text)
    if groq:
        return int(groq.group(1) or 0) * 60 + float(groq.group(2))
    for pattern in _RETRY_AFTER[1:]:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def classify_http_error(status: int, body: str, retry_after_header: str | None = None) -> tuple[str, float | None]:
    """(kind, retry_after) for an HTTP error from an LLM API."""
    retry_after = retry_after_seconds(body)
    if retry_after is None and retry_after_header:
        try:
            retry_after = float(retry_after_header)
        except ValueError:
            retry_after = None
    if status == 429 or "RESOURCE_EXHAUSTED" in body:
        return (QUOTA_EXHAUSTED if _PER_DAY.search(body) else RATE_LIMIT), retry_after
    if status in (401, 403):
        return AUTH, None
    if status == 408:
        return TIMEOUT, retry_after
    if status in (400, 404, 409, 413, 422):
        return BAD_REQUEST, None
    if status >= 500:
        return OVERLOADED, retry_after
    return OTHER, retry_after


# Account identifiers and key-like strings that providers put in error messages.
_REDACT = (
    (re.compile(r"\borg_[A-Za-z0-9]+"), "org_[redacted]"),  # Groq organization id
    (re.compile(r"\bprojects/\d+"), "projects/[redacted]"),  # Google Cloud project number
    (re.compile(r"\b(?:gsk_|sk-)[A-Za-z0-9_-]{8,}"), "[redacted key]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"), "[redacted key]"),
)


def redact(text: str) -> str:
    """``text`` without account ids or keys, so errors can be logged, stored and returned to clients."""
    for pattern, replacement in _REDACT:
        text = pattern.sub(replacement, text)
    return text


def error_message(body: str, limit: int = 500) -> str:
    """A short, redacted message from an error body (the JSON ``error.message`` when present)."""
    import json

    try:
        data = json.loads(body)
    except ValueError:
        return redact(body[:limit])
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return redact(str(error["message"])[:limit])
    return redact(body[:limit])
