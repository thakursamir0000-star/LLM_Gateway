import asyncio
import json

import httpx
import pytest

from gateway.providers import ModelRegistry, build_providers
from gateway.providers.base import (
    AUTH,
    BAD_REQUEST,
    OVERLOADED,
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    TIMEOUT,
    Completion,
    ProviderError,
    classify_http_error,
    message_text,
    retry_after_seconds,
)
from gateway.providers.fake import FakeProvider
from gateway.providers.gemini import GeminiProvider, build_request
from gateway.providers.gemini import parse_response as parse_gemini
from gateway.providers.groq import GroqProvider

USER = [{"role": "user", "content": "What is 2+2?"}]


def run(coro):
    return asyncio.run(coro)


def transport(handler):
    """A mock transport that records requests and answers with ``handler(request)``."""
    seen = []

    def wrapped(request: httpx.Request):
        seen.append(request)
        return handler(request)

    return httpx.MockTransport(wrapped), seen


# ── error parsing ───────────────────────────────────────────────────

def test_retry_after_parsing():
    assert retry_after_seconds("Please try again in 1m2.5s. Visit") == pytest.approx(62.5)
    assert retry_after_seconds("Please retry in 33.6s.") == pytest.approx(33.6)
    assert retry_after_seconds('"retryDelay": "17s"') == 17
    assert retry_after_seconds("nothing here") is None


@pytest.mark.parametrize("status, body, kind", [
    (429, "Rate limit reached on tokens per minute (TPM): Limit 8000", RATE_LIMIT),
    (429, "Rate limit reached on tokens per day (TPD): Limit 200000", QUOTA_EXHAUSTED),
    (429, '{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}', QUOTA_EXHAUSTED),
    (401, "Invalid API Key", AUTH),
    (400, "bad", BAD_REQUEST),
    (503, "UNAVAILABLE", OVERLOADED),
    (408, "", TIMEOUT),
])
def test_classify_http_error(status, body, kind):
    assert classify_http_error(status, body)[0] == kind


def test_retry_after_header_is_used_when_body_has_none():
    assert classify_http_error(429, "slow down", "7") == (RATE_LIMIT, 7.0)


def test_message_text_accepts_text_parts_and_rejects_images():
    assert message_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    with pytest.raises(ProviderError) as info:
        message_text([{"type": "image_url", "image_url": {"url": "x"}}])
    assert info.value.kind == BAD_REQUEST


# ── Groq ────────────────────────────────────────────────────────────

GROQ_OK = {
    "model": "openai/gpt-oss-20b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
}


def test_groq_success_and_request_shape():
    mock, seen = transport(lambda r: httpx.Response(200, json=GROQ_OK))
    provider = GroqProvider("k-test", "https://groq.test/openai/v1", transport=mock)
    params = {"temperature": 0.2, "max_tokens": 50, "reasoning_effort": "low", "logit_bias": {"1": 2}}
    completion = run(provider.complete("openai/gpt-oss-20b", USER, params))
    assert completion == Completion("4", "stop", 12, 30, "openai/gpt-oss-20b")
    request = seen[0]
    assert request.url == "https://groq.test/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k-test"
    body = json.loads(request.content)
    assert body["max_completion_tokens"] == 50 and "max_tokens" not in body
    assert body["reasoning_effort"] == "low" and body["temperature"] == 0.2
    assert "logit_bias" not in body  # not passed through


def test_groq_rate_limit_error_is_classified():
    message = {"error": {"message": "Rate limit reached for model on tokens per minute (TPM). "
                                    "Please try again in 7.5s."}}
    mock, _ = transport(lambda r: httpx.Response(429, json=message))
    provider = GroqProvider("k", transport=mock)
    with pytest.raises(ProviderError) as info:
        run(provider.complete("m", USER, {}))
    assert info.value.kind == RATE_LIMIT and info.value.retry_after == pytest.approx(7.5)
    assert "tokens per minute" in str(info.value)


def test_groq_missing_key_and_timeout():
    with pytest.raises(ProviderError) as info:
        run(GroqProvider(None).complete("m", USER, {}))
    assert info.value.kind == AUTH

    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    mock, _ = transport(timeout)
    with pytest.raises(ProviderError) as info:
        run(GroqProvider("k", transport=mock).complete("m", USER, {}))
    assert info.value.kind == TIMEOUT


# ── Gemini ──────────────────────────────────────────────────────────

CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "Part 1"},
    {"role": "user", "content": "Part 2"},
]


def test_gemini_request_uses_system_instruction_and_model_role():
    body = build_request("gemini-4-hydrogen", CHAT, {"temperature": 0.1, "max_tokens": 64, "stop": "END"})
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][2]["parts"][0]["text"] == "Part 1\n\nPart 2"  # consecutive turns merged
    assert body["generationConfig"] == {"temperature": 0.1, "maxOutputTokens": 64, "stopSequences": ["END"]}


def test_gemma_gets_system_text_in_the_first_user_turn():
    body = build_request("gemma-4-31b-it", CHAT, {})
    assert "systemInstruction" not in body
    assert body["contents"][0]["parts"][0]["text"] == "Be brief.\n\nHi"
    assert "generationConfig" not in body


def test_gemini_rejects_requests_without_turns():
    with pytest.raises(ProviderError) as info:
        build_request("gemini-x", [{"role": "system", "content": "x"}], {})
    assert info.value.kind == BAD_REQUEST


def test_gemini_response_drops_thoughts_and_counts_their_tokens():
    data = {
        "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "4"}]},
                        "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "thoughtsTokenCount": 40},
        "modelVersion": "gemini-4-hydrogen",
    }
    assert parse_gemini(data, "x") == Completion("4", "length", 10, 42, "gemini-4-hydrogen")


def test_gemini_blocked_prompt_is_a_content_filter_finish():
    data = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 5}}
    completion = parse_gemini(data, "gemini-x")
    assert completion.text == "" and completion.finish_reason == "content_filter"


def test_gemini_call_and_daily_quota_error():
    ok = {"candidates": [{"content": {"parts": [{"text": "4"}]}, "finishReason": "STOP"}],
          "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1}}
    mock, seen = transport(lambda r: httpx.Response(200, json=ok))
    provider = GeminiProvider("g-test", "https://gemini.test/v1beta", transport=mock)
    assert run(provider.complete("gemma-4-31b-it", USER, {})).text == "4"
    assert seen[0].url == "https://gemini.test/v1beta/models/gemma-4-31b-it:generateContent"
    assert seen[0].headers["x-goog-api-key"] == "g-test"

    quota = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota.",
                       "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                                   {"retryDelay": "40s"}]}}
    mock, _ = transport(lambda r: httpx.Response(429, json=quota))
    with pytest.raises(ProviderError) as info:
        run(GeminiProvider("g", transport=mock).complete("gemini-x", USER, {}))
    assert info.value.kind == QUOTA_EXHAUSTED and info.value.retry_after == 40


# ── registry: params, retries, pacing ───────────────────────────────

class Sleeper:
    def __init__(self):
        self.waits = []

    async def __call__(self, seconds):
        self.waits.append(seconds)


def failing_then(*outcomes):
    """A fake script that raises/returns ``outcomes`` in order."""
    items = list(outcomes)

    def script(model, messages, params):
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return script


def test_registry_merges_default_params(config):
    fake = FakeProvider("groq")
    registry = ModelRegistry(config, {"groq": fake, "gemini": FakeProvider("gemini")})
    run(registry.complete("small", USER, {"temperature": 0.3, "top_p": None}))
    assert fake.calls[0]["model"] == "openai/gpt-oss-20b"
    assert fake.calls[0]["params"] == {"reasoning_effort": "low", "temperature": 0.3}
    run(registry.complete("small", USER, {"reasoning_effort": "high"}))
    assert fake.calls[1]["params"]["reasoning_effort"] == "high"


def test_registry_retries_overloaded_with_backoff(config):
    ok = Completion("fine", "stop", 1, 1, "m")
    script = failing_then(ProviderError("busy", kind=OVERLOADED), ProviderError("busy", kind=OVERLOADED), ok)
    sleeper = Sleeper()
    registry = ModelRegistry(config, {"groq": FakeProvider("groq", script), "gemini": FakeProvider()}, sleep=sleeper)
    assert run(registry.complete("small", USER, {})).text == "fine"
    assert sleeper.waits == [1.0, 2.0]


def test_registry_waits_the_suggested_delay_for_rate_limits(config):
    ok = Completion("fine", "stop", 1, 1, "m")
    script = failing_then(ProviderError("slow down", kind=RATE_LIMIT, retry_after=6.0), ok)
    sleeper = Sleeper()
    registry = ModelRegistry(config, {"groq": FakeProvider("groq", script), "gemini": FakeProvider()}, sleep=sleeper)
    run(registry.complete("small", USER, {}))
    assert sleeper.waits == [6.0]


@pytest.mark.parametrize("error", [
    ProviderError("daily", kind=QUOTA_EXHAUSTED),
    ProviderError("bad", kind=BAD_REQUEST),
    ProviderError("key", kind=AUTH),
    ProviderError("too long", kind=RATE_LIMIT, retry_after=600),  # longer than max_wait
])
def test_registry_fails_fast_when_retrying_cannot_help(config, error):
    sleeper = Sleeper()
    registry = ModelRegistry(config, {"groq": FakeProvider("groq", failing_then(error)), "gemini": FakeProvider()},
                             sleep=sleeper)
    with pytest.raises(ProviderError):
        run(registry.complete("small", USER, {}))
    assert sleeper.waits == []


def test_registry_gives_up_after_the_last_attempt(config):
    errors = [ProviderError("busy", kind=OVERLOADED) for _ in range(3)]
    registry = ModelRegistry(config, {"groq": FakeProvider("groq", failing_then(*errors)), "gemini": FakeProvider()},
                             sleep=Sleeper())
    with pytest.raises(ProviderError) as info:
        run(registry.complete("small", USER, {}))
    assert info.value.kind == OVERLOADED


def test_registry_turns_a_long_local_queue_into_a_rate_limit(config):
    from tests.conftest import make_config

    paced = make_config(models={"small": {"provider": "groq", "model": "m", "limits": {"rpm": 1}}})
    registry = ModelRegistry(paced, {"groq": FakeProvider("groq"), "gemini": FakeProvider()}, max_wait=1)
    run(registry.complete("small", USER, {}))
    with pytest.raises(ProviderError) as info:
        run(registry.complete("small", USER, {}))
    assert info.value.kind == RATE_LIMIT and info.value.retry_after > 1


def test_build_providers(config, monkeypatch):
    monkeypatch.setenv("TEST_GROQ_KEY", "k")
    providers = build_providers(config)
    assert isinstance(providers["groq"], GroqProvider) and isinstance(providers["gemini"], GeminiProvider)
    fakes = build_providers(config, fake=True)
    assert all(isinstance(p, FakeProvider) for p in fakes.values())


def test_registry_without_pacing_ignores_model_limits(config):
    from tests.conftest import make_config

    paced = make_config(models={"small": {"provider": "groq", "model": "m", "limits": {"rpm": 1}}})
    registry = ModelRegistry(paced, {"groq": FakeProvider("groq"), "gemini": FakeProvider()}, max_wait=1,
                             pacing=False)
    for _ in range(3):
        run(registry.complete("small", USER, {}))


def test_registry_records_the_model_time_of_the_successful_call(config):
    registry = ModelRegistry(config, {"groq": FakeProvider("groq", latency_s=0.02), "gemini": FakeProvider()})
    completion = run(registry.complete("small", USER, {}))
    assert completion.latency_ms is not None and completion.latency_ms >= 5


def test_error_messages_hide_account_ids_and_keys():
    from gateway.providers.base import error_message, redact

    groq = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` in organization "
                                 "`org_01abcDEF234` service tier `on_demand` on tokens per day (TPD)"}}
    message = error_message(json.dumps(groq))
    assert "org_01abcDEF234" not in message and "org_[redacted]" in message and "tokens per day" in message
    assert redact("key gsk_abcdefghijklmnop and AIzaSyA1234567890abcdefghij") == "key [redacted key] and [redacted key]"
    assert redact("quota for projects/123456789 exceeded") == "quota for projects/[redacted] exceeded"


def test_gateway_errors_returned_to_clients_are_redacted():
    body = {"error": {"message": "Limit reached in organization `org_secret99` on tokens per day (TPD)"}}
    mock, _ = transport(lambda r: httpx.Response(429, json=body))
    with pytest.raises(ProviderError) as info:
        run(GroqProvider("k", transport=mock).complete("m", USER, {}))
    assert "org_secret99" not in str(info.value) and info.value.kind == QUOTA_EXHAUSTED
