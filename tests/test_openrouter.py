"""OpenRouter provider, provider selection, separate resumable answer files and the provider comparison.

No real API calls: the adapter is tested against an httpx mock transport.
"""

import asyncio
import json
import random
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import DEFAULT_CONFIG, load_config
from gateway.monitoring.store import Store
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
)
from gateway.providers.fake import FakeProvider
from gateway.providers.openrouter import OpenRouterProvider
from gateway.routing import RulesRouter
from router_lab.collect import collect
from router_lab.compare import agreement, common_questions, latest_answers, model_stats, wilson
from router_lab.progress import AnswerLog, AnswerLogLocked
from router_lab.questions import Question, sample
from tests.conftest import make_config

USER = [{"role": "user", "content": "What is 2+2?"}]
OR_OK = {
    "id": "gen-1", "model": "openai/gpt-oss-20b", "provider": "DeepInfra",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer: 4"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100, "cost": 0.0000096},
}
PIN = {"only": ["deepinfra/bf16"], "allow_fallbacks": False}


def run(coro):
    return asyncio.run(coro)


def provider_with(handler, key="sk-or-v1-test"):
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    return OpenRouterProvider(key, "https://openrouter.test/api/v1", transport=httpx.MockTransport(wrapped)), seen


def failing(status, message, headers=None):
    body = {"error": {"code": status, "message": message}}
    return provider_with(lambda r: httpx.Response(status, json=body, headers=headers or {}))[0]


def error_of(provider):
    with pytest.raises(ProviderError) as info:
        run(provider.complete("openai/gpt-oss-20b", USER, {}))
    return info.value


# ── adapter ─────────────────────────────────────────────────────────

def test_authentication_and_request_shape():
    provider, seen = provider_with(lambda r: httpx.Response(200, json=OR_OK))
    run(provider.complete("openai/gpt-oss-20b", USER,
                          {"reasoning_effort": "low", "provider": PIN, "max_completion_tokens": 64, "seed": 1,
                           "logit_bias": {"1": 1}}))
    request = seen[0]
    assert request.url == "https://openrouter.test/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-or-v1-test"
    assert request.headers["x-title"] == "llm-gateway"
    body = json.loads(request.content)
    assert body["reasoning"] == {"effort": "low"} and "reasoning_effort" not in body
    assert body["provider"] == PIN  # host pinning passes through from the model's params
    assert body["usage"] == {"include": True}  # ask for the real cost
    assert body["max_completion_tokens"] == 64 and body["seed"] == 1 and "logit_bias" not in body


def test_successful_completion_reports_cost_and_host():
    provider, _ = provider_with(lambda r: httpx.Response(200, json=OR_OK))
    completion = run(provider.complete("openai/gpt-oss-20b", USER, {}))
    assert completion.text == "Answer: 4" and completion.finish_reason == "stop"
    assert (completion.prompt_tokens, completion.completion_tokens) == (40, 60)
    assert completion.provider_cost_usd == pytest.approx(0.0000096) and completion.served_by == "DeepInfra"


def test_missing_key():
    error = error_of(OpenRouterProvider(None, "https://openrouter.test/api/v1"))
    assert error.kind == AUTH and "OPENROUTER_API_KEY is not set" in str(error)


@pytest.mark.parametrize("status, message, headers, kind, retry_after", [
    (401, "No auth credentials found", None, AUTH, None),  # invalid key
    (402, "Insufficient credits. Add more using https://openrouter.ai/credits", None, QUOTA_EXHAUSTED, None),
    (429, "Rate limit exceeded: limit_rpm/openai/gpt-oss-20b", {"retry-after": "12"}, RATE_LIMIT, 12.0),
    (429, "Rate limit exceeded: free-models-per-day", None, QUOTA_EXHAUSTED, None),
    (403, "Input was flagged by moderation", None, BAD_REQUEST, None),
    (408, "Request timed out upstream", None, TIMEOUT, None),
    (503, "No endpoints available matching your data policy", None, OVERLOADED, None),
])
def test_http_errors_are_classified(status, message, headers, kind, retry_after):
    error = error_of(failing(status, message, headers))
    assert error.kind == kind and error.status == status
    if retry_after is not None:
        assert error.retry_after == retry_after
    assert str(error).startswith(f"{kind}: OpenRouter HTTP {status}")


def test_timeout():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    provider, _ = provider_with(timeout)
    assert error_of(provider).kind == TIMEOUT


@pytest.mark.parametrize("payload, kind", [
    ({"error": {"code": 502, "message": "upstream host failed"}}, OVERLOADED),
    ({"choices": [{"index": 0, "error": {"code": 429, "message": "host busy"}, "finish_reason": "error"}]}, RATE_LIMIT),
])
def test_errors_inside_a_200_body(payload, kind):
    provider, _ = provider_with(lambda r: httpx.Response(200, json=payload))
    assert error_of(provider).kind == kind


def test_error_messages_are_sanitised():
    leaky = "Key sk-or-v1-0123456789abcdef is invalid for org_01ABCdef99 (projects/123456789)"
    for provider in (failing(401, leaky),
                     provider_with(lambda r: httpx.Response(200, json={"error": {"code": 401, "message": leaky}}))[0]):
        message = str(error_of(provider))
        assert "sk-or-v1-0123456789abcdef" not in message and "org_01ABCdef99" not in message
        assert "123456789" not in message and "[redacted key]" in message


# ── provider selection ──────────────────────────────────────────────

OR_CONFIG = {
    "providers": {"openrouter": {"base_url": "https://openrouter.test/api/v1", "api_key_env": "TEST_OPENROUTER_KEY"}},
    "models": {"or-small": {"provider": "openrouter", "model": "openai/gpt-oss-20b",
                            "price_per_1m": {"input": 0.03, "output": 0.14},
                            "params": {"reasoning_effort": "low", "provider": PIN}}},
}


def or_config():
    return make_config(**OR_CONFIG)


def test_build_providers_creates_the_openrouter_adapter(monkeypatch):
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "sk-or-v1-x")
    providers = build_providers(or_config())
    assert isinstance(providers["openrouter"], OpenRouterProvider) and set(providers) == {"groq", "gemini", "openrouter"}


def test_each_model_alias_goes_to_its_provider():
    fakes = {name: FakeProvider(name) for name in ("groq", "gemini", "openrouter")}
    registry = ModelRegistry(or_config(), fakes)
    run(registry.complete("or-small", USER, {}))
    run(registry.complete("small", USER, {}))
    assert [c["model"] for c in fakes["openrouter"].calls] == ["openai/gpt-oss-20b"]
    assert fakes["openrouter"].calls[0]["params"]["provider"] == PIN
    assert [c["model"] for c in fakes["groq"].calls] == ["openai/gpt-oss-20b"]


def test_gateway_logs_the_cost_openrouter_reports(tmp_path):
    def script(model, messages, params):
        return Completion("4", "stop", 40, 60, model, provider_cost_usd=0.00042, served_by="DeepInfra")

    providers = {"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini"),
                 "openrouter": FakeProvider("openrouter", script)}
    store = Store(tmp_path / "gw.db")
    app = create_app(or_config(), providers=providers, store=store, router=RulesRouter(), rng=random.Random(0))
    body = TestClient(app).post("/v1/chat/completions", json={"model": "or-small", "messages": USER}).json()
    assert body["gateway"]["model"] == "or-small" and body["gateway"]["cost_usd"] == pytest.approx(0.00042)
    assert store.query("SELECT cost_usd FROM calls")[0]["cost_usd"] == pytest.approx(0.00042)


def test_shipped_config_has_openrouter_and_keeps_the_groq_only_release():
    config = load_config(DEFAULT_CONFIG)
    assert config.providers["openrouter"].api_key_env == "OPENROUTER_API_KEY"
    assert config.providers["openrouter"].base_url == "https://openrouter.ai/api/v1"
    assert config.models["or-gpt-oss-20b"].provider == "openrouter"
    assert config.models["or-gpt-oss-120b"].model == "openai/gpt-oss-120b"
    assert config.releases["v1"].cheap == "gpt-oss-20b" and config.releases["v1"].batch is None


# ── separate, resumable answer files ────────────────────────────────

QUESTIONS = [Question(f"gsm8k-{i:04d}", "gsm8k", "train", "q", "4") for i in range(4)]


def or_script(model, messages, params):
    return Completion("Answer: 4", "stop", 40, 60, model, provider_cost_usd=0.00001, served_by="DeepInfra")


def collect_or(log, questions=QUESTIONS, script=or_script):
    async def no_sleep(_):
        return None

    fakes = {"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini"),
             "openrouter": FakeProvider("openrouter", script)}
    registry = ModelRegistry(or_config(), fakes, sleep=no_sleep)
    return run(collect(questions, ["or-small"], registry, log, report=lambda _: None))


def test_openrouter_answers_are_stored_with_provider_metadata(tmp_path):
    groq_file = tmp_path / "answers.jsonl"
    groq_file.write_text('{"question_id": "gsm8k-0000", "model": "small", "correct": true, "error": null}\n')
    log = AnswerLog(tmp_path / "answers_openrouter.jsonl")
    summary = collect_or(log, QUESTIONS[:2])
    assert summary.answered == 2
    record = log.answers()[("gsm8k-0000", "or-small")]
    assert record["provider"] == "openrouter" and record["provider_model"] == "openai/gpt-oss-20b"
    assert record["served_by"] == "DeepInfra" and record["provider_cost_usd"] == pytest.approx(0.00001)
    assert record["cost_usd"] == pytest.approx((40 * 0.03 + 60 * 0.14) / 1e6)
    assert record["text"] == "Answer: 4" and record["correct"] is True and record["ts"]
    assert record["prompt_tokens"] == 40 and record["latency_ms"] >= 0 and record["wait_ms"] >= 0
    assert groq_file.read_text().count("\n") == 1  # the Groq file is untouched


def test_resume_never_duplicates_answers(tmp_path):
    log = AnswerLog(tmp_path / "answers_openrouter.jsonl")
    collect_or(log, QUESTIONS[:2])
    summary = collect_or(log)  # rerun over all four
    assert (summary.answered, summary.skipped) == (2, 2)
    ids = [r["question_id"] for r in log.records()]
    assert sorted(ids) == sorted(q.id for q in QUESTIONS) and len(ids) == len(set(ids))


def test_errors_are_saved_sanitised_and_retried_on_resume(tmp_path):
    log = AnswerLog(tmp_path / "answers_openrouter.jsonl")

    def out_of_credits(model, messages, params):
        raise ProviderError("OpenRouter HTTP 402: Insufficient credits for org_01SECRET", kind=QUOTA_EXHAUSTED,
                            status=402)

    summary = collect_or(log, QUESTIONS[:1], out_of_credits)
    assert summary.stopped == "out of credits for or-small"
    error = log.records()[0]
    assert error["error_kind"] == QUOTA_EXHAUSTED and "org_01SECRET" not in error["error"]
    assert collect_or(log, QUESTIONS[:1]).answered == 1  # retried after topping up


def test_a_second_writer_is_refused(tmp_path):
    log = AnswerLog(tmp_path / "answers_openrouter.jsonl")
    with log.lock():
        with pytest.raises(AnswerLogLocked):
            collect_or(log)
    assert not (tmp_path / "answers_openrouter.jsonl.lock").exists()
    assert collect_or(log).answered == 4  # the lock is released afterwards


def test_question_selection_helpers(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    for qid, model in [("a", "m1"), ("a", "m2"), ("b", "m1"), ("c", "m2")]:
        log.append({"question_id": qid, "model": model, "error": None})
    log.append({"question_id": "b", "model": "m2", "error": "quota"})
    assert log.answered_by_all() == {"a"}
    qs = [Question(f"q{i}", "gsm8k", "train", "q", "1") for i in range(50)]
    picked = sample(qs, 10, seed=3)
    assert len(picked) == 10 and picked == sample(qs, 10, seed=3) and picked == sorted(picked, key=qs.index)
    assert sample(qs, 100) == qs


# ── comparison ──────────────────────────────────────────────────────

def test_wilson_interval():
    low, high = wilson(8, 10)
    assert 0.49 < low < 0.5 and 0.94 < high < 0.95
    assert wilson(0, 0) == (0.0, 0.0)


def rec(qid, model, correct, provider, **extra):
    return {"question_id": qid, "model": model, "correct": correct, "provider": provider, "error": None,
            "source": "gsm8k", "provider_model": "openai/gpt-oss-20b", "prompt_tokens": 100, "completion_tokens": 100,
            **extra}


def test_comparison_uses_common_questions_and_clean_latency():
    config = make_config(**OR_CONFIG)
    records = [
        rec("q1", "small", True, "groq", latency_ms=50_000),  # pilot record: latency may include waits
        rec("q2", "small", False, "groq", latency_ms=400, wait_ms=0),
        rec("q3", "small", True, "groq", latency_ms=500, wait_ms=0),
        rec("q1", "or-small", True, "openrouter", latency_ms=900, wait_ms=0, provider_cost_usd=0.00002),
        rec("q2", "or-small", True, "openrouter", latency_ms=700, wait_ms=0, provider_cost_usd=0.00002),
        {"question_id": "q3", "model": "or-small", "error": "busy", "error_kind": RATE_LIMIT},
        rec("q2", "or-small", True, "openrouter", latency_ms=800, wait_ms=0, provider_cost_usd=0.00002),  # duplicate
    ]
    answers = latest_answers(records)
    common = common_questions(answers, ["small", "or-small"])
    assert common == {"q1", "q2"}
    groq = model_stats("small", answers, records, common, config)
    openrouter = model_stats("or-small", answers, records, common, config)
    assert (groq.n, groq.correct, openrouter.correct) == (2, 1, 2)
    assert groq.latency_n == 1 and groq.latency_p50_ms == 400  # the pilot record is left out of latency
    assert openrouter.latency_mean_ms == 850  # the duplicate collapses to the latest answer
    assert openrouter.cost_per_1k_reported == pytest.approx(0.02) and groq.cost_per_1k_reported is None
    assert groq.cost_per_1k_estimated == pytest.approx((100 * 0.10 + 100 * 0.50) / 1e6 * 1000)
    assert (openrouter.failures, openrouter.rate_limited, openrouter.calls) == (1, 1, 4)
    pair = agreement(answers, "small", "or-small", common)
    assert {k: pair[k] for k in ("n", "both_right", "both_wrong", "only_first", "only_second")} == {
        "n": 2, "both_right": 1, "both_wrong": 0, "only_first": 0, "only_second": 1}
    assert pair["mcnemar_p"] == 1.0


def test_mcnemar_and_same_answer_counts():
    from router_lab.compare import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(8, 13) == pytest.approx(0.383, abs=1e-3)
    assert mcnemar_exact(12, 31) == pytest.approx(0.005, abs=1e-3)
    answers = {("q1", "a"): {"correct": True, "extracted": "4"}, ("q1", "b"): {"correct": True, "extracted": "4"},
               ("q2", "a"): {"correct": False, "extracted": "7"}, ("q2", "b"): {"correct": False, "extracted": "9"},
               ("q3", "a"): {"correct": False, "extracted": None}, ("q3", "b"): {"correct": False, "extracted": None}}
    pair = agreement(answers, "a", "b", {"q1", "q2", "q3"})
    assert pair["same_answer"] == 1 and pair["both_wrong"] == 2  # two different wrong answers are not "the same"


def test_compare_script(tmp_path, monkeypatch, capsys):
    import importlib.util

    import yaml

    from gateway.config import ROOT
    from tests.conftest import BASE_CONFIG

    cfg = {**BASE_CONFIG, "providers": {**BASE_CONFIG["providers"], **OR_CONFIG["providers"]},
           "models": {**BASE_CONFIG["models"], **OR_CONFIG["models"]}}
    (tmp_path / "gateway.yaml").write_text(yaml.safe_dump(cfg))
    groq, orf = tmp_path / "answers.jsonl", tmp_path / "answers_openrouter.jsonl"
    groq.write_text("\n".join(json.dumps(rec(f"q{i}", "small", i % 2 == 0, "groq", latency_ms=500, wait_ms=0))
                              for i in range(6)) + "\n")
    orf.write_text("\n".join(json.dumps(rec(f"q{i}", "or-small", True, "openrouter", latency_ms=800, wait_ms=0,
                                            provider_cost_usd=0.00002, served_by="DeepInfra"))
                             for i in range(4)) + "\n")
    spec = importlib.util.spec_from_file_location("compare_providers", ROOT / "scripts" / "compare_providers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "cmp.md"
    monkeypatch.setattr(sys, "argv", ["compare", "--answers", str(groq), str(orf), "--config",
                                      str(tmp_path / "gateway.yaml"), "--out", str(out)])
    assert module.main() == 0
    text = capsys.readouterr().out
    assert "Questions answered by every model: 4." in text and "openrouter (DeepInfra 4)" in text
    assert "same model, different provider" in text
    data = json.loads(out.with_suffix(".json").read_text())
    assert {m["model"] for m in data["models"]} == {"small", "or-small"}


# ── the collection script's new options ─────────────────────────────

def test_collect_script_selects_questions_and_never_prints_keys(tmp_path, monkeypatch, capsys):
    import importlib.util

    import yaml

    from gateway.config import ROOT
    from router_lab.questions import save_questions
    from tests.conftest import BASE_CONFIG

    cfg = {**BASE_CONFIG, "providers": {**BASE_CONFIG["providers"], **OR_CONFIG["providers"]},
           "models": {**BASE_CONFIG["models"], **OR_CONFIG["models"]}}
    (tmp_path / "gateway.yaml").write_text(yaml.safe_dump(cfg))
    questions = [Question(f"gsm8k-{i:04d}", "gsm8k", "train", "q", "4") for i in range(30)]
    save_questions(tmp_path / "questions.jsonl", questions)
    groq = AnswerLog(tmp_path / "answers.jsonl")
    for q in questions[:20]:
        for model in ("small", "large"):
            groq.append({"question_id": q.id, "model": model, "correct": True, "error": None})
    env = tmp_path / "other.env"
    env.write_text("TEST_OPENROUTER_KEY=sk-or-v1-supersecretvalue\n")
    monkeypatch.delenv("TEST_OPENROUTER_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("collect_labels", ROOT / "scripts" / "collect_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = ["collect", "--questions", str(tmp_path / "questions.jsonl"), "--config", str(tmp_path / "gateway.yaml"),
            "--cheap", "or-small", "--strong", "or-small", "--answers", str(tmp_path / "answers_openrouter.jsonl"),
            "--only-answered-in", str(tmp_path / "answers.jsonl"), "--sample", "5", "--env-file", str(env),
            "--dry-run"]
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "questions: 5 " in out and "API key set" in out and "supersecretvalue" not in out
    monkeypatch.setattr(sys, "argv", [*args[:-2], str(tmp_path / "missing.env"), "--dry-run"])
    assert module.main() == 1
