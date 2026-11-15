"""The collection spending cap (--max-cost-usd), with fake providers only."""

import asyncio
import importlib.util
import sys

import pytest

from gateway.config import ROOT
from gateway.providers import ModelRegistry
from gateway.providers.base import Completion
from gateway.providers.fake import FakeProvider
from router_lab.budget import Budget, record_cost, spent_with_tag, worst_case_cost
from router_lab.collect import collect
from router_lab.progress import AnswerLog
from router_lab.questions import Question, messages_for, save_questions
from tests.conftest import make_config

OR = {
    "providers": {"openrouter": {"base_url": "https://openrouter.test/api/v1", "api_key_env": "TEST_OPENROUTER_KEY"}},
    "models": {"or-small": {"provider": "openrouter", "model": "openai/gpt-oss-20b",
                            "price_per_1m": {"input": 0.03, "output": 0.14}},
               "or-large": {"provider": "openrouter", "model": "openai/gpt-oss-120b",
                            "price_per_1m": {"input": 0.037, "output": 0.17}}},
}
QUESTIONS = [Question(f"gsm8k-{i:04d}", "gsm8k", "train", "How many apples are in the basket?", "4") for i in range(10)]
PARAMS = {"max_completion_tokens": 2048}


def config():
    return make_config(**OR)


def answer(cost="list", prompt=100, completion=100):
    """A fake script: every call uses the given tokens; the provider reports ``cost``: "list" = the list price
    (what OpenRouter reported on every real call), None = no figure, or a number."""
    prices = {spec.model: spec for spec in config().models.values() if spec.provider == "openrouter"}

    def script(model, messages, params):
        reported = prices[model].cost(prompt, completion) if cost == "list" else cost
        return Completion("Answer: 4", "stop", prompt, completion, model, provider_cost_usd=reported,
                          served_by="DeepInfra")

    return script


LIST_COST = {"or-small": (100 * 0.03 + 100 * 0.14) / 1e6, "or-large": (100 * 0.037 + 100 * 0.17) / 1e6}


def run(log, budget, script=None, questions=QUESTIONS, models=("or-small", "or-large")):
    async def no_sleep(_):
        return None

    provider = FakeProvider("openrouter", script or answer())
    registry = ModelRegistry(config(), {"groq": FakeProvider("groq"), "gemini": FakeProvider("gemini"),
                                        "openrouter": provider}, sleep=no_sleep)
    summary = asyncio.run(collect(questions, list(models), registry, log, params=PARAMS, budget=budget,
                                  report=lambda _: None))
    return summary, provider


def worst(alias):
    return worst_case_cost(config().models[alias], messages_for(QUESTIONS[0]), PARAMS)


# ── the pieces ──────────────────────────────────────────────────────

def test_worst_case_uses_the_output_cap():
    spec = config().models["or-large"]
    bound = worst_case_cost(spec, messages_for(QUESTIONS[0]), PARAMS)
    assert bound > spec.cost(0, 2048) and bound < spec.cost(1000, 2048)
    assert worst_case_cost(spec, messages_for(QUESTIONS[0]), {}) > bound  # no cap set: assume 4,096 tokens


def test_charge_prefers_reported_cost_and_flags_overbilling():
    spec = config().models["or-small"]
    budget = Budget(1.0)
    assert budget.charge(spec, Completion("x", "stop", 1000, 1000, "m", provider_cost_usd=0.0002)) == (0.0002, False)
    estimate = spec.cost(1000, 1000)
    cost, unexpected = budget.charge(spec, Completion("x", "stop", 1000, 1000, "m"))  # no reported cost
    assert cost == pytest.approx(estimate) and not unexpected
    assert budget.charge(spec, Completion("x", "stop", 1000, 1000, "m", provider_cost_usd=estimate * 3))[1]
    assert budget.spent_usd == pytest.approx(0.0002 + estimate * 4)


def test_tagged_spend_and_record_cost():
    records = [{"budget_tag": "b", "provider_cost_usd": 0.01, "error": None},
               {"budget_tag": "b", "provider_cost_usd": None, "cost_usd": 0.02, "error": None},
               {"budget_tag": "b", "error": "rate limit"},
               {"budget_tag": "other", "provider_cost_usd": 0.5, "error": None},
               {"provider_cost_usd": 0.7, "error": None}]  # an earlier, already-paid record
    assert spent_with_tag(records, "b") == pytest.approx(0.03)
    assert spent_with_tag(records, None) == 0.0
    assert record_cost({"cost_usd": 0.004}) == 0.004


# ── the guard in a collection run ───────────────────────────────────

def test_stops_before_a_call_that_could_pass_the_cap(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    cap = 0.0006
    summary, provider = run(log, Budget(cap, tag="t"))
    assert summary.stopped.startswith("budget: the next call could cost up to")
    assert summary.spent_usd <= cap
    assert summary.spent_usd + max(worst("or-small"), worst("or-large")) > cap  # it stopped only when it had to
    assert len(provider.calls) == summary.answered  # no request was started and then abandoned
    saved = log.answers()
    assert len(saved) == summary.answered and all(r["budget_tag"] == "t" for r in saved.values())


def test_the_cap_holds_even_when_every_call_uses_its_whole_output_cap(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    huge = answer(cost=None, prompt=60, completion=2048)  # each call as expensive as the guard assumes
    summary, _ = run(log, Budget(0.003, tag="t"), huge)
    assert summary.spent_usd <= 0.003 and summary.answered >= 1


def test_resume_counts_earlier_spend_with_the_same_tag(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    first, _ = run(log, Budget(0.0006, tag="t"))
    assert first.stopped
    spent = spent_with_tag(log.records(), "t")
    assert spent == pytest.approx(first.spent_usd)
    again, provider = run(log, Budget(0.0006, spent, tag="t"))  # same budget: nothing more fits
    assert again.answered == 0 and provider.calls == [] and again.skipped == first.answered
    raised, _ = run(log, Budget(0.01, spent_with_tag(log.records(), "t"), tag="t"))  # a larger approved cap
    assert raised.stopped is None and raised.skipped == first.answered
    records = log.records()
    assert len(records) == len({(r["question_id"], r["model"]) for r in records}) == 20  # no duplicates
    assert spent_with_tag(records, "t") <= 0.01


def test_missing_cost_metadata_falls_back_to_the_estimate(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    summary, _ = run(log, Budget(1.0, tag="t"), answer(cost=None), questions=QUESTIONS[:2])
    estimate = config().models["or-small"].cost(100, 100) + config().models["or-large"].cost(100, 100)
    assert summary.spent_usd == pytest.approx(estimate * 2)
    assert all(r["provider_cost_usd"] is None and r["cost_usd"] > 0 for r in log.records())
    assert spent_with_tag(log.records(), "t") == pytest.approx(summary.spent_usd)


def test_unexpected_billing_saves_the_answer_and_stops(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    summary, provider = run(log, Budget(1.0, tag="t"), answer(cost=0.01))  # far above the list price
    assert summary.stopped.startswith("unexpected billing") and summary.answered == 1
    assert len(provider.calls) == 1 and len(log.answers()) == 1


def test_no_budget_means_no_cap(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    summary, _ = run(log, None, questions=QUESTIONS[:3])
    assert summary.stopped is None and summary.answered == 6
    assert summary.spent_usd == pytest.approx(3 * (LIST_COST["or-small"] + LIST_COST["or-large"]))
    assert "budget_tag" not in log.records()[0]


# ── the script ──────────────────────────────────────────────────────

def test_dry_run_reports_estimate_and_budget(tmp_path, monkeypatch, capsys):
    import yaml

    from tests.conftest import BASE_CONFIG

    cfg = {**BASE_CONFIG, "providers": {**BASE_CONFIG["providers"], **OR["providers"]},
           "models": {**BASE_CONFIG["models"], **OR["models"]}}
    (tmp_path / "gateway.yaml").write_text(yaml.safe_dump(cfg))
    save_questions(tmp_path / "questions.jsonl", QUESTIONS)
    log = AnswerLog(tmp_path / "answers.jsonl")
    for q in QUESTIONS[:4]:
        for model in ("or-small", "or-large"):
            log.append({"question_id": q.id, "model": model, "source": "gsm8k", "correct": True, "error": None,
                        "provider_cost_usd": 0.00003, "budget_tag": "t"})
    spec = importlib.util.spec_from_file_location("collect_labels", ROOT / "scripts" / "collect_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [
        "collect", "--questions", str(tmp_path / "questions.jsonl"), "--config", str(tmp_path / "gateway.yaml"),
        "--cheap", "or-small", "--strong", "or-large", "--answers", str(tmp_path / "answers.jsonl"),
        "--max-cost-usd", "0.05", "--budget-tag", "t", "--dry-run"])
    assert module.main() == 0
    out = capsys.readouterr().out
    assert "12 calls to go" in out
    assert "estimated cost of the remaining calls: $0.0004 expected" in out  # 12 x $0.00003
    assert "already spent $0.000240 (tag 't')" in out and "$0.049760 left" in out
