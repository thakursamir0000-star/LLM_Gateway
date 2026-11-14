"""Router lab: grading, sampling, resumable collection, training and evaluation (no real API calls)."""

import asyncio
import importlib.util
import json
import sys

import numpy as np
import pytest

from gateway.config import ROOT
from gateway.providers import ModelRegistry
from gateway.providers.base import AUTH, OVERLOADED, QUOTA_EXHAUSTED, Completion, ProviderError
from gateway.providers.fake import FakeProvider
from gateway.routing import RulesRouter
from router_lab.collect import collect
from router_lab.dataset import build_pairs
from router_lab.evaluate import (
    Item,
    cascade_point,
    cheapest_within,
    markdown_table,
    random_point,
    route_point,
    router_latency_ms,
    sweep,
)
from router_lab.progress import AnswerLog
from router_lab.questions import (
    Question,
    extract_choice,
    extract_number,
    format_prompt,
    from_gsm8k,
    from_mmlu,
    grade,
    gsm8k_gold,
    load_questions,
    messages_for,
    sample_and_split,
    save_questions,
)
from tests.conftest import make_config

GSM = Question("gsm8k-0001", "gsm8k", "train", "Tom has 3 boxes of 4 apples. How many apples?", "12")
MMLU = Question("mmlu-00001", "mmlu", "test", "Capital of France?", "B", ["Berlin", "Paris", "Rome", "Madrid"],
                "geography")


# ── grading ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("3 boxes x 4 = 12\nAnswer: 12", "12"),
    ("**Answer:** $1,234", "1234"),
    ("Answer: 12.0 apples", "12"),
    ("so the answer is 7.", "7"),
    ("we get \\boxed{18}", "18"),
    ("first 5, then 9, finally 11", "11"),
    ("Answer: 3, wait no.\nAnswer: 4", "4"),
    ("no numbers here", None),
])
def test_extract_number(text, expected):
    assert extract_number(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Paris is the capital.\nAnswer: B", "B"),
    ("Answer: **C**", "C"),
    ("Answer: (A) Berlin", "A"),
    ("I think it is (D)", "D"),
    ("The correct option is **B**.", "B"),
    ("Answer: none of these", None),
])
def test_extract_choice(text, expected):
    assert extract_choice(text) == expected


def test_grade():
    assert grade(GSM, "Answer: 12") == (True, "12")
    assert grade(GSM, "Answer: 13") == (False, "13")
    assert grade(GSM, "I don't know") == (False, None)
    assert grade(MMLU, "Answer: B") == (True, "B")
    assert grade(MMLU, "Answer: A") == (False, "A")


def test_dataset_rows_become_questions():
    gsm = from_gsm8k([{"question": " Q? ", "answer": "work...\n#### 1,050"}])
    assert gsm[0].id == "gsm8k-0000" and gsm[0].answer == "1050" and gsm[0].question == "Q?"
    assert gsm8k_gold("#### -3") == "-3"
    mmlu = from_mmlu([{"question": "Q", "subject": "law", "choices": ["a", "b", "c", "d"], "answer": 2}])
    assert mmlu[0].answer == "C" and mmlu[0].subject == "law"


def test_prompts_ask_for_an_answer_line_without_tripping_the_rules():
    assert format_prompt(GSM).endswith('"Answer: <number>".')
    assert "B. Paris" in format_prompt(MMLU) and '"Answer: <letter>"' in format_prompt(MMLU)
    assert "multi-step reasoning" not in RulesRouter().score(messages_for(GSM)).reason


def test_sample_and_split_is_fixed_by_the_seed(tmp_path):
    pools = {"gsm8k": [Question(f"g{i}", "gsm8k", "", "q", "1") for i in range(50)],
             "mmlu": [Question(f"m{i}", "mmlu", "", "q", "A", ["a", "b"]) for i in range(80)]}
    picked = sample_and_split(pools, {"gsm8k": 20, "mmlu": 40}, 0.25, seed=1)
    assert len(picked) == 60
    assert sum(q.split == "test" for q in picked if q.source == "gsm8k") == 5
    assert sum(q.split == "test" for q in picked if q.source == "mmlu") == 10
    assert [q.id for q in picked] == [q.id for q in sample_and_split(pools, {"gsm8k": 20, "mmlu": 40}, 0.25, seed=1)]
    assert {q.source for q in picked[:10]} == {"gsm8k", "mmlu"}  # sources are interleaved
    path = tmp_path / "q.jsonl"
    save_questions(path, picked)
    assert load_questions(path) == picked
    with pytest.raises(ValueError):
        sample_and_split(pools, {"gsm8k": 51}, 0.2)


# ── answer log and collection ───────────────────────────────────────

def test_answer_log_resumes_and_skips_errors_and_torn_lines(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    log.append({"question_id": "q1", "model": "small", "correct": True, "error": None})
    log.append({"question_id": "q2", "model": "small", "error": "boom"})
    with log.path.open("a") as f:
        f.write('{"question_id": "q3", "mod')  # a crash mid-write
    assert set(log.answers()) == {("q1", "small")}
    assert len(log.records()) == 2


def run_collect(questions, script, log, models=("small", "large")):
    config = make_config()

    async def no_sleep(_):
        return None

    registry = ModelRegistry(config, {"groq": FakeProvider("groq", script), "gemini": FakeProvider("gemini")},
                             sleep=no_sleep)
    return asyncio.run(collect(questions, list(models), registry, log, params={"max_completion_tokens": 64},
                               report=lambda _: None))


def answer_twelve(model, messages, params):
    return Completion("Answer: 12", "stop", 20, 5, model)


def test_collect_saves_every_answer_and_resumes(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    questions = [GSM, Question("gsm8k-0002", "gsm8k", "test", "Half of 30?", "15")]
    summary = run_collect(questions[:1], answer_twelve, log)
    assert (summary.answered, summary.skipped, summary.stopped) == (2, 0, None)
    record = log.answers()[("gsm8k-0001", "small")]
    assert record["correct"] is True and record["prompt_tokens"] == 20 and record["split"] == "train"
    assert record["provider_model"] == "openai/gpt-oss-20b" and record["text"] == "Answer: 12"
    summary = run_collect(questions, answer_twelve, log)
    assert (summary.answered, summary.skipped) == (2, 2)
    assert log.answers()[("gsm8k-0002", "large")]["correct"] is False


def test_collect_stops_at_the_daily_quota_and_keeps_progress(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    questions = [Question(f"gsm8k-{i:04d}", "gsm8k", "train", "q", "12") for i in range(5)]
    calls = {"n": 0}

    def script(model, messages, params):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ProviderError("tokens per day (TPD)", kind=QUOTA_EXHAUSTED)
        return answer_twelve(model, messages, params)

    summary = run_collect(questions, script, log)
    assert summary.stopped == "daily quota used up for large" and summary.answered == 3
    assert len(log.answers()) == 3 and log.records()[-1]["error_kind"] == QUOTA_EXHAUSTED
    summary = run_collect(questions, answer_twelve, log)  # the next day
    assert (summary.answered, summary.skipped) == (7, 3) and len(log.answers()) == 10


def test_collect_stops_on_key_problems_and_repeated_errors(tmp_path):
    def auth(model, messages, params):
        raise ProviderError("GROQ_API_KEY is not set", kind=AUTH)

    assert run_collect([GSM], auth, AnswerLog(tmp_path / "a.jsonl")).stopped.startswith("API key problem")

    def down(model, messages, params):
        raise ProviderError("busy", kind=OVERLOADED)

    questions = [Question(f"q{i}", "gsm8k", "train", "q", "1") for i in range(10)]
    summary = run_collect(questions, down, AnswerLog(tmp_path / "b.jsonl"))
    assert summary.stopped == "5 errors in a row" and summary.errors == 5


# ── labels and evaluation ───────────────────────────────────────────

def test_cheap_ok_label():
    def rec(correct):
        return {"correct": correct}

    questions = [Question(f"q{i}", "gsm8k", "test", "q", "1") for i in range(4)]
    answers = {}
    for q, (cheap, strong) in zip(questions, [(True, True), (False, True), (False, False), (True, False)],
                                  strict=True):
        answers[(q.id, "small")], answers[(q.id, "large")] = rec(cheap), rec(strong)
    pairs = build_pairs(questions + [Question("lonely", "gsm8k", "test", "q", "1")], answers, "small", "large")
    assert [p.cheap_ok for p in pairs] == [True, False, True, True]
    assert build_pairs(questions, answers, "small", "large", split="train") == []


def items(outcomes):
    """Items from (cheap_correct, strong_correct) pairs; cheap costs 1, strong 10."""
    return [Item(f"q{i}", c, s, 1.0, 10.0, 1.0, 3.0, "fine answer", "stop", "q") for i, (c, s) in enumerate(outcomes)]


def test_route_and_cascade_points():
    data = items([(True, True), (False, True), (False, False), (True, True)])
    cheap = route_point(data, [False] * 4, "always-cheap", None)
    strong = route_point(data, [True] * 4, "always-strong", None)
    assert (cheap.accuracy, cheap.cost_usd, cheap.strong_share) == (0.5, 4.0, 0.0)
    assert (strong.accuracy, strong.cost_usd, strong.cost_per_1k) == (0.75, 40.0, 10_000.0)
    oracle = route_point(data, [False, True, False, False], "oracle", None)
    assert (oracle.accuracy, oracle.cost_usd) == (0.75, 13.0)
    cascade = cascade_point(data, [False, True, False, False], 0.5)
    assert (cascade.accuracy, cascade.cost_usd, cascade.llm_latency_s) == (0.75, 14.0, 1.75)
    half = random_point(data, 0.5)
    assert half.accuracy == pytest.approx(0.625) and half.cost_usd == pytest.approx(22.0)
    assert "| always-cheap | - | 50.0% |" in markdown_table([cheap])


def test_threshold_sweep_moves_from_cheap_to_strong():
    data = items([(True, True), (False, True), (False, False), (True, True)])
    points = sweep(data, [0.9, 0.2, 0.5, 0.8], "router", [0.0, 0.3, 0.6, 1.0])
    assert [p.strong_share for p in points] == [0.0, 0.25, 0.5, 1.0]
    assert [p.accuracy for p in points] == [0.5, 0.75, 0.75, 0.75]
    best = cheapest_within(points, 0.75)
    assert best.threshold == 0.3 and best.cost_usd == 13.0
    assert cheapest_within(points, 0.9) is None


def test_router_latency_measurement():
    stats = router_latency_ms(lambda text: RulesRouter().score([{"role": "user", "content": text}]), ["a", "b"] * 5)
    assert stats["n"] == 10 and 0 <= stats["p50_ms"] <= stats["p95_ms"]


# ── the scripts, end to end on synthetic answers ────────────────────

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lab(tmp_path):
    """Questions and both models' answers: the cheap model fails the 'integral' questions."""
    rng = np.random.default_rng(0)
    questions, log = [], AnswerLog(tmp_path / "answers.jsonl")
    for i in range(120):
        hard = i % 3 == 0
        text = (f"Compute the definite integral number {i} of the polynomial" if hard
                else f"How many apples are in basket number {i}")
        q = Question(f"gsm8k-{i:04d}", "gsm8k", "test" if i % 4 == 0 else "train", text, "12")
        questions.append(q)
        cheap_right = (not hard) and rng.random() > 0.05
        for model, right, tokens in (("small", cheap_right, 300), ("large", rng.random() > 0.05, 400)):
            log.append({"question_id": q.id, "split": q.split, "source": q.source, "model": model,
                        "correct": bool(right), "text": "Answer: 12" if right else "Answer: 7",
                        "finish_reason": "stop", "prompt_tokens": 50, "completion_tokens": tokens,
                        "latency_ms": 900.0, "error": None})
    save_questions(tmp_path / "questions.jsonl", questions)
    return tmp_path


def run_main(module, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    return module.main()


def write_config(tmp_path):
    import yaml

    from tests.conftest import BASE_CONFIG

    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump(BASE_CONFIG))
    return path


def test_train_and_evaluate_scripts(lab, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    config = write_config(lab)
    common = ["--questions", str(lab / "questions.jsonl"), "--answers", str(lab / "answers.jsonl"),
              "--config", str(config)]
    router = lab / "router.json"
    assert run_main(load_script("train_router"), monkeypatch,
                    [*common, "--embedder", "hashing:256", "--out", str(router)]) == 0
    meta = json.loads(router.read_text())
    assert meta["cheap"] == "small" and meta["strong"] == "large" and meta["n_train"] == 90
    assert meta["cv"]["results"][0]["auc"] is not None

    out = lab / "results"
    assert run_main(load_script("evaluate_router"), monkeypatch,
                    [*common, "--router", str(router), "--out", str(out)]) == 0
    report = (out / "router_eval.md").read_text()
    assert "Held-out questions: 30 (gsm8k 30)" in report
    for strategy in ("always-cheap", "always-strong", "random", "rules", "classifier", "cascade"):
        assert f"| {strategy} |" in report
    assert "| classifier | " in report.split("Added latency")[1]
    assert (out / "cost_vs_accuracy.png").stat().st_size > 10_000
    data = json.loads((out / "router_eval.json").read_text())
    by_name = {p["strategy"]: p for p in data["operating"]}
    # The trained router learned the pattern: close to always-strong accuracy at a lower cost.
    assert by_name["classifier"]["accuracy"] >= by_name["always-strong"]["accuracy"] - 0.1
    assert by_name["classifier"]["cost_usd"] < by_name["always-strong"]["cost_usd"]
    assert data["router_latency"]["classifier"]["n"] == 30


def test_scripts_refuse_without_enough_data(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    save_questions(tmp_path / "questions.jsonl", [GSM])
    common = ["--questions", str(tmp_path / "questions.jsonl"), "--answers", str(tmp_path / "none.jsonl"),
              "--config", str(config)]
    assert run_main(load_script("train_router"), monkeypatch, [*common, "--embedder", "hashing:64"]) == 1
    assert run_main(load_script("evaluate_router"), monkeypatch,
                    [*common, "--router", str(tmp_path / "none.json"), "--out", str(tmp_path / "r")]) == 1


def test_collect_dry_run_makes_no_calls(lab, monkeypatch, capsys):
    config = write_config(lab)
    assert run_main(load_script("collect_labels"), monkeypatch,
                    ["--questions", str(lab / "questions.jsonl"), "--answers", str(lab / "new.jsonl"),
                     "--config", str(config), "--limit", "3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "6 calls to go" in out and "sample prompt" in out
    assert not (lab / "new.jsonl").exists()


def test_collect_refuses_to_spend_quota_non_interactively(lab, monkeypatch):
    config = write_config(lab)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert run_main(load_script("collect_labels"), monkeypatch,
                    ["--questions", str(lab / "questions.jsonl"), "--answers", str(lab / "new.jsonl"),
                     "--config", str(config), "--limit", "1"]) == 1


def test_prepare_keeps_an_existing_split(tmp_path, monkeypatch):
    existing = tmp_path / "questions.jsonl"
    existing.write_text("keep me")
    assert run_main(load_script("prepare_questions"), monkeypatch, ["--out", str(existing)]) == 1
    assert existing.read_text() == "keep me"
