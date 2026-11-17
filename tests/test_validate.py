"""Dataset quality checks find the problems they are meant to find."""

import importlib.util
import json
import sys

from gateway.config import ROOT
from router_lab.progress import AnswerLog
from router_lab.questions import Question, save_questions
from router_lab.validate import integrity_problems, markdown, question_problems, validate

QUESTIONS = [Question(f"gsm8k-{i:04d}", "gsm8k", "test" if i < 2 else "train", "q?", str(i)) for i in range(5)]


def good(qid, model, answer, correct=True):
    return {"question_id": qid, "model": model, "provider_model": "openai/" + model, "split": "test" if qid < "gsm8k-0002"
            else "train", "source": "gsm8k", "text": f"Answer: {answer}", "correct": correct, "extracted": str(answer),
            "gold": str(int(qid[-4:])), "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 5,
            "latency_ms": 100.0, "ts": "2026-11-15T00:00:00+00:00", "error": None, "provider_cost_usd": 0.00001,
            "served_by": "DeepInfra"}


def write(tmp_path, rows, name="answers.jsonl"):
    log = AnswerLog(tmp_path / name)
    for row in rows:
        log.append(row)
    return log.path


def test_clean_data_has_no_problems(tmp_path):
    rows = [good(q.id, m, int(q.id[-4:])) for q in QUESTIONS for m in ("a", "b")]
    result = validate(QUESTIONS, [write(tmp_path, rows)])
    assert integrity_problems(result) == 0 and result["answered_by_every_model"] == 5
    a = next(m for m in result["models"] if m.model == "a")
    assert (a.answered, a.missing, a.accuracy, a.served_by) == (5, 0, 1.0, {"DeepInfra": 5})
    assert "Integrity problems (" in markdown(result) and "): 0." in markdown(result)


def test_problems_are_found(tmp_path):
    rows = [good(q.id, "a", int(q.id[-4:])) for q in QUESTIONS[:4]]
    rows.append(good("gsm8k-0000", "a", 0))  # duplicate
    rows.append({**good("gsm8k-0001", "a", 1), "question_id": "gsm8k-9999"})  # unknown id
    bad_gold = {**good("gsm8k-0002", "b", 2), "gold": "7"}
    missing_field = {k: v for k, v in good("gsm8k-0003", "b", 3).items() if k != "latency_ms"}
    wrong_grade = {**good("gsm8k-0001", "b", 5), "correct": True}  # "Answer: 5" for gold 1 is wrong
    no_answer = {**good("gsm8k-0000", "b", 0), "text": "I am not sure.", "extracted": None, "correct": False}
    truncated = {**good("gsm8k-0004", "b", 4), "finish_reason": "length"}
    rows += [bad_gold, missing_field, wrong_grade, no_answer, truncated,
             {"question_id": "gsm8k-0002", "model": "b", "error": "busy", "error_kind": "rate_limit"}]
    result = validate(QUESTIONS, [write(tmp_path, rows)])
    a, b = (next(m for m in result["models"] if m.model == name) for name in ("a", "b"))
    assert a.duplicates == 1 and a.unknown_ids == ["gsm8k-9999"] and a.missing == 1
    assert b.gold_mismatches == 1 and b.incomplete_records == 1 and b.regrade_mismatches == 1
    assert b.no_extracted_answer == 1 and b.truncated == 1 and b.errors == {"rate_limit": 1}
    assert integrity_problems(result) == 5
    assert "regrade" in markdown(result)


def test_question_set_problems():
    bad = [Question("x", "mmlu", "test", "q", "E", ["a", "b"]), Question("x", "gsm8k", "dev", "", "seven")]
    problems = question_problems(bad)
    assert any("not one of the choices" in p for p in problems)
    assert any("duplicate id" in p for p in problems) and any("unknown split" in p for p in problems)
    assert any("not a number" in p for p in problems)


def test_script_writes_the_report_and_fails_on_problems(tmp_path, monkeypatch, capsys):
    save_questions(tmp_path / "questions.jsonl", QUESTIONS)
    clean = write(tmp_path, [good(q.id, "a", int(q.id[-4:])) for q in QUESTIONS], "clean.jsonl")
    spec = importlib.util.spec_from_file_location("validate_dataset", ROOT / "scripts" / "validate_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "quality.md"
    monkeypatch.setattr(sys, "argv", ["v", "--questions", str(tmp_path / "questions.jsonl"), "--answers", str(clean),
                                      "--out", str(out)])
    assert module.main() == 0
    assert json.loads(out.with_suffix(".json").read_text())["answered_by_every_model"] == 5
    broken = write(tmp_path, [good("gsm8k-0000", "a", 0), good("gsm8k-0000", "a", 0)], "broken.jsonl")
    monkeypatch.setattr(sys, "argv", ["v", "--questions", str(tmp_path / "questions.jsonl"), "--answers", str(broken)])
    assert module.main() == 1
