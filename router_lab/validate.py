"""Dataset quality checks for the question set and the answer files (offline).

For each answer file and model: coverage of the question set, missing and
unknown question ids, duplicate answers, error records, records missing
required fields, gold answers that disagree with questions.jsonl, answers with
no extractable final answer, truncated or empty answers, stored grades that
disagree with re-grading the saved text, serving hosts, and cost.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gateway.config import GatewayConfig
from router_lab.budget import record_cost
from router_lab.compare import load_records
from router_lab.questions import LETTERS, Question, grade, normalise_number

REQUIRED = ("question_id", "model", "provider_model", "split", "source", "text", "correct", "extracted", "gold",
            "finish_reason", "prompt_tokens", "completion_tokens", "latency_ms", "ts")


def question_problems(questions: list[Question]) -> list[str]:
    problems = []
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if not q.question.strip() or not q.answer:
            problems.append(f"{q.id}: empty question or answer")
        if q.split not in ("train", "test"):
            problems.append(f"{q.id}: unknown split {q.split!r}")
        if q.source == "mmlu" and (not q.choices or q.answer not in LETTERS[:len(q.choices)]):
            problems.append(f"{q.id}: answer {q.answer!r} is not one of the choices")
        if q.source == "gsm8k" and normalise_number(q.answer) is None:
            problems.append(f"{q.id}: gold answer {q.answer!r} is not a number")
    return problems


@dataclass
class ModelReport:
    file: str
    model: str
    provider_models: dict[str, int]
    records: int
    answered: int  # distinct questions with a successful answer
    errors: dict[str, int]
    duplicates: int  # extra successful answers for a question already answered
    missing: int
    missing_examples: list[str]
    unknown_ids: list[str]
    incomplete_records: int  # successful records missing a required field
    gold_mismatches: int  # stored gold differs from questions.jsonl
    split_or_source_mismatches: int
    no_extracted_answer: int
    truncated: int  # finish_reason length
    empty_text: int
    regrade_mismatches: int  # stored grade differs from grading the saved text again
    accuracy: float
    served_by: dict[str, int]
    reported_cost_usd: float
    records_without_reported_cost: int
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def problems(self) -> int:
        return (self.duplicates + len(self.unknown_ids) + self.incomplete_records + self.gold_mismatches
                + self.split_or_source_mismatches + self.regrade_mismatches)


def _cost(record: dict[str, Any], config: GatewayConfig | None) -> float:
    """Reported cost, else the stored estimate, else list price x tokens (records that predate cost fields)."""
    if record.get("provider_cost_usd") is not None or record.get("cost_usd") is not None or config is None:
        return record_cost(record)
    spec = config.models.get(record.get("model"))
    return spec.cost(record.get("prompt_tokens", 0), record.get("completion_tokens", 0)) if spec else 0.0


def check_model(path: str, model: str, records: list[dict[str, Any]], questions: dict[str, Question],
                config: GatewayConfig | None = None) -> ModelReport:
    mine = [r for r in records if r.get("model") == model]
    ok = [r for r in mine if not r.get("error")]
    errors = Counter(r.get("error_kind", "other") for r in mine if r.get("error"))
    per_question = Counter(r.get("question_id") for r in ok)
    answered = {qid for qid in per_question if qid in questions}
    examples: dict[str, list[str]] = {}

    def note(kind: str, text: str) -> None:
        examples.setdefault(kind, [])
        if len(examples[kind]) < 5:
            examples[kind].append(text)

    incomplete = gold = split_source = no_extract = truncated = empty = regrade = 0
    for r in ok:
        q = questions.get(r.get("question_id"))
        missing_fields = [f for f in REQUIRED if f not in r]
        if missing_fields:
            incomplete += 1
            note("incomplete", f"{r.get('question_id')}: missing {', '.join(missing_fields)}")
        if q is None:
            continue
        if "gold" in r and str(r["gold"]) != q.answer:
            gold += 1
            note("gold", f"{q.id}: stored gold {r['gold']!r}, questions.jsonl {q.answer!r}")
        if r.get("split") != q.split or r.get("source") != q.source:
            split_source += 1
            note("split/source", f"{q.id}: {r.get('split')}/{r.get('source')} vs {q.split}/{q.source}")
        text = r.get("text") or ""
        if not text.strip():
            empty += 1
        if r.get("finish_reason") == "length":
            truncated += 1
            note("truncated", q.id)
        if r.get("extracted") is None:
            no_extract += 1
            note("no extracted answer", f"{q.id}: ...{text[-80:]!r}")
        correct, extracted = grade(q, text)
        if bool(r.get("correct")) != correct or r.get("extracted") != extracted:
            regrade += 1
            note("regrade", f"{q.id}: stored {r.get('correct')}/{r.get('extracted')!r}, now {correct}/{extracted!r}")
    missing = sorted(set(questions) - answered)
    latest = {}
    for r in ok:
        latest[r.get("question_id")] = r
    graded = [r for qid, r in latest.items() if qid in questions]
    return ModelReport(
        file=path, model=model, provider_models=dict(Counter(r.get("provider_model") for r in ok)),
        records=len(mine), answered=len(answered), errors=dict(errors),
        duplicates=sum(n - 1 for n in per_question.values() if n > 1), missing=len(missing),
        missing_examples=missing[:5], unknown_ids=sorted({r.get("question_id") for r in mine} - set(questions)),
        incomplete_records=incomplete, gold_mismatches=gold, split_or_source_mismatches=split_source,
        no_extracted_answer=no_extract, truncated=truncated, empty_text=empty, regrade_mismatches=regrade,
        accuracy=sum(bool(r.get("correct")) for r in graded) / len(graded) if graded else 0.0,
        served_by=dict(Counter(r.get("served_by") for r in ok if r.get("served_by"))),
        reported_cost_usd=sum(_cost(r, config) for r in ok),
        records_without_reported_cost=sum(r.get("provider_cost_usd") is None for r in ok),
        examples=examples,
    )


def validate(questions: list[Question], answer_files: list[str | Path],
             config: GatewayConfig | None = None) -> dict[str, Any]:
    by_id = {q.id: q for q in questions}
    reports: list[ModelReport] = []
    complete: dict[str, set[str]] = {}
    for path in answer_files:
        records = load_records([path])
        for model in sorted({r.get("model") for r in records if r.get("model")}):
            report = check_model(str(path), model, records, by_id, config)
            reports.append(report)
            complete[model] = {r.get("question_id") for r in records
                               if r.get("model") == model and not r.get("error")} & set(by_id)
    everywhere = set.intersection(*complete.values()) if complete else set()
    return {
        "questions": len(questions),
        "splits": dict(Counter(f"{q.split}/{q.source}" for q in questions)),
        "question_problems": question_problems(questions),
        "models": reports,
        "answered_by_every_model": len(everywhere),
        "answered_by_every_model_by_split": dict(Counter(by_id[q].split for q in everywhere)),
    }


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Dataset quality report", "",
             f"Questions: {result['questions']} ({', '.join(f'{k} {v}' for k, v in sorted(result['splits'].items()))}). "
             f"Problems in the question set: {len(result['question_problems'])}.", ""]
    lines += ["| File | Model | Answered | Missing | Errors (kept, retried later) | Duplicates | Unknown ids | "
              "Incomplete | Gold mismatch | No extracted answer | Truncated | Re-grade mismatch | Accuracy | Hosts |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in result["models"]:
        errors = ", ".join(f"{k} {v}" for k, v in m.errors.items()) or "0"
        hosts = ", ".join(f"{k} {v}" for k, v in m.served_by.items()) or "-"
        lines.append(f"| {Path(m.file).name} | {m.model} | {m.answered} | {m.missing} | {errors} | {m.duplicates} | "
                     f"{len(m.unknown_ids)} | {m.incomplete_records} | {m.gold_mismatches} | {m.no_extracted_answer} | "
                     f"{m.truncated} | {m.regrade_mismatches} | {m.accuracy:.1%} | {hosts} |")
    lines += ["", f"Questions answered by every model: {result['answered_by_every_model']} "
                  f"({', '.join(f'{k} {v}' for k, v in sorted(result['answered_by_every_model_by_split'].items()))}).",
              "", "| Model | Cost (reported, else list-price estimate) | Answers without a reported cost |",
              "|---|---|---|"]
    for m in result["models"]:
        lines.append(f"| {m.model} | ${m.reported_cost_usd:.5f} | {m.records_without_reported_cost} |")
    shown = [(m.model, kind, ex) for m in result["models"] for kind, ex in m.examples.items()]
    if result["question_problems"] or shown:
        lines += ["", "## Examples"]
        for problem in result["question_problems"][:10]:
            lines.append(f"- question set: {problem}")
        for model, kind, ex in shown:
            lines.append(f"- {model}, {kind}: " + "; ".join(ex))
    problems = len(result["question_problems"]) + sum(m.problems for m in result["models"])
    lines += ["", f"Integrity problems (duplicates, unknown ids, incomplete records, gold/split mismatches, "
                  f"re-grade mismatches): {problems}. Answers with no extracted final answer and truncated answers "
                  f"are model failures and stay in the data as wrong answers."]
    return "\n".join(lines)


def to_json(result: dict[str, Any]) -> dict[str, Any]:
    return {**result, "models": [asdict(m) for m in result["models"]]}


def integrity_problems(result: dict[str, Any]) -> int:
    return len(result["question_problems"]) + sum(m.problems for m in result["models"])
