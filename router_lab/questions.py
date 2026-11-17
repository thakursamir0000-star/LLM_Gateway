"""Benchmark questions (GSM8K maths, MMLU multiple choice): prompts, grading, sampling.

The prompt is what an app would send through the gateway, and the router is
trained and evaluated on exactly that text. Prompts avoid the phrases the
rule-based router looks for ("step by step" and so on), so the rules aren't
handed the answer by our own wording.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LETTERS = "ABCDEFGHIJ"


@dataclass
class Question:
    id: str
    source: str  # gsm8k | mmlu
    split: str  # train | test (router training / held-out evaluation)
    question: str
    answer: str  # gold: a number (gsm8k) or a letter (mmlu)
    choices: list[str] | None = None
    subject: str | None = None


# ── loading public datasets (rows as returned by Hugging Face `datasets`) ──

def gsm8k_gold(answer: str) -> str:
    """The final number of a GSM8K solution ("... #### 1,234" -> "1234")."""
    return normalise_number(answer.split("####")[-1]) or answer.strip()


def from_gsm8k(rows: list[dict[str, Any]]) -> list[Question]:
    return [Question(f"gsm8k-{i:04d}", "gsm8k", "", r["question"].strip(), gsm8k_gold(r["answer"]))
            for i, r in enumerate(rows)]


def from_mmlu(rows: list[dict[str, Any]]) -> list[Question]:
    return [Question(f"mmlu-{i:05d}", "mmlu", "", r["question"].strip(), LETTERS[int(r["answer"])],
                     [str(c) for c in r["choices"]], r.get("subject"))
            for i, r in enumerate(rows)]


def sample_and_split(pools: dict[str, list[Question]], counts: dict[str, int], test_fraction: float,
                     seed: int = 0) -> list[Question]:
    """Sample ``counts[source]`` questions per source, mark a test share, and shuffle sources together.

    The test split is fixed here, before any model sees a question, so the
    router evaluation is always on held-out questions. Sources are shuffled
    together so a partly finished collection still covers both.
    """
    rng = random.Random(seed)
    sampled: list[Question] = []
    for source, n in counts.items():
        pool = pools[source]
        if n > len(pool):
            raise ValueError(f"{source}: asked for {n} questions, only {len(pool)} available")
        picked = rng.sample(pool, n)
        n_test = round(n * test_fraction)
        for i, q in enumerate(picked):
            sampled.append(Question(**{**asdict(q), "split": "test" if i < n_test else "train"}))
    rng.shuffle(sampled)
    return sampled


def sample(questions: list[Question], n: int, seed: int = 0) -> list[Question]:
    """A seeded random ``n`` of ``questions``, kept in their original order (all of them if n >= len)."""
    if n >= len(questions):
        return list(questions)
    keep = {q.id for q in random.Random(seed).sample(questions, n)}
    return [q for q in questions if q.id in keep]


def save_questions(path: str | Path, questions: list[Question]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def load_questions(path: str | Path) -> list[Question]:
    with Path(path).open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


# ── prompts ─────────────────────────────────────────────────────────

def format_prompt(q: Question) -> str:
    if q.source == "gsm8k":
        return f'{q.question}\n\nEnd your reply with a final line of the form "Answer: <number>".'
    options = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(q.choices or []))
    return (f"{q.question}\n\n{options}\n\nChoose the correct option. "
            f'End your reply with a final line of the form "Answer: <letter>".')


def messages_for(q: Question) -> list[dict[str, str]]:
    return [{"role": "user", "content": format_prompt(q)}]


# ── grading ─────────────────────────────────────────────────────────

_ANSWER_LINE = re.compile(r"answer\**\s*(?:is\b\s*:?|[:：])\s*\**\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def normalise_number(text: str) -> str | None:
    """The first number in ``text`` without $ or commas, as a canonical string."""
    match = _NUMBER.search(text)
    if not match:
        return None
    raw = match.group(0).replace("$", "").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return str(int(value)) if value.is_integer() else repr(round(value, 6))


def extract_number(text: str) -> str | None:
    """The model's final number: the last "Answer:" line, else \\boxed{}, else the last number."""
    lines = [m.group(1) for m in _ANSWER_LINE.finditer(text)]
    for candidate in reversed(lines):
        number = normalise_number(candidate)
        if number is not None:
            return number
    boxed = _BOXED.findall(text)
    if boxed:
        number = normalise_number(boxed[-1])
        if number is not None:
            return number
    numbers = _NUMBER.findall(text)
    return normalise_number(numbers[-1]) if numbers else None


def extract_choice(text: str, n_choices: int = 4) -> str | None:
    """The model's chosen letter: from the last "Answer:" line, else a lone letter at the end."""
    valid = LETTERS[:n_choices]
    letter = rf"[{valid}]"
    for candidate in reversed([m.group(1) for m in _ANSWER_LINE.finditer(text)]):
        match = re.search(rf"(?<![A-Za-z]){letter}(?![A-Za-z])", candidate)
        if match:
            return match.group(0)
    match = re.search(rf"(?<![A-Za-z])\(?\**({letter})\**\)?\.?\s*$", text.strip())
    return match.group(1) if match else None


def grade(q: Question, text: str) -> tuple[bool, str | None]:
    """(correct, extracted answer)."""
    if q.source == "gsm8k":
        extracted = extract_number(text)
        if extracted is None:
            return False, None
        try:
            return abs(float(extracted) - float(q.answer)) < 1e-6, extracted
        except ValueError:
            return False, extracted
    extracted = extract_choice(text, len(q.choices or "ABCD"))
    return extracted == q.answer, extracted
