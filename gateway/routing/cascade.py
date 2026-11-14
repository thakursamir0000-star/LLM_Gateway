"""Cascade: answer with the cheap model, check the answer, escalate if unsure.

The check is a heuristic confidence in [0, 1] from warning signs in the cheap
answer; the gateway escalates to the strong model when it is below the
routing threshold (the same cost/quality knob as the router). It costs no
extra model call, but it only catches answers that *look* wrong (empty,
truncated, refused, hedged, looping); a confident wrong answer passes. The
router evaluation measures how much that matters.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_HEDGES = re.compile(
    r"\bI(?: am|'m) not (?:sure|certain)\b|\bI don'?t know\b|\bI do not know\b|\bcannot be determined\b|"
    r"\bcan'?t be determined\b|\bnot enough information\b|\binsufficient information\b|"
    r"\bI(?: am|'m) unable to\b|\bI can(?:not|'t) (?:help|answer|assist|determine)\b|\bas an AI\b|"
    r"\bunclear from\b|\bI might be wrong\b",
    re.IGNORECASE,
)

EMPTY, BLOCKED, TRUNCATED, REPETITIVE, HEDGED, SHORT, OK = 0.0, 0.05, 0.15, 0.2, 0.3, 0.7, 0.9


@dataclass(frozen=True)
class AnswerCheck:
    confidence: float
    reason: str


def _is_repetitive(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    return bool(lines) and Counter(lines).most_common(1)[0][1] >= 4


def check_answer(question: str, answer: str, finish_reason: str = "stop") -> AnswerCheck:
    """Heuristic confidence that the cheap answer is usable."""
    if finish_reason == "content_filter":
        return AnswerCheck(BLOCKED, "blocked by the provider's filter")
    if not answer.strip():
        return AnswerCheck(EMPTY, "empty answer")
    if finish_reason == "length":
        return AnswerCheck(TRUNCATED, "truncated at the token limit")
    if _is_repetitive(answer):
        return AnswerCheck(REPETITIVE, "repeats itself")
    hedge = _HEDGES.search(answer)
    if hedge:
        return AnswerCheck(HEDGED, f"hedged ({hedge.group(0)!r})")
    if len(answer.split()) < 4 and len(question) > 300:
        return AnswerCheck(SHORT, "very short answer to a long request")
    return AnswerCheck(OK, "no warning signs")


def should_escalate(check: AnswerCheck, threshold: float) -> bool:
    return check.confidence < threshold
