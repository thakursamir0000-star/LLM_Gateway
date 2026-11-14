"""A rule-based baseline router.

Hand-written signals of a hard request (long context, code, maths,
multi-step reasoning, several questions, a long conversation) add to a
difficulty estimate; the score is ``1 - difficulty``. The weights were set by
hand, not trained: this is the baseline the trained classifier has to beat.
"""

from __future__ import annotations

import re
from typing import Any

from gateway.routing.base import RouteScore, routing_text, safe_text

_CODE = re.compile(r"```|\bdef \w+\(|\bclass \w+|\bfunction\b|\bimport \w+|#include|\bSELECT\b.+\bFROM\b|"
                   r"Traceback|\bstack trace\b|\bregex\b|\bcompile[sr]?\b|\bSQL\b|\bAPI\b", re.IGNORECASE)
_MATH = re.compile(r"\d+\s*[-+*/^=<>]\s*\d+|\bintegral\b|\bderivative\b|\bprove\b|\btheorem\b|\bequations?\b|"
                   r"\bprobability\b|\bmatri(x|ces)\b|\beigen|\bsolve for\b|\blimit of\b|\bpercent(age)?\b",
                   re.IGNORECASE)
_REASONING = re.compile(r"step[- ]by[- ]step|\bexplain why\b|\banaly[sz]e\b|\bcompare\b|\btrade-?offs?\b|"
                        r"\bdesign an?\b|\barchitecture\b|\boptimi[sz]e\b|\bdebug\b|\bderive\b|\bjustify\b|"
                        r"\bpros and cons\b|\bevaluate\b", re.IGNORECASE)
_SIMPLE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you)\b|\btranslate\b|\bsummari[sz]e\b|\brephrase\b|"
                     r"\bfix (the )?(grammar|spelling|typos?)\b|\bdefine\b|\bwhat is the capital\b", re.IGNORECASE)

BASE_DIFFICULTY = 0.15


class RulesRouter:
    name = "rules"

    def __init__(self, note: str = ""):
        self.description = "rule-based baseline" + (f" ({note})" if note else "")

    def score(self, messages: list[dict[str, Any]]) -> RouteScore:
        text = routing_text(messages)
        context_tokens = sum(len(safe_text(m.get("content"))) for m in messages) / 4
        turns = sum(1 for m in messages if m.get("role") in ("user", "assistant"))

        signals: list[tuple[float, str]] = []
        if context_tokens > 2000:
            signals.append((0.25, "long context"))
        elif context_tokens > 500:
            signals.append((0.10, "medium context"))
        if _CODE.search(text):
            signals.append((0.20, "code"))
        if _MATH.search(text):
            signals.append((0.20, "maths"))
        if _REASONING.search(text):
            signals.append((0.20, "multi-step reasoning"))
        if text.count("?") >= 3:
            signals.append((0.10, "several questions"))
        if turns > 6:
            signals.append((0.10, "long conversation"))
        if _SIMPLE.search(text) and len(signals) <= 1:
            signals.append((-0.10, "simple task"))

        difficulty = min(1.0, max(0.0, BASE_DIFFICULTY + sum(weight for weight, _ in signals)))
        reason = ", ".join(name for _, name in signals) or "short general request"
        return RouteScore(round(1.0 - difficulty, 4), self.name, reason)
