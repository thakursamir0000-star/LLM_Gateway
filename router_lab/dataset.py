"""Pair each question with both models' answers and label it for the router.

Label ``cheap_ok`` = the cheap model was right, or the strong model was wrong
too. When both are wrong, paying for the strong model buys nothing, so the
cheap model is "good enough" there. This is the routing target: send a
request to the strong model only where it actually helps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gateway.routing import routing_text
from router_lab.questions import Question, messages_for


@dataclass
class Pair:
    question: Question
    cheap: dict[str, Any]
    strong: dict[str, Any]

    @property
    def cheap_ok(self) -> bool:
        return bool(self.cheap["correct"]) or not bool(self.strong["correct"])

    @property
    def text(self) -> str:
        """What the router reads for this question (as in the gateway)."""
        return routing_text(messages_for(self.question))


def build_pairs(questions: list[Question], answers: dict[tuple[str, str], dict[str, Any]], cheap: str,
                strong: str, split: str | None = None) -> list[Pair]:
    """Questions (of ``split``) that both models answered without an error."""
    pairs = []
    for q in questions:
        if split and q.split != split:
            continue
        a, b = answers.get((q.id, cheap)), answers.get((q.id, strong))
        if a is not None and b is not None:
            pairs.append(Pair(q, a, b))
    return pairs
