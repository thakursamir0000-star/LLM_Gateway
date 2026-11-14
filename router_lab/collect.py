"""Run the cheap and the strong model on benchmark questions and record who got them right.

Calls go through the gateway's ModelRegistry, so they get the model's default
params, pacing under its per-minute limits and bounded retries. Every answer
is saved as soon as it arrives. The run stops early when retrying can't help:
a daily quota is used up, the API key is missing or rejected, or several
calls in a row failed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from gateway.providers import ModelRegistry
from gateway.providers.base import AUTH, QUOTA_EXHAUSTED, ProviderError
from router_lab.progress import AnswerLog
from router_lab.questions import Question, grade, messages_for


@dataclass
class CollectSummary:
    answered: int = 0
    skipped: int = 0  # already in the answer log
    errors: int = 0
    stopped: str | None = None  # why the run stopped early (None = finished)


async def collect(questions: list[Question], models: list[str], registry: ModelRegistry, log: AnswerLog, *,
                  params: dict[str, Any] | None = None, max_consecutive_errors: int = 5,
                  report: Callable[[str], None] = print) -> CollectSummary:
    summary = CollectSummary()
    done = log.answers()
    consecutive = 0
    total = len(questions) * len(models)
    for q in questions:
        for alias in models:
            if (q.id, alias) in done:
                summary.skipped += 1
                continue
            started = time.perf_counter()
            record: dict[str, Any] = {"question_id": q.id, "split": q.split, "source": q.source, "model": alias,
                                      "provider_model": registry.spec(alias).model,
                                      "ts": datetime.now(UTC).isoformat(timespec="seconds")}
            try:
                completion = await registry.complete(alias, messages_for(q), params or {})
            except ProviderError as e:
                record.update(error=str(e)[:500], error_kind=e.kind,
                              latency_ms=round((time.perf_counter() - started) * 1000, 1))
                log.append(record)
                summary.errors += 1
                consecutive += 1
                report(f"  {q.id} {alias}: {e}")
                if e.kind == QUOTA_EXHAUSTED:
                    summary.stopped = f"daily quota used up for {alias}"
                    return summary
                if e.kind == AUTH:
                    summary.stopped = f"API key problem for {alias}: {e}"
                    return summary
                if consecutive >= max_consecutive_errors:
                    summary.stopped = f"{consecutive} errors in a row"
                    return summary
                continue
            consecutive = 0
            correct, extracted = grade(q, completion.text)
            record.update(correct=correct, extracted=extracted, gold=q.answer, text=completion.text,
                          finish_reason=completion.finish_reason, prompt_tokens=completion.prompt_tokens,
                          completion_tokens=completion.completion_tokens,
                          latency_ms=round((time.perf_counter() - started) * 1000, 1), error=None)
            log.append(record)
            summary.answered += 1
            position = summary.answered + summary.skipped + summary.errors
            report(f"[{position}/{total}] {q.id} {alias}: {'right' if correct else 'wrong'} "
                   f"({completion.prompt_tokens}+{completion.completion_tokens} tokens)")
    return summary
