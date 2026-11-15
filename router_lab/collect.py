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
from gateway.providers.base import AUTH, QUOTA_EXHAUSTED, ProviderError, redact
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
    with log.lock():  # one writer per answer file, so no duplicate records
        return await _collect(questions, models, registry, log, params or {}, max_consecutive_errors, report)


async def _collect(questions: list[Question], models: list[str], registry: ModelRegistry, log: AnswerLog,
                   params: dict[str, Any], max_consecutive_errors: int,
                   report: Callable[[str], None]) -> CollectSummary:
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
            spec = registry.spec(alias)
            record: dict[str, Any] = {"question_id": q.id, "split": q.split, "source": q.source, "model": alias,
                                      "provider": spec.provider, "provider_model": spec.model,
                                      "ts": datetime.now(UTC).isoformat(timespec="seconds")}
            try:
                completion = await registry.complete(alias, messages_for(q), params)
            except ProviderError as e:
                record.update(error=redact(str(e))[:500], error_kind=e.kind, status=e.status,
                              latency_ms=round((time.perf_counter() - started) * 1000, 1))
                log.append(record)
                summary.errors += 1
                consecutive += 1
                report(f"  {q.id} {alias}: {e}")
                if e.kind == QUOTA_EXHAUSTED:
                    summary.stopped = (f"out of credits for {alias}" if e.status == 402
                                       else f"daily quota used up for {alias}")
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
            total_ms = (time.perf_counter() - started) * 1000
            model_ms = completion.latency_ms if completion.latency_ms is not None else total_ms
            # latency_ms is the model's own time; wait_ms is local pacing and retry waits on top of it.
            record.update(correct=correct, extracted=extracted, gold=q.answer, text=completion.text,
                          finish_reason=completion.finish_reason, prompt_tokens=completion.prompt_tokens,
                          completion_tokens=completion.completion_tokens, latency_ms=round(model_ms, 1),
                          wait_ms=round(max(0.0, total_ms - model_ms), 1),
                          # list-price estimate from the config, and the provider's own figure when it reports one
                          cost_usd=round(spec.cost(completion.prompt_tokens, completion.completion_tokens), 10),
                          provider_cost_usd=completion.provider_cost_usd, served_by=completion.served_by,
                          error=None)
            log.append(record)
            summary.answered += 1
            position = summary.answered + summary.skipped + summary.errors
            report(f"[{position}/{total}] {q.id} {alias}: {'right' if correct else 'wrong'} "
                   f"({completion.prompt_tokens}+{completion.completion_tokens} tokens)")
    return summary
