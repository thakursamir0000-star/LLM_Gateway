"""Run the cheap and the strong model on benchmark questions and record who got them right.

Calls go through the gateway's ModelRegistry, so they get the model's default
params, pacing under its per-minute limits and bounded retries. Every answer
is saved as soon as it arrives. The run stops early when retrying can't help:
a daily quota is used up, the API key is missing or rejected, or several
calls in a row failed. With a Budget (router_lab.budget) it also stops before
any request that could take spending past the cap, and on unexpected billing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from gateway.providers import ModelRegistry
from gateway.providers.base import AUTH, QUOTA_EXHAUSTED, ProviderError, redact
from router_lab.budget import Budget, worst_case_cost
from router_lab.progress import AnswerLog
from router_lab.questions import Question, grade, messages_for


@dataclass
class CollectSummary:
    answered: int = 0
    skipped: int = 0  # already in the answer log
    errors: int = 0
    stopped: str | None = None  # why the run stopped early (None = finished)
    spent_usd: float = 0.0  # this run's spend (reported cost, else list-price estimate)


async def collect(questions: list[Question], models: list[str], registry: ModelRegistry, log: AnswerLog, *,
                  params: dict[str, Any] | None = None, max_consecutive_errors: int = 5,
                  budget: Budget | None = None, report: Callable[[str], None] = print) -> CollectSummary:
    with log.lock():  # one writer per answer file, so no duplicate records
        return await _collect(questions, models, registry, log, params or {}, max_consecutive_errors, budget,
                              report)


async def _collect(questions: list[Question], models: list[str], registry: ModelRegistry, log: AnswerLog,
                   params: dict[str, Any], max_consecutive_errors: int, budget: Budget | None,
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
            spec = registry.spec(alias)
            if budget is not None:
                worst = worst_case_cost(spec, messages_for(q), params)
                if not budget.allows(worst):
                    summary.stopped = (f"budget: the next call could cost up to ${worst:.6f} and only "
                                       f"${budget.remaining_usd:.6f} of ${budget.limit_usd:.4f} is left")
                    return summary
            started = time.perf_counter()
            record: dict[str, Any] = {"question_id": q.id, "split": q.split, "source": q.source, "model": alias,
                                      "provider": spec.provider, "provider_model": spec.model,
                                      "ts": datetime.now(UTC).isoformat(timespec="seconds")}
            if budget is not None and budget.tag:
                record["budget_tag"] = budget.tag
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
            log.append(record)  # saved before any budget decision, so a paid answer is never lost
            summary.answered += 1
            if budget is not None:
                cost, unexpected = budget.charge(spec, completion)
                summary.spent_usd += cost
                if unexpected:
                    summary.stopped = (f"unexpected billing: {q.id} {alias} was reported at "
                                       f"${completion.provider_cost_usd:.6f}, above the list-price estimate "
                                       f"${record['cost_usd']:.6f}")
                    return summary
            else:
                summary.spent_usd += completion.provider_cost_usd if completion.provider_cost_usd is not None                     else record["cost_usd"]
            position = summary.answered + summary.skipped + summary.errors
            report(f"[{position}/{total}] {q.id} {alias}: {'right' if correct else 'wrong'} "
                   f"({completion.prompt_tokens}+{completion.completion_tokens} tokens)")
    return summary
