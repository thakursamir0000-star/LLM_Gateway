"""Compare models (possibly from different providers) on the questions they all answered.

Records come from one or more answer files (each provider keeps its own).
Per model: accuracy with a 95% Wilson interval, mean input/output tokens,
cost per 1,000 questions (list-price estimate from the config, and the
provider-reported cost when every answer has one), latency from the model's
own time (records written before wait_ms existed may include pacing waits,
so they are left out of latency), and the failures and rate limits the
collection hit.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gateway.config import GatewayConfig
from gateway.providers.base import QUOTA_EXHAUSTED, RATE_LIMIT


def load_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    """All parsable records from the answer files that exist."""
    records = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def latest_answers(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest successful answer per (question id, model): duplicates collapse to one."""
    answers = {}
    for record in records:
        if not record.get("error"):
            answers[(record["question_id"], record["model"])] = record
    return answers


def common_questions(answers: dict[tuple[str, str], dict[str, Any]], models: list[str]) -> set[str]:
    sets = [{qid for (qid, m) in answers if m == model} for model in models]
    return set.intersection(*sets) if sets else set()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass
class ModelStats:
    model: str
    provider: str
    provider_model: str
    n: int
    correct: int
    accuracy: float
    ci_low: float
    ci_high: float
    prompt_tokens_mean: float
    completion_tokens_mean: float
    cost_per_1k_estimated: float  # config list prices x tokens
    cost_per_1k_reported: float | None  # provider-reported (OpenRouter), if every answer has it
    latency_n: int  # answers with a clean model time
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    calls: int  # every call the collection made for this model (all questions, all files)
    failures: int
    rate_limited: int  # per-minute or daily limits hit (after the registry's own retries)
    failure_kinds: dict[str, int] = field(default_factory=dict)
    served_by: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)


def model_stats(model: str, answers: dict[tuple[str, str], dict[str, Any]], records: list[dict[str, Any]],
                question_ids: set[str], config: GatewayConfig) -> ModelStats:
    rows = [answers[(qid, model)] for qid in sorted(question_ids)]
    n = len(rows)
    correct = sum(bool(r.get("correct")) for r in rows)
    low, high = wilson(correct, n)
    spec = config.models.get(model)
    estimated = [spec.cost(r.get("prompt_tokens", 0), r.get("completion_tokens", 0)) if spec else 0.0 for r in rows]
    reported = [r.get("provider_cost_usd") for r in rows]
    clean_latency = [r["latency_ms"] for r in rows if "wait_ms" in r and r.get("latency_ms") is not None]
    mine = [r for r in records if r.get("model") == model]
    failures = [r for r in mine if r.get("error")]
    kinds = Counter(r.get("error_kind", "other") for r in failures)
    sources: dict[str, dict[str, float]] = {}
    for source in sorted({r.get("source", "?") for r in rows}):
        subset = [r for r in rows if r.get("source", "?") == source]
        sources[source] = {"n": len(subset), "accuracy": sum(bool(r.get("correct")) for r in subset) / len(subset)}
    return ModelStats(
        model=model,
        provider=(rows[0].get("provider") if rows else None) or (spec.provider if spec else "?"),
        provider_model=(rows[0].get("provider_model") if rows else None) or (spec.model if spec else "?"),
        n=n, correct=correct, accuracy=correct / n if n else 0.0, ci_low=low, ci_high=high,
        prompt_tokens_mean=statistics.fmean(r.get("prompt_tokens", 0) for r in rows) if rows else 0.0,
        completion_tokens_mean=statistics.fmean(r.get("completion_tokens", 0) for r in rows) if rows else 0.0,
        cost_per_1k_estimated=statistics.fmean(estimated) * 1000 if rows else 0.0,
        cost_per_1k_reported=(statistics.fmean(reported) * 1000
                              if rows and all(isinstance(c, int | float) for c in reported) else None),
        latency_n=len(clean_latency),
        latency_mean_ms=statistics.fmean(clean_latency) if clean_latency else None,
        latency_p50_ms=_pct(clean_latency, 0.5), latency_p95_ms=_pct(clean_latency, 0.95),
        calls=len(mine), failures=len(failures), rate_limited=kinds[RATE_LIMIT] + kinds[QUOTA_EXHAUSTED],
        failure_kinds=dict(kinds), served_by=dict(Counter(r.get("served_by") for r in rows if r.get("served_by"))),
        by_source=sources,
    )


def agreement(answers: dict[tuple[str, str], dict[str, Any]], a: str, b: str,
              question_ids: set[str]) -> dict[str, int]:
    """How often two models (e.g. one model on two hosts) are right on the same questions."""
    out = {"both_right": 0, "both_wrong": 0, f"only_{a}": 0, f"only_{b}": 0}
    for qid in question_ids:
        ra, rb = bool(answers[(qid, a)].get("correct")), bool(answers[(qid, b)].get("correct"))
        key = "both_right" if ra and rb else "both_wrong" if not ra and not rb else f"only_{a}" if ra else f"only_{b}"
        out[key] += 1
    return out


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def markdown(stats: list[ModelStats], pairs: dict[str, dict[str, int]]) -> str:
    n = stats[0].n if stats else 0
    lines = [
        f"Questions answered by every model: {n}.",
        "",
        "| Model | Provider (host) | Accuracy (95% CI) | Input tok | Output tok | Cost / 1k q (est.) | "
        "Cost / 1k q (reported) | Latency mean / p50 / p95 ms (n) | Calls | Failures | Rate limited |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        hosts = ", ".join(f"{h} {c}" for h, c in s.served_by.items())
        provider = f"{s.provider} ({hosts})" if hosts else s.provider
        reported = "-" if s.cost_per_1k_reported is None else f"${s.cost_per_1k_reported:.4f}"
        lines.append(
            f"| {s.model} (`{s.provider_model}`) | {provider} | {s.accuracy:.1%} ({s.ci_low:.0%}-{s.ci_high:.0%}) | "
            f"{s.prompt_tokens_mean:.0f} | {s.completion_tokens_mean:.0f} | ${s.cost_per_1k_estimated:.4f} | "
            f"{reported} | {_ms(s.latency_mean_ms)} / {_ms(s.latency_p50_ms)} / {_ms(s.latency_p95_ms)} "
            f"({s.latency_n}) | {s.calls} | {s.failures} | {s.rate_limited} |")
    sources = sorted({src for s in stats for src in s.by_source})
    if sources:
        lines += ["", "| Model | " + " | ".join(f"{src} accuracy (n)" for src in sources) + " |",
                  "|---|" + "---|" * len(sources)]
        for s in stats:
            cells = [f"{s.by_source[src]['accuracy']:.1%} ({int(s.by_source[src]['n'])})" if src in s.by_source
                     else "-" for src in sources]
            lines.append(f"| {s.model} | " + " | ".join(cells) + " |")
    for name, counts in pairs.items():
        lines += ["", f"{name}: " + ", ".join(f"{k} {v}" for k, v in counts.items())]
    lines += ["", "Latency is the model's own time; answers saved before pacing waits were recorded separately "
                  "are left out of it (n shows how many count). Calls, failures and rate limits cover every call "
                  "the collection made for the model, after the registry's own retries."]
    return "\n".join(lines)


def to_json(stats: list[ModelStats], pairs: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {"models": [asdict(s) for s in stats], "agreement": pairs}
