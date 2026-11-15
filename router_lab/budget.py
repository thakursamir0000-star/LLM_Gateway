"""A spending cap for paid collection runs (OpenRouter bills every call).

- Spend is the cost the provider reported for each call, or the list-price
  estimate from the config when it reported none.
- Before every request the guard computes a worst case for that call: the
  prompt's tokens estimated generously plus the whole output-token cap, at the
  model's prices. A request is never started if spend + worst case would pass
  the cap, so the cap holds even if every call ran to its output limit.
- A reported cost clearly above the list-price estimate for the tokens used
  (the host charges more than the config says) is unexpected billing: the run
  saves that answer and stops.
- A tag groups one budget's records across runs: a resumed run counts what
  earlier runs with the same tag spent. Records without the tag (earlier,
  already-paid collections) never count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gateway.config import ModelSpec
from gateway.providers.base import Completion, message_text

DEFAULT_MAX_OUTPUT_TOKENS = 4096  # assumed output cap when a request sets none
PROMPT_OVERHEAD_TOKENS = 64  # chat-template tokens around the messages
UNEXPECTED_FACTOR = 1.25  # reported cost above this x the list-price estimate stops the run


def record_cost(record: dict[str, Any]) -> float:
    """What a saved answer cost: the reported figure, else the list-price estimate."""
    reported = record.get("provider_cost_usd")
    return float(reported) if isinstance(reported, int | float) else float(record.get("cost_usd") or 0.0)


def spent_with_tag(records: list[dict[str, Any]], tag: str | None) -> float:
    """What earlier runs under ``tag`` spent (0 without a tag)."""
    if not tag:
        return 0.0
    return sum(record_cost(r) for r in records if r.get("budget_tag") == tag and not r.get("error"))


def worst_case_cost(spec: ModelSpec, messages: list[dict[str, Any]], params: dict[str, Any]) -> float:
    """The most one request can cost: generous prompt estimate (3 characters a token) + full output cap."""
    chars = sum(len(message_text(m.get("content"))) for m in messages)
    prompt_tokens = chars // 3 + PROMPT_OVERHEAD_TOKENS
    merged = {**spec.params, **params}
    output_cap = merged.get("max_completion_tokens") or merged.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS
    return spec.cost(prompt_tokens, int(output_cap))


@dataclass
class Budget:
    limit_usd: float
    spent_usd: float = 0.0  # includes earlier runs with the same tag
    tag: str | None = None

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def allows(self, worst_case_usd: float) -> bool:
        return self.spent_usd + worst_case_usd <= self.limit_usd + 1e-12

    def charge(self, spec: ModelSpec, completion: Completion) -> tuple[float, bool]:
        """Add one call's cost; returns (cost, unexpected) where unexpected flags billing above list price."""
        estimate = spec.cost(completion.prompt_tokens, completion.completion_tokens)
        reported = completion.provider_cost_usd
        cost = reported if reported is not None else estimate
        self.spent_usd += cost
        unexpected = reported is not None and reported > estimate * UNEXPECTED_FACTOR + 1e-7
        return cost, unexpected
