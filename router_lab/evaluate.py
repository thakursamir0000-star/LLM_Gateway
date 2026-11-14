"""Compare routing strategies on held-out questions, from answers already collected.

The collection step ran both models on every question, so each strategy can
be replayed offline: a question routed to the cheap model scores the cheap
model's answer and costs its tokens, and so on. No API call is needed here.

Strategies:
- always-cheap, always-strong;
- random: send a share ``s`` to the strong model at random (expected values,
  the straight line between the two "always" points);
- rules and classifier routers swept over the threshold;
- cascade: the cheap answer, escalated when its check is below the threshold
  (pays for both calls when it escalates).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gateway.config import GatewayConfig
from gateway.routing.cascade import check_answer
from router_lab.dataset import Pair


@dataclass
class Item:
    """One held-out question with both models' outcomes."""

    id: str
    cheap_correct: bool
    strong_correct: bool
    cheap_cost: float
    strong_cost: float
    cheap_latency_s: float
    strong_latency_s: float
    cheap_text: str
    cheap_finish: str
    text: str  # what the router reads


@dataclass
class Point:
    strategy: str
    threshold: float | None
    accuracy: float
    cost_usd: float  # total over the evaluation set
    cost_per_1k: float  # USD per 1,000 requests
    strong_share: float  # share of requests that called the strong model
    llm_latency_s: float  # mean model time per request (both calls for an escalated cascade)
    n: int


def items_from_pairs(pairs: list[Pair], config: GatewayConfig, cheap: str, strong: str) -> list[Item]:
    cheap_spec, strong_spec = config.models[cheap], config.models[strong]
    return [Item(
        id=p.question.id,
        cheap_correct=bool(p.cheap["correct"]), strong_correct=bool(p.strong["correct"]),
        cheap_cost=cheap_spec.cost(p.cheap.get("prompt_tokens", 0), p.cheap.get("completion_tokens", 0)),
        strong_cost=strong_spec.cost(p.strong.get("prompt_tokens", 0), p.strong.get("completion_tokens", 0)),
        cheap_latency_s=(p.cheap.get("latency_ms") or 0) / 1000, strong_latency_s=(p.strong.get("latency_ms") or 0) / 1000,
        cheap_text=p.cheap.get("text") or "", cheap_finish=p.cheap.get("finish_reason") or "stop", text=p.text,
    ) for p in pairs]


def _point(strategy: str, threshold: float | None, correct: np.ndarray, cost: np.ndarray, strong_used: np.ndarray,
           latency: np.ndarray) -> Point:
    n = len(correct)
    total = float(cost.sum())
    return Point(strategy, threshold, float(correct.mean()), total, total / n * 1000, float(strong_used.mean()),
                 float(latency.mean()), n)


def route_point(items: list[Item], to_strong: Sequence[bool], strategy: str, threshold: float | None) -> Point:
    """A strategy that sends each question to exactly one model."""
    s = np.asarray(to_strong, dtype=bool)
    correct = np.where(s, [i.strong_correct for i in items], [i.cheap_correct for i in items])
    cost = np.where(s, [i.strong_cost for i in items], [i.cheap_cost for i in items])
    latency = np.where(s, [i.strong_latency_s for i in items], [i.cheap_latency_s for i in items])
    return _point(strategy, threshold, correct, cost, s, latency)


def cascade_point(items: list[Item], escalate: Sequence[bool], threshold: float | None) -> Point:
    """Cheap first; escalated questions pay for both calls and score the strong answer."""
    e = np.asarray(escalate, dtype=bool)
    correct = np.where(e, [i.strong_correct for i in items], [i.cheap_correct for i in items])
    cheap_cost = np.array([i.cheap_cost for i in items])
    cost = cheap_cost + np.where(e, [i.strong_cost for i in items], 0.0)
    latency = np.array([i.cheap_latency_s for i in items]) + np.where(e, [i.strong_latency_s for i in items], 0.0)
    return _point("cascade", threshold, correct, cost, e, latency)


def random_point(items: list[Item], strong_share: float) -> Point:
    """Expected result of sending a random ``strong_share`` of requests to the strong model."""
    cheap = route_point(items, [False] * len(items), "random", None)
    strong = route_point(items, [True] * len(items), "random", None)

    def mix(a: float, b: float) -> float:
        return (1 - strong_share) * a + strong_share * b

    return Point("random", None, mix(cheap.accuracy, strong.accuracy), mix(cheap.cost_usd, strong.cost_usd),
                 mix(cheap.cost_per_1k, strong.cost_per_1k), strong_share,
                 mix(cheap.llm_latency_s, strong.llm_latency_s), len(items))


def sweep(items: list[Item], scores: Sequence[float], strategy: str, thresholds: Sequence[float]) -> list[Point]:
    """Router points over thresholds: strong when score < threshold."""
    s = np.asarray(scores, dtype=float)
    return [route_point(items, s < t, strategy, float(t)) for t in thresholds]


def cascade_confidences(items: list[Item]) -> np.ndarray:
    return np.array([check_answer(i.text, i.cheap_text, i.cheap_finish).confidence for i in items])


def cascade_sweep(items: list[Item], thresholds: Sequence[float]) -> list[Point]:
    confidence = cascade_confidences(items)
    return [cascade_point(items, confidence < t, float(t)) for t in thresholds]


def cheapest_within(points: list[Point], target_accuracy: float) -> Point | None:
    """The cheapest point whose accuracy reaches ``target_accuracy``."""
    good = [p for p in points if p.accuracy >= target_accuracy - 1e-12]
    return min(good, key=lambda p: p.cost_usd) if good else None


def router_latency_ms(score: Callable[[str], Any], texts: list[str], warmup: int = 3) -> dict[str, float]:
    """p50/p95/mean time of one ``score`` call per request (the router's added latency)."""
    for text in texts[:warmup]:
        score(text)
    times = []
    for text in texts:
        started = time.perf_counter()
        score(text)
        times.append((time.perf_counter() - started) * 1000)
    ordered = sorted(times)

    def pct(q: float) -> float:
        position = (len(ordered) - 1) * q
        low, high = int(np.floor(position)), int(np.ceil(position))
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"p50_ms": pct(0.5), "p95_ms": pct(0.95), "mean_ms": statistics.fmean(times), "n": len(times)}


# ── report ──────────────────────────────────────────────────────────

def markdown_table(points: list[Point]) -> str:
    lines = ["| Strategy | Threshold | Accuracy | Cost per 1k requests | Sent to strong | Mean model time |",
             "|---|---|---|---|---|---|"]
    for p in points:
        threshold = "-" if p.threshold is None else f"{p.threshold:.2f}"
        lines.append(f"| {p.strategy} | {threshold} | {p.accuracy:.1%} | ${p.cost_per_1k:.4f} | "
                     f"{p.strong_share:.0%} | {p.llm_latency_s:.2f} s |")
    return "\n".join(lines)


def to_dict(points: list[Point]) -> list[dict[str, Any]]:
    return [asdict(p) for p in points]


# Reference palette (light mode): categorical slots 1-3, muted reference ink, chrome.
SERIES = {"classifier": "#2a78d6", "rules": "#eb6834", "cascade": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def plot_cost_vs_accuracy(curves: dict[str, list[Point]], operating: dict[str, Point], cheap: Point, strong: Point,
                          path: str, title: str) -> None:
    """Cost per 1k requests (x) vs accuracy (y): one line per router, random as a reference line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([cheap.cost_per_1k, strong.cost_per_1k], [cheap.accuracy * 100, strong.accuracy * 100],
            color=MUTED, linewidth=1.5, label="random routing", zorder=1)
    for name, points in curves.items():
        ordered = sorted(points, key=lambda p: p.cost_per_1k)
        color = SERIES.get(name, INK_2)
        ax.plot([p.cost_per_1k for p in ordered], [p.accuracy * 100 for p in ordered], color=color, linewidth=2,
                label=name, zorder=2, solid_capstyle="round")
        if name in operating:
            p = operating[name]
            ax.scatter([p.cost_per_1k], [p.accuracy * 100], s=64, color=color, edgecolors=SURFACE, linewidths=2,
                       zorder=3)
    thresholds = {p.threshold for p in operating.values() if p.threshold is not None}
    if thresholds:
        # One legend entry explains every curve's marker (labels on the points collide when they coincide).
        ax.scatter([], [], s=64, color=INK_2, edgecolors=SURFACE, linewidths=2,
                   label="at threshold " + ", ".join(f"{t:.2f}" for t in sorted(thresholds)))
    for p, label in ((cheap, "always cheap"), (strong, "always strong")):
        ax.scatter([p.cost_per_1k], [p.accuracy * 100], s=64, color=INK, edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.annotate(label, (p.cost_per_1k, p.accuracy * 100), textcoords="offset points", xytext=(8, 4), fontsize=8,
                    color=INK)
    ax.set_xlabel("Cost per 1,000 requests (USD, list prices)", color=INK_2)
    ax.set_ylabel("Accuracy on held-out questions (%)", color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    legend = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
