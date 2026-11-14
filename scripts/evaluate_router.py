"""Evaluate routing strategies on the held-out questions (offline: no API calls).

    python scripts/evaluate_router.py
    python scripts/evaluate_router.py --threshold 0.6 --max-drop 0.02

Replays the collected answers (router_data/answers.jsonl) for the test split:
always-cheap, always-strong, random routing, the rule-based router, the
trained classifier and the cascade, at the configured threshold and swept over
thresholds. Reports accuracy, cost (list prices from the config), the share
sent to the strong model, the router's added latency, and the cheapest
threshold within --max-drop of always-strong accuracy. Writes
results/router_eval.md, results/router_eval.json and results/cost_vs_accuracy.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from gateway.config import load_config
from gateway.routing import RulesRouter
from gateway.routing.cascade import check_answer
from gateway.routing.classifier import ClassifierRouter
from router_lab.dataset import build_pairs
from router_lab.evaluate import (
    cascade_confidences,
    cascade_point,
    cascade_sweep,
    cheapest_within,
    items_from_pairs,
    markdown_table,
    plot_cost_vs_accuracy,
    random_point,
    route_point,
    router_latency_ms,
    sweep,
    to_dict,
)
from router_lab.progress import AnswerLog
from router_lab.questions import load_questions

MIN_ITEMS = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(ROOT / "router_data" / "questions.jsonl"))
    parser.add_argument("--answers", default=str(ROOT / "router_data" / "answers.jsonl"))
    parser.add_argument("--router", default=str(ROOT / "models" / "router.json"))
    parser.add_argument("--config", help="gateway config (prices, threshold; default: config/gateway.yaml)")
    parser.add_argument("--cheap", help="cheap model alias (default: the router file's, else the active release's)")
    parser.add_argument("--strong", help="strong model alias (default: the router file's, else the active release's)")
    parser.add_argument("--threshold", type=float, help="operating threshold (default: routing.threshold)")
    parser.add_argument("--max-drop", type=float, default=0.01,
                        help="accuracy you'd give up vs always-strong when searching for the cheapest threshold")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    config = load_config(args.config)
    release = config.releases[config.deployment.active]
    router_path = Path(args.router)
    classifier = ClassifierRouter.load(router_path) if router_path.exists() else None
    meta = classifier.meta if classifier else {}
    cheap = args.cheap or meta.get("cheap") or release.cheap
    strong = args.strong or meta.get("strong") or release.strong
    threshold = config.routing.threshold if args.threshold is None else args.threshold

    pairs = build_pairs(load_questions(args.questions), AnswerLog(args.answers).answers(), cheap, strong,
                        split="test")
    if len(pairs) < MIN_ITEMS:
        print(f"only {len(pairs)} held-out questions have answers from both {cheap} and {strong}; "
              f"collect more first (scripts/collect_labels.py)", file=sys.stderr)
        return 1
    items = items_from_pairs(pairs, config, cheap, strong)
    texts = [i.text for i in items]
    thresholds = [round(t, 3) for t in np.linspace(0, 1, 41)]
    rules = RulesRouter()

    def rules_score(text: str) -> float:
        return rules.score([{"role": "user", "content": text}]).score

    scores = {"rules": np.array([rules_score(t) for t in texts])}
    latency = {"rules": router_latency_ms(rules_score, texts)}
    if classifier is None:
        print(f"no trained router at {router_path}: evaluating the rules and the cascade only")
    else:
        scores["classifier"] = classifier.score_texts(texts)
        latency["classifier"] = router_latency_ms(lambda text: classifier.score_texts([text]), texts)
    by_id = {i.id: i for i in items}
    latency["cascade check"] = router_latency_ms(
        lambda text_id: check_answer(by_id[text_id].text, by_id[text_id].cheap_text, by_id[text_id].cheap_finish),
        [i.id for i in items])

    always_cheap = route_point(items, [False] * len(items), "always-cheap", None)
    always_strong = route_point(items, [True] * len(items), "always-strong", None)
    curves = {name: sweep(items, s, name, thresholds) for name, s in scores.items()}
    curves["cascade"] = cascade_sweep(items, thresholds)
    operating = {name: route_point(items, s < threshold, name, threshold) for name, s in scores.items()}
    operating["cascade"] = cascade_point(items, cascade_confidences(items) < threshold, threshold)
    reference = operating.get("classifier", operating["rules"])
    random_match = random_point(items, reference.strong_share)
    target = always_strong.accuracy - args.max_drop
    best = {name: cheapest_within(points, target) for name, points in curves.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chart = out / "cost_vs_accuracy.png"
    plot_cost_vs_accuracy(curves, operating, always_cheap, always_strong, str(chart),
                          f"Routing {cheap} vs {strong}: {len(items)} held-out questions")

    sources = Counter(p.question.source for p in pairs)
    rows = [always_cheap, always_strong, random_match, *operating.values()]
    lines = [
        "# Router evaluation",
        "",
        f"Held-out questions: {len(items)} ({', '.join(f'{k} {v}' for k, v in sorted(sources.items()))}). "
        f"Cheap model `{cheap}`, strong model `{strong}`. Costs use the list prices in the config.",
        f"Router: {classifier.description if classifier else 'none trained'}.",
        "",
        f"## At threshold {threshold:.2f}",
        "",
        f"Random routing sends the same share to the strong model as the "
        f"{'classifier' if 'classifier' in operating else 'rules'} ({reference.strong_share:.0%}).",
        "",
        markdown_table(rows),
        "",
        f"## Cheapest threshold within {args.max_drop * 100:g} percentage point(s) of always-strong accuracy "
        f"({always_strong.accuracy:.1%})",
        "",
        "| Router | Threshold | Accuracy | Cost per 1k requests | Saving vs always-strong |",
        "|---|---|---|---|---|",
    ]
    for name, point in best.items():
        if point is None:
            lines.append(f"| {name} | not reached | - | - | - |")
        else:
            saving = 1 - point.cost_usd / always_strong.cost_usd if always_strong.cost_usd else 0.0
            lines.append(f"| {name} | {point.threshold:.3f} | {point.accuracy:.1%} | ${point.cost_per_1k:.4f} | "
                         f"{saving:.0%} |")
    lines += ["", "## Added latency per request (this machine, CPU)", "",
              "| Step | p50 | p95 |", "|---|---|---|"]
    for name, stats in latency.items():
        lines.append(f"| {name} | {stats['p50_ms']:.2f} ms | {stats['p95_ms']:.2f} ms |")
    lines += ["", "## By source", "", "| Source | Questions | Cheap right | Strong right | Cheap good enough |",
              "|---|---|---|---|---|"]
    for source in sorted(sources):
        subset = [p for p in pairs if p.question.source == source]
        n = len(subset)
        lines.append(f"| {source} | {n} | {sum(p.cheap['correct'] for p in subset) / n:.1%} | "
                     f"{sum(p.strong['correct'] for p in subset) / n:.1%} | "
                     f"{sum(p.cheap_ok for p in subset) / n:.1%} |")
    lines += ["", "![Cost vs accuracy](cost_vs_accuracy.png)", ""]
    report = "\n".join(lines)
    (out / "router_eval.md").write_text(report, encoding="utf-8")
    (out / "router_eval.json").write_text(json.dumps({
        "cheap": cheap, "strong": strong, "threshold": threshold, "n": len(items), "sources": dict(sources),
        "router": meta | {"path": str(router_path)} if classifier else None,
        "prices_per_1m": {m: config.models[m].price_per_1m.model_dump() for m in (cheap, strong)},
        "operating": to_dict(rows), "curves": {k: to_dict(v) for k, v in curves.items()},
        "cheapest_within_drop": {k: (None if v is None else to_dict([v])[0]) for k, v in best.items()},
        "max_drop": args.max_drop, "router_latency": latency,
    }, indent=1), encoding="utf-8")
    print(report)
    print(f"wrote {out / 'router_eval.md'}, {out / 'router_eval.json'} and {chart}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
