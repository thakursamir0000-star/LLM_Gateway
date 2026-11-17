"""Evaluate routing strategies on the held-out questions (offline: no API calls).

    python scripts/evaluate_router.py --answers router_data/answers_openrouter.jsonl \
        --cheap or-gpt-oss-20b --strong or-gpt-oss-120b --baseline-answers router_data/answers.jsonl

Replays the collected answers of one cheap/strong pair for the test split:
always-cheap, always-strong, random routing, the rule-based router, the
trained classifier and the cascade.

No choice is made on the held-out questions. Each strategy's threshold is
chosen on the training split (the cheapest within --max-drop of always-strong
accuracy there): for the rules and the cascade from the training answers, for
the classifier by scripts/train_router.py from out-of-fold scores. The
held-out questions are then scored once at those thresholds, and every
strategy is tested against always-strong with an exact McNemar test.
--threshold uses one fixed threshold for all strategies instead.

Also reported: cost per 1,000 requests (list prices from the config), the
share sent to the strong model, model time mean/p50/p95, the router's added
latency, a descriptive threshold sweep on the held-out set (a curve, not a
claim), and, with --baseline-answers, every model in those files as a
fixed-model baseline on the held-out questions all of them answered. Writes
router_eval.md, router_eval.json and cost_vs_accuracy.png to --out.
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
from router_lab import safe_console
from router_lab.compare import common_questions, format_p, latest_answers, load_records, mcnemar_exact, model_stats
from router_lab.dataset import build_pairs
from router_lab.evaluate import (
    cascade_confidences,
    cascade_point,
    cascade_sweep,
    cheapest_within,
    choose_threshold,
    items_from_pairs,
    markdown_table,
    plot_cost_vs_accuracy,
    random_point,
    route_correct,
    route_point,
    router_latency_ms,
    sweep,
    to_dict,
)
from router_lab.progress import AnswerLog
from router_lab.questions import load_questions

MIN_ITEMS = 10
GRID = [round(t, 3) for t in np.linspace(0, 1, 201)]


def main() -> int:
    safe_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(ROOT / "router_data" / "questions.jsonl"))
    parser.add_argument("--answers", default=str(ROOT / "router_data" / "answers.jsonl"))
    parser.add_argument("--baseline-answers", nargs="*", default=[],
                        help="more answer files; every model in them is shown as a fixed-model baseline")
    parser.add_argument("--router", default=str(ROOT / "models" / "router.json"))
    parser.add_argument("--config", help="gateway config (prices; default: config/gateway.yaml)")
    parser.add_argument("--cheap", help="cheap model alias (default: the router file's, else the active release's)")
    parser.add_argument("--strong", help="strong model alias (default: the router file's, else the active release's)")
    parser.add_argument("--threshold", type=float, help="one fixed threshold for every strategy (default: chosen "
                                                        "on the training split for each strategy)")
    parser.add_argument("--max-drop", type=float, default=0.01,
                        help="accuracy you'd give up vs always-strong when choosing thresholds on training data")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    config = load_config(args.config)
    release = config.releases[config.deployment.active]
    router_path = Path(args.router)
    classifier = ClassifierRouter.load(router_path) if router_path.exists() else None
    meta = classifier.meta if classifier else {}
    cheap = args.cheap or meta.get("cheap") or release.cheap
    strong = args.strong or meta.get("strong") or release.strong

    questions = load_questions(args.questions)
    answers = AnswerLog(args.answers).answers()
    pairs = build_pairs(questions, answers, cheap, strong, split="test")
    train_pairs = build_pairs(questions, answers, cheap, strong, split="train")
    if len(pairs) < MIN_ITEMS:
        print(f"only {len(pairs)} held-out questions have answers from both {cheap} and {strong}; "
              f"collect more first (scripts/collect_labels.py)", file=sys.stderr)
        return 1
    items = items_from_pairs(pairs, config, cheap, strong)
    train_items = items_from_pairs(train_pairs, config, cheap, strong)
    texts = [i.text for i in items]
    rules = RulesRouter()

    def rules_score(text: str) -> float:
        return rules.score([{"role": "user", "content": text}]).score

    scores = {"rules": np.array([rules_score(t) for t in texts])}
    train_scores = {"rules": np.array([rules_score(i.text) for i in train_items])}
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

    # ── thresholds, chosen without the held-out questions ──
    chosen: dict[str, tuple[float, str]] = {}
    if args.threshold is not None:
        chosen = {name: (args.threshold, "fixed (--threshold)") for name in [*scores, "cascade"]}
    else:
        if not train_items:
            print("no training questions with both answers: pass --threshold", file=sys.stderr)
            return 1
        strong_train = route_point(train_items, [True] * len(train_items), "always-strong", None)
        for name in ("rules",):
            t = choose_threshold(sweep(train_items, train_scores[name], name, GRID), strong_train.accuracy,
                                 args.max_drop)
            chosen[name] = (t, "chosen on the training answers")
        t = choose_threshold(cascade_sweep(train_items, GRID), strong_train.accuracy, args.max_drop)
        chosen["cascade"] = (t, "chosen on the training answers")
        if classifier is not None:
            if meta.get("threshold") is not None:
                chosen["classifier"] = (float(meta["threshold"]), "chosen by train_router.py (out-of-fold)")
            else:
                chosen["classifier"] = (config.routing.threshold, "config default (router file has none)")

    always_cheap = route_point(items, [False] * len(items), "always-cheap", None)
    always_strong = route_point(items, [True] * len(items), "always-strong", None)
    strong_correct = route_correct(items, [True] * len(items))
    confidence = cascade_confidences(items)
    operating, decisions = {}, {}
    for name, s in scores.items():
        t = chosen[name][0]
        decisions[name] = s < t
        operating[name] = route_point(items, decisions[name], name, t)
    decisions["cascade"] = confidence < chosen["cascade"][0]
    operating["cascade"] = cascade_point(items, decisions["cascade"], chosen["cascade"][0])
    reference = operating.get("classifier", operating["rules"])
    random_match = random_point(items, reference.strong_share)

    def vs_strong(name: str) -> dict[str, float]:
        mine = route_correct(items, decisions[name])
        only_mine = int(np.sum(mine & ~strong_correct))
        only_strong = int(np.sum(strong_correct & ~mine))
        saving = 1 - operating[name].cost_usd / always_strong.cost_usd if always_strong.cost_usd else 0.0
        return {"accuracy_diff": operating[name].accuracy - always_strong.accuracy, "only_router_right": only_mine,
                "only_strong_right": only_strong, "mcnemar_p": mcnemar_exact(only_mine, only_strong),
                "cost_saving": saving}

    tests = {name: vs_strong(name) for name in operating}

    # ── descriptive sweep on the held-out set ──
    curves = {name: sweep(items, s, name, GRID) for name, s in scores.items()}
    curves["cascade"] = cascade_sweep(items, GRID)
    best = {name: cheapest_within(points, always_strong.accuracy - args.max_drop) for name, points in curves.items()}

    # ── fixed-model baselines (every model in the answer files) ──
    baselines, baseline_ids = [], set()
    if args.baseline_answers:
        records = load_records([args.answers, *args.baseline_answers])
        every = latest_answers(records)
        models = sorted({m for (_, m) in every})
        test_ids = {q.id for q in questions if q.split == "test"}
        baseline_ids = common_questions(every, models) & test_ids
        if baseline_ids:
            baselines = [model_stats(m, every, records, baseline_ids, config) for m in models]

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
        f"Cheap model `{cheap}`, strong model `{strong}`. Training questions: {len(train_items)}. "
        "Costs use the list prices in the config.",
        f"Router: {classifier.description if classifier else 'none trained'}.",
        "",
        "## Held-out results at thresholds chosen on the training split",
        "",
        "Thresholds: " + "; ".join(f"{name} {t:.3f} ({how})" for name, (t, how) in chosen.items()) + ". "
        f"Random routing sends the same share to the strong model as the "
        f"{'classifier' if 'classifier' in operating else 'rules'} ({reference.strong_share:.0%}).",
        "",
        markdown_table(rows),
        "",
        "## Against always-strong (held-out, exact McNemar test)",
        "",
        "| Strategy | Accuracy difference | Only it right | Only always-strong right | p | Cost saving |",
        "|---|---|---|---|---|---|",
    ]
    for name, t in tests.items():
        lines.append(f"| {name} | {t['accuracy_diff'] * 100:+.1f} pts | {t['only_router_right']} | "
                     f"{t['only_strong_right']} | {format_p(t['mcnemar_p'])} | {t['cost_saving']:.0%} |")
    lines += ["", "A strategy improves on always-strong only if it costs less at an accuracy difference the test "
                  "can't tell from zero, or is more accurate with p < 0.05."]
    if baselines:
        lines += ["", f"## Fixed-model baselines (held-out questions every model answered: {len(baseline_ids)})",
                  "", "| Model | Provider | Accuracy (95% CI) | Cost per 1k (est.) | Latency p50 / p95 ms |",
                  "|---|---|---|---|---|"]
        for s in baselines:
            p50 = "-" if s.latency_p50_ms is None else f"{s.latency_p50_ms:,.0f}"
            p95 = "-" if s.latency_p95_ms is None else f"{s.latency_p95_ms:,.0f}"
            lines.append(f"| {s.model} | {s.provider} | {s.accuracy:.1%} ({s.ci_low:.0%}-{s.ci_high:.0%}) | "
                         f"${s.cost_per_1k_estimated:.4f} | {p50} / {p95} |")
        subset = [i for i in items if i.id in baseline_ids]
        if subset:
            sub_rows = [route_point(subset, [False] * len(subset), "always-cheap", None),
                        route_point(subset, [True] * len(subset), "always-strong", None)]
            index = {i.id: k for k, i in enumerate(items)}
            for name in operating:
                picks = [bool(decisions[name][index[i.id]]) for i in subset]
                sub_rows.append(cascade_point(subset, picks, chosen[name][0]) if name == "cascade"
                                else route_point(subset, picks, name, chosen[name][0]))
            lines += ["", f"The pair's strategies on the same {len(subset)} questions:", "", markdown_table(sub_rows)]
    lines += ["", f"## Descriptive: threshold sweep on the held-out set (cheapest within "
                  f"{args.max_drop * 100:g} percentage point(s) of always-strong, {always_strong.accuracy:.1%})", "",
              "Chosen by looking at the held-out answers, so this is a curve to read, not a result to claim.", "",
              "| Router | Threshold | Accuracy | Cost per 1k requests | Saving vs always-strong |",
              "|---|---|---|---|---|"]
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
        subset_pairs = [p for p in pairs if p.question.source == source]
        n = len(subset_pairs)
        lines.append(f"| {source} | {n} | {sum(p.cheap['correct'] for p in subset_pairs) / n:.1%} | "
                     f"{sum(p.strong['correct'] for p in subset_pairs) / n:.1%} | "
                     f"{sum(p.cheap_ok for p in subset_pairs) / n:.1%} |")
    lines += ["", "![Cost vs accuracy](cost_vs_accuracy.png)", ""]
    report = "\n".join(lines)
    (out / "router_eval.md").write_text(report, encoding="utf-8")
    (out / "router_eval.json").write_text(json.dumps({
        "cheap": cheap, "strong": strong, "n": len(items), "n_train": len(train_items), "sources": dict(sources),
        "router": meta | {"path": str(router_path)} if classifier else None,
        "prices_per_1m": {m: config.models[m].price_per_1m.model_dump() for m in (cheap, strong)},
        "thresholds": {name: {"threshold": t, "how": how} for name, (t, how) in chosen.items()},
        "operating": to_dict(rows), "vs_always_strong": tests,
        "baselines": [b.__dict__ for b in baselines], "baseline_questions": len(baseline_ids),
        "curves": {k: to_dict(v) for k, v in curves.items()},
        "cheapest_within_drop_on_test": {k: (None if v is None else to_dict([v])[0]) for k, v in best.items()},
        "max_drop": args.max_drop, "router_latency": latency,
    }, indent=1, default=float), encoding="utf-8")
    print(report)
    print(f"wrote {out / 'router_eval.md'}, {out / 'router_eval.json'} and {chart}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
