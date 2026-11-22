"""Read router shadow mode and a controlled classifier rollout from the request log.

    python scripts/router_report.py                 # last 7 days, from $GATEWAY_DATABASE_URL or $GATEWAY_DB
    python scripts/router_report.py --days 30 --json results/router_shadow.json

Shadow mode (routing.shadow: true): for requests the classifier did not route, it was still asked which tier it
would choose. The report gives how often it agrees with the tier used, how much traffic it would send to the cheap
tier, and an estimate of the saving: strong answers it would have sent cheap, re-priced at the cheap model's list
prices with the same token counts (an estimate: the cheap model's answer would have had its own length). It cannot
say whether the cheap answers would have been good enough; user feedback, where there is any, is shown per group.

Controlled activation (routing.classifier_share < 1): router-mode requests split between the classifier and a
control group answered by the strong tier. The report compares the two groups' error rate, cost, latency and
feedback. Read it before raising the share; small groups can't show small differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.config import database_target, load_config  # noqa: E402
from gateway.monitoring.store import Store, iso, utc_now  # noqa: E402


def rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def report(store: Store, config: Any, since: str) -> dict[str, Any]:
    rows = store.query(
        "SELECT r.id, r.mode, r.router, r.tier, r.release, r.latency_class, r.status, r.cost_usd, r.latency_ms, "
        "r.prompt_tokens, r.completion_tokens, r.shadow_tier, r.shadow_score, f.score AS feedback "
        "FROM requests r LEFT JOIN feedback f ON f.request_id = r.id WHERE r.ts >= :since", {"since": since})

    shadowed = [r for r in rows if r["shadow_tier"]]
    agree = sum(r["shadow_tier"] == r["tier"] for r in shadowed)
    would_cheap = [r for r in shadowed if r["shadow_tier"] == "cheap"]
    strong_to_cheap = [r for r in would_cheap if r["tier"] == "strong" and r["status"] == "ok"]
    saving = 0.0
    for r in strong_to_cheap:
        release = config.releases.get(r["release"] or "")
        if release is None:
            continue
        cheap = config.models[release.model_for("cheap", r["latency_class"] or "interactive")]
        saving += float(r["cost_usd"] or 0) - cheap.cost(r["prompt_tokens"] or 0, r["completion_tokens"] or 0)
    total_cost = sum(float(r["cost_usd"] or 0) for r in shadowed)

    def group(members: list[dict[str, Any]]) -> dict[str, Any]:
        rated = [float(r["feedback"]) for r in members if r["feedback"] is not None]
        ok = [r for r in members if r["status"] == "ok"]
        return {"requests": len(members), "error_rate": rate(sum(r["status"] == "error" for r in members), len(members)),
                "cheap_share": rate(sum(r["tier"] == "cheap" for r in members), len(members)),
                "cost_per_request_usd": mean([float(r["cost_usd"] or 0) for r in ok]),
                "latency_ms_mean": mean([float(r["latency_ms"]) for r in ok if r["latency_ms"] is not None]),
                "feedback": {"ratings": len(rated), "mean": mean(rated)}}

    routed = [r for r in rows if r["mode"] == "router"]
    return {
        "since": since,
        "requests": len(rows),
        "shadow": {
            "scored": len(shadowed),
            "agreement": rate(agree, len(shadowed)),
            "would_send_cheap": rate(len(would_cheap), len(shadowed)),
            "strong_answers_it_would_send_cheap": len(strong_to_cheap),
            "estimated_saving_usd": round(saving, 6),
            "estimated_saving_share": rate(saving, total_cost) if total_cost else None,
            "feedback_when_it_would_send_cheap": group(would_cheap)["feedback"],
            "feedback_when_it_would_send_strong": group([r for r in shadowed if r["shadow_tier"] == "strong"])["feedback"],
        },
        "activation": {
            "classifier_share": config.routing.classifier_share,
            "classifier": group([r for r in routed if r["router"] == "classifier"]),
            "control": group([r for r in routed if r["router"] == "control"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--config", help="gateway config (default: $GATEWAY_CONFIG or config/gateway.yaml)")
    parser.add_argument("--json", help="also write the report here")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    store = Store(database_target(config))
    try:
        result = report(store, config, iso(utc_now() - timedelta(days=args.days)))
    finally:
        store.close()
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
