"""Compare models across providers on the questions they all answered (offline: no API calls).

    python scripts/compare_providers.py
    python scripts/compare_providers.py --models gpt-oss-20b,or-gpt-oss-20b --out results/provider_comparison.md

Reads router_data/answers.jsonl (Groq) and router_data/answers_openrouter.jsonl
by default; each provider's answers stay in their own file. For every pair of
aliases of the same provider model (e.g. Groq and OpenRouter gpt-oss-20b) it
also counts the questions where only one of them was right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.config import load_config
from router_lab import safe_console
from router_lab.compare import (
    agreement,
    common_questions,
    latest_answers,
    load_records,
    markdown,
    model_stats,
    to_json,
)

DEFAULT_ANSWERS = [ROOT / "router_data" / "answers.jsonl", ROOT / "router_data" / "answers_openrouter.jsonl"]


def main() -> int:
    safe_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", nargs="+", default=[str(p) for p in DEFAULT_ANSWERS], help="answer files")
    parser.add_argument("--models", help="comma-separated aliases (default: every model in the files)")
    parser.add_argument("--config", help="gateway config, for list prices (default: config/gateway.yaml)")
    parser.add_argument("--out", help="also write the report here (.md) and the numbers next to it (.json)")
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_records(args.answers)
    answers = latest_answers(records)
    models = args.models.split(",") if args.models else sorted({m for (_, m) in answers})
    if len(models) < 2:
        print(f"need answers from at least two models; found {models}", file=sys.stderr)
        return 1
    question_ids = common_questions(answers, models)
    if not question_ids:
        print(f"no question was answered by all of: {', '.join(models)}", file=sys.stderr)
        return 1
    stats = [model_stats(m, answers, records, question_ids, config) for m in models]
    pairs = {}
    for i, a in enumerate(stats):
        for b in stats[i + 1:]:
            if a.provider_model == b.provider_model and a.provider != b.provider:
                pairs[f"{a.model} vs {b.model} (same model, different provider)"] = agreement(
                    answers, a.model, b.model, question_ids)
    report = markdown(stats, pairs)
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# Provider comparison\n\n" + report + "\n", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(to_json(stats, pairs), indent=1), encoding="utf-8")
        print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
