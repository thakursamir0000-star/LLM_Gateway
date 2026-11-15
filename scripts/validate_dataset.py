"""Check the question set and the answer files, and write a dataset quality report (offline).

    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --out results/dataset_quality.md

Exits with 1 when it finds integrity problems (duplicate answers, unknown
question ids, incomplete records, gold or split mismatches, stored grades that
re-grading doesn't reproduce), so it can gate training.
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
from router_lab.questions import load_questions
from router_lab.validate import integrity_problems, markdown, to_json, validate

DEFAULT_ANSWERS = [ROOT / "router_data" / "answers.jsonl", ROOT / "router_data" / "answers_openrouter.jsonl"]


def main() -> int:
    safe_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(ROOT / "router_data" / "questions.jsonl"))
    parser.add_argument("--answers", nargs="+", default=[str(p) for p in DEFAULT_ANSWERS])
    parser.add_argument("--config", help="gateway config, for list prices (default: config/gateway.yaml)")
    parser.add_argument("--out", help="write the report here (.md) and the numbers next to it (.json)")
    args = parser.parse_args()

    result = validate(load_questions(args.questions), [p for p in args.answers if Path(p).exists()],
                      load_config(args.config))
    report = markdown(result)
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(to_json(result), indent=1), encoding="utf-8")
        print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 1 if integrity_problems(result) else 0


if __name__ == "__main__":
    sys.exit(main())
