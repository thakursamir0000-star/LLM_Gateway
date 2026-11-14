"""Sample GSM8K and MMLU questions and fix the router's train/test split.

    python scripts/prepare_questions.py                  # 400 GSM8K + 800 MMLU, 25% held out
    python scripts/prepare_questions.py --gsm8k 100 --mmlu 200

Downloads the public test sets with Hugging Face `datasets`
(pip install -r requirements-lab.txt). No API key and no LLM calls.
Writes router_data/questions.jsonl. The split is decided here, before any
model answers a question, and the script refuses to overwrite an existing
file (that would move questions between train and test after answers were
collected); pass --force to start over.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from router_lab.questions import from_gsm8k, from_mmlu, sample_and_split, save_questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gsm8k", type=int, default=400, help="GSM8K questions (maths word problems)")
    parser.add_argument("--mmlu", type=int, default=800, help="MMLU questions (57 subjects, multiple choice)")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="share held out for the evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "router_data" / "questions.jsonl"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing questions file")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists; keeping it so the train/test split doesn't change. Use --force to redo it.")
        return 1
    try:
        from datasets import load_dataset
    except ImportError:
        print("needs Hugging Face datasets: pip install -r requirements-lab.txt", file=sys.stderr)
        return 1

    print("loading openai/gsm8k (main, test) and cais/mmlu (all, test)...")
    gsm8k = from_gsm8k(list(load_dataset("openai/gsm8k", "main", split="test")))
    mmlu = from_mmlu(list(load_dataset("cais/mmlu", "all", split="test")))
    questions = sample_and_split({"gsm8k": gsm8k, "mmlu": mmlu}, {"gsm8k": args.gsm8k, "mmlu": args.mmlu},
                                 args.test_fraction, args.seed)
    save_questions(out, questions)
    counts = Counter((q.split, q.source) for q in questions)
    print(f"wrote {len(questions)} questions to {out}")
    for split in ("train", "test"):
        print(f"  {split}: " + ", ".join(f"{source} {counts[(split, source)]}" for source in ("gsm8k", "mmlu")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
