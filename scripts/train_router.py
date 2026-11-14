"""Train the classifier router on the collected answers (training split only).

    python scripts/train_router.py
    python scripts/train_router.py --embedder sentence-transformers/all-MiniLM-L6-v2 --out models/router.json

Label per question: cheap_ok = the cheap model was right, or the strong model
was wrong too. Features: a sentence embedding of the prompt as the gateway
sees it. Model: L2-regularised logistic regression, with the regularisation
picked by stratified 5-fold cross-validation (log loss). No API calls; the
embedder is downloaded from Hugging Face on first use.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from gateway.config import load_config
from gateway.routing.classifier import ClassifierRouter
from gateway.routing.embeddings import load_embedder
from gateway.routing.logistic import cross_validate, fit_logistic, predict_proba, roc_auc
from router_lab.dataset import build_pairs
from router_lab.progress import AnswerLog
from router_lab.questions import load_questions

L2_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MIN_PAIRS = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(ROOT / "router_data" / "questions.jsonl"))
    parser.add_argument("--answers", default=str(ROOT / "router_data" / "answers.jsonl"))
    parser.add_argument("--config", help="gateway config (default: config/gateway.yaml)")
    parser.add_argument("--cheap", help="cheap model alias (default: the active release's)")
    parser.add_argument("--strong", help="strong model alias (default: the active release's)")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="sentence-transformers model, or hashing:<dim> for a no-download baseline")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "models" / "router.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    release = config.releases[config.deployment.active]
    cheap, strong = args.cheap or release.cheap, args.strong or release.strong
    pairs = build_pairs(load_questions(args.questions), AnswerLog(args.answers).answers(), cheap, strong,
                        split="train")
    if len(pairs) < MIN_PAIRS:
        print(f"only {len(pairs)} training questions have answers from both {cheap} and {strong}; "
              f"collect at least {MIN_PAIRS} first (scripts/collect_labels.py)", file=sys.stderr)
        return 1
    y = np.array([p.cheap_ok for p in pairs], dtype=float)
    print(f"training questions: {len(pairs)} ({dict(Counter(p.question.source for p in pairs))}); "
          f"cheap good enough on {y.mean():.1%}")
    if y.min() == y.max():
        print("every training question has the same label; a classifier can't learn anything from that",
              file=sys.stderr)
        return 1

    print(f"embedding with {args.embedder}...")
    embedder = load_embedder(args.embedder)
    X = embedder.embed([p.text for p in pairs])

    cv = cross_validate(X, y, L2_GRID, k=args.folds)
    for row in cv:
        auc = "-" if row["auc"] is None else f"{row['auc']:.3f}"
        print(f"  l2={row['l2']:<7} log loss {row['log_loss']:.4f}  AUC {auc}")
    best = min(cv, key=lambda r: r["log_loss"])
    weights, bias = fit_logistic(X, y, best["l2"])
    train_auc = roc_auc(y, predict_proba(X, weights, bias))

    router = ClassifierRouter(weights, bias, embedder)
    router.save(args.out, cheap=cheap, strong=strong, label="cheap_ok = cheap correct or strong wrong",
                n_train=len(pairs), positive_rate=round(float(y.mean()), 4), l2=best["l2"],
                cv={"folds": args.folds, "results": cv}, train_auc=train_auc,
                sources=dict(Counter(p.question.source for p in pairs)))
    print(f"chose l2={best['l2']} (CV log loss {best['log_loss']:.4f}, AUC {best['auc']}); saved {args.out}")
    print("evaluate it on the held-out questions with scripts/evaluate_router.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
