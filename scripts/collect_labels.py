"""Run the cheap and the strong model on the prepared questions and record which answers are right.

    python scripts/collect_labels.py --dry-run         # the plan and a sample prompt; no API calls
    python scripts/collect_labels.py --limit 20        # a small pilot (asks before spending quota)
    python scripts/collect_labels.py --yes             # everything; rerun the same command to resume

THIS SPENDS REAL API QUOTA (the keys in .env). The models are the active
release's cheap and strong ones (config/gateway.yaml) unless --cheap/--strong
name others. Calls are paced under each model's per-minute limits from the
config and retried within bounds; every answer is appended to
router_data/answers.jsonl the moment it arrives, so nothing finished is ever
asked again. The run stops at a daily quota, a missing/rejected key, or five
errors in a row: rerun it after the quota resets.

On Groq's free tier openai/gpt-oss-120b allows 8,000 tokens per minute and
200,000 per day, so a full run takes several days of daily runs.

Another provider's models go in their own answer file, on the same questions:

    python scripts/collect_labels.py --cheap or-gpt-oss-20b --strong or-gpt-oss-120b         --answers router_data/answers_openrouter.jsonl --only-answered-in router_data/answers.jsonl         --sample 10 --env-file "D:/New folder/Agentai/.env" --yes

Paid providers can run under a spending cap. --max-cost-usd never starts a
request that could take spending past the cap (worst case: the whole output
token cap), and stops on unexpected billing. --budget-tag ties runs together:
a resumed run with the same tag counts what the earlier runs spent.

    python scripts/collect_labels.py ... --max-cost-usd 0.05 --budget-tag or-remaining --yes
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from gateway.config import load_config
from gateway.providers import ModelRegistry, build_providers
from router_lab import safe_console
from router_lab.budget import Budget, record_cost, spent_with_tag, worst_case_cost
from router_lab.collect import collect
from router_lab.progress import AnswerLog, AnswerLogLocked
from router_lab.questions import format_prompt, load_questions, messages_for, sample


def main() -> int:
    safe_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(ROOT / "router_data" / "questions.jsonl"))
    parser.add_argument("--answers", default=str(ROOT / "router_data" / "answers.jsonl"))
    parser.add_argument("--config", help="gateway config (default: config/gateway.yaml)")
    parser.add_argument("--cheap", help="cheap model alias (default: the active release's)")
    parser.add_argument("--strong", help="strong model alias (default: the active release's)")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--only-answered-in", metavar="ANSWERS",
                        help="only questions that every model in this answer file answered (same-question comparisons)")
    parser.add_argument("--sample", type=int, help="a seeded random N of the selected questions")
    parser.add_argument("--seed", type=int, default=0, help="seed for --sample")
    parser.add_argument("--env-file", help="also read API keys from this .env file (never printed)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="output token cap per answer")
    parser.add_argument("--max-cost-usd", type=float,
                        help="spending cap for this budget (reported cost, else list-price estimate)")
    parser.add_argument("--budget-tag",
                        help="name of the budget; earlier runs with the same tag count toward --max-cost-usd")
    parser.add_argument("--dry-run", action="store_true", help="show what would run; no API calls")
    parser.add_argument("--yes", action="store_true", help="don't ask before spending API quota")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.env_file:
        if not Path(args.env_file).exists():
            print(f"--env-file not found: {args.env_file}", file=sys.stderr)
            return 1
        load_dotenv(args.env_file)  # adds missing variables; never overrides ones already set
    config = load_config(args.config)
    release = config.releases[config.deployment.active]
    models = [args.cheap or release.cheap, args.strong or release.strong]
    for alias in models:
        if alias not in config.models:
            print(f"unknown model alias {alias!r} (known: {', '.join(config.models)})", file=sys.stderr)
            return 1
    questions = load_questions(args.questions)
    if args.split != "all":
        questions = [q for q in questions if q.split == args.split]
    if args.only_answered_in:
        keep = AnswerLog(args.only_answered_in).answered_by_all()
        questions = [q for q in questions if q.id in keep]
    if args.sample:
        questions = sample(questions, args.sample, args.seed)
    questions = questions[: args.limit] if args.limit else questions
    log = AnswerLog(args.answers)
    done = log.answers()
    todo = sum(1 for q in questions for m in models if (q.id, m) not in done)

    print(f"questions: {len(questions)} ({dict(Counter(q.source for q in questions))})")
    for alias in models:
        spec = config.models[alias]
        key = "set" if os.environ.get(config.providers[spec.provider].api_key_env) else "NOT SET"
        print(f"model {alias}: {spec.provider} {spec.model}, limits rpm={spec.limits.rpm} tpm={spec.limits.tpm}, "
              f"params {spec.params}, API key {key}")
    print(f"answers file: {args.answers} ({len(done)} answers saved, {todo} calls to go)")

    # Expected cost: each model's mean cost per question source so far; worst case: the guard's per-call bound.
    params = {"max_completion_tokens": args.max_tokens}
    costs: dict[tuple[str, str], list[float]] = {}
    for (_, alias), answer in done.items():
        costs.setdefault((alias, answer.get("source", "?")), []).append(record_cost(answer))
    expected = worst = 0.0
    for q in questions:
        for alias in models:
            if (q.id, alias) not in done:
                bound = worst_case_cost(config.models[alias], messages_for(q), params)
                seen = costs.get((alias, q.source))
                worst += bound
                expected += sum(seen) / len(seen) if seen else bound
    print(f"estimated cost of the remaining calls: ${expected:.4f} expected (from the answers so far), "
          f"${worst:.4f} if every call used its whole {args.max_tokens}-token output cap")
    budget = None
    if args.max_cost_usd is not None:
        budget = Budget(args.max_cost_usd, spent_with_tag(log.records(), args.budget_tag), args.budget_tag)
        tag = f"tag {args.budget_tag!r}" if args.budget_tag else "no tag: only this run counts"
        print(f"budget: ${budget.limit_usd:.4f}, already spent ${budget.spent_usd:.6f} ({tag}), "
              f"${budget.remaining_usd:.6f} left")
    if args.dry_run:
        if questions:
            print("\nsample prompt:\n" + format_prompt(questions[0]))
        return 0
    if todo == 0:
        print("nothing to do")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("refusing to spend API quota without --yes (not an interactive terminal)", file=sys.stderr)
            return 1
        if input(f"Make up to {todo} real API calls? [y/N] ").strip().lower() != "y":
            return 1

    async def run():
        registry = ModelRegistry(config, build_providers(config), attempts=6, max_wait=180)
        try:
            return await collect(questions, models, registry, log, params=params, budget=budget)
        finally:
            await registry.aclose()

    try:
        summary = asyncio.run(run())
    except AnswerLogLocked as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    answers = log.answers()
    print(f"\nanswered {summary.answered}, already done {summary.skipped}, errors {summary.errors}, "
          f"spent this run ${summary.spent_usd:.6f}")
    if budget is not None:
        print(f"budget: ${budget.spent_usd:.6f} of ${budget.limit_usd:.4f} spent, ${budget.remaining_usd:.6f} left")
    for alias in models:
        mine = [a for (qid, m), a in answers.items() if m == alias]
        if mine:
            right = sum(1 for a in mine if a["correct"])
            print(f"  {alias}: {len(mine)} answered so far, {right / len(mine):.1%} right")
    if summary.stopped:
        print(f"\nstopped early: {summary.stopped}. Rerun the same command later to continue.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
