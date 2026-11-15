# Router data

Written by the router lab scripts (see the main README, "Train and evaluate the router"):

- `questions.jsonl`: sampled GSM8K and MMLU questions with a fixed train/test split
  (`scripts/prepare_questions.py`).
- `answers.jsonl`: both Groq models' answers, graded, one line per call
  (`scripts/collect_labels.py`, append-only and resumable).
- `answers_openrouter.jsonl`: the same gpt-oss models through OpenRouter (pinned to the DeepInfra bf16 host), on
  questions both Groq models answered, kept separate from the Groq file. Each record also has `provider`,
  `served_by` (the host), `cost_usd` (list-price estimate) and `provider_cost_usd` (the cost OpenRouter reported).
  Complete for the 377 questions both Groq models answered (2026-11-15: a 10-question pilot, then the rest; total
  reported cost $0.0207).

Commit both once collected: the answers cost days of free-tier quota, and the
evaluation and training are reproducible from them without any API calls.

Status: `questions.jsonl` is prepared (1,200 questions: 900 train, 300 held out). `answers.jsonl` covers 377 of
them with both models (2026-11-15: a 20-question pilot, then a run that stopped at gpt-oss-120b's daily token limit;
Groq's limit is a rolling 24-hour window). Rerun `scripts/collect_labels.py --yes` to continue. In the first 40
records `latency_ms` includes local pacing waits; later records keep the model's own time in `latency_ms` and the
waits in `wait_ms`. Account ids are redacted from stored error messages.
