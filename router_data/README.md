# Router data

Written by the router lab scripts (see the main README, "Train and evaluate the router"):

- `questions.jsonl`: sampled GSM8K and MMLU questions with a fixed train/test split
  (`scripts/prepare_questions.py`).
- `answers.jsonl`: both Groq models' answers, graded, one line per call
  (`scripts/collect_labels.py`, append-only and resumable).
- `answers_openrouter.jsonl`: the same gpt-oss models through OpenRouter (pinned to the DeepInfra bf16 host), on
  all 1,200 questions, kept separate from the Groq file. Each record also has `provider`,
  `served_by` (the host), `cost_usd` (list-price estimate) and `provider_cost_usd` (the cost OpenRouter reported).
  Complete for both models (2026-11-15 to 2026-11-16: a 10-question pilot, the 377-question comparison set, then
  the remaining questions). The final collection run added 1,646 answers for $0.045385. The complete OpenRouter
  file cost $0.06608 at reported prices.

Keep both answer files committed: the answers cost quota, and the
evaluation and training are reproducible from them without any API calls.

Status: `questions.jsonl` contains 1,200 questions (900 train, 300 held out). OpenRouter has a successful answer from
both models for every question. Groq has 378 answers from gpt-oss-20b and 377 from gpt-oss-120b after its collection
run stopped at the 120b model's daily token limit. In the first 40 Groq records `latency_ms` includes local pacing
waits. Later records keep the model's own time in `latency_ms` and waits in `wait_ms`. Account ids are redacted from
stored error messages.
