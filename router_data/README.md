# Router data

Written by the router lab scripts (see the main README, "Train and evaluate the router"):

- `questions.jsonl`: sampled GSM8K and MMLU questions with a fixed train/test split
  (`scripts/prepare_questions.py`).
- `answers.jsonl`: both models' answers, graded, one line per call
  (`scripts/collect_labels.py`, append-only and resumable).

Commit both once collected: the answers cost days of free-tier quota, and the
evaluation and training are reproducible from them without any API calls.
