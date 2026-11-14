# Router data

Written by the router lab scripts (see the main README, "Train and evaluate the router"):

- `questions.jsonl`: sampled GSM8K and MMLU questions with a fixed train/test split
  (`scripts/prepare_questions.py`).
- `answers.jsonl`: both models' answers, graded, one line per call
  (`scripts/collect_labels.py`, append-only and resumable).

Commit both once collected: the answers cost days of free-tier quota, and the
evaluation and training are reproducible from them without any API calls.

Status: `questions.jsonl` is prepared (1,200 questions: 900 train, 300 held out). `answers.jsonl` holds a 20-question
pilot (2026-11-15, Groq, both models), which checked the pipeline and the grader on real answers. In those first 40
records `latency_ms` includes local pacing waits; later records keep the model's own time in `latency_ms` and the
waits in `wait_ms`.
