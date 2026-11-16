# Dataset quality report

Questions: 1200 (test/gsm8k 100, test/mmlu 200, train/gsm8k 300, train/mmlu 600). Problems in the question set: 0.

| File | Model | Answered | Missing | Errors (kept, retried later) | Duplicates | Unknown ids | Incomplete | Gold mismatch | No extracted answer | Truncated | Re-grade mismatch | Accuracy | Hosts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answers.jsonl | gpt-oss-120b | 377 | 823 | quota_exhausted 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 87.3% | - |
| answers.jsonl | gpt-oss-20b | 378 | 822 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 83.6% | - |
| answers_openrouter.jsonl | or-gpt-oss-120b | 1200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 88.8% | DeepInfra 1200 |
| answers_openrouter.jsonl | or-gpt-oss-20b | 1200 | 0 | rate_limit 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 84.9% | DeepInfra 1200 |

Questions answered by every model: 377 (test 86, train 291).

| Model | Cost (reported, else list-price estimate) | Answers without a reported cost |
|---|---|---|
| gpt-oss-120b | $0.04036 | 377 |
| gpt-oss-20b | $0.01963 | 378 |
| or-gpt-oss-120b | $0.03601 | 0 |
| or-gpt-oss-20b | $0.03007 | 0 |

## Examples
- or-gpt-oss-120b, truncated: gsm8k-0205
- or-gpt-oss-20b, no extracted answer: mmlu-12314: ...'iagnosis should not be disclosed to any of the listed parties.  \n\nAnswer: None\n\n'; mmlu-00457: ...'I’m sorry, but I don’t have enough information to determine the correct answer.'

Integrity problems (duplicates, unknown ids, incomplete records, gold/split mismatches, re-grade mismatches): 0. Answers with no extracted final answer and truncated answers are model failures and stay in the data as wrong answers.
