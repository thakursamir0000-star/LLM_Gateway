# Provider comparison

Questions answered by every model: 1200.

| Model | Provider (host) | Accuracy (95% CI) | Input tok | Output tok | Cost / 1k q (est.) | Cost / 1k q (reported) | Latency mean / p50 / p95 ms (n) | Calls | Failures | Rate limited |
|---|---|---|---|---|---|---|---|---|---|---|
| or-gpt-oss-120b (`openai/gpt-oss-120b`) | openrouter (DeepInfra 1200) | 88.8% (87%-90%) | 163 | 141 | $0.0300 | $0.0300 | 5,595 / 4,244 / 13,400 (1200) | 1200 | 0 | 0 |
| or-gpt-oss-20b (`openai/gpt-oss-20b`) | openrouter (DeepInfra 1200) | 84.9% (83%-87%) | 163 | 144 | $0.0251 | $0.0251 | 2,183 / 1,557 / 5,445 (1200) | 1202 | 2 | 2 |

| Model | gsm8k accuracy (n) | mmlu accuracy (n) |
|---|---|---|
| or-gpt-oss-120b | 96.2% (400) | 85.1% (800) |
| or-gpt-oss-20b | 93.8% (400) | 80.5% (800) |

Question by question (first vs second model): right or wrong together, the same final answer, and the exact McNemar test of the accuracy difference.

| Pair | Both right | Both wrong | Only first right | Only second right | Same final answer | McNemar p |
|---|---|---|---|---|---|---|
| or-gpt-oss-120b vs or-gpt-oss-20b (same provider) | 987 | 102 | 79 | 32 | 1062 (88%) | <0.001 |

Latency is the model's own time; answers saved before pacing waits were recorded separately are left out of it (n shows how many count). Calls, failures and rate limits cover every call the collection made for the model, after the registry's own retries.
