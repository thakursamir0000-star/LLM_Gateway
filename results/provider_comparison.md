# Provider comparison

Questions answered by every model: 377.

| Model | Provider (host) | Accuracy (95% CI) | Input tok | Output tok | Cost / 1k q (est.) | Cost / 1k q (reported) | Latency mean / p50 / p95 ms (n) | Calls | Failures | Rate limited |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b (`openai/gpt-oss-20b`) | groq | 83.6% (79%-87%) | 177 | 128 | $0.0517 | - | 592 / 568 / 905 (357) | 378 | 0 | 0 |
| gpt-oss-120b (`openai/gpt-oss-120b`) | groq | 87.3% (84%-90%) | 177 | 134 | $0.1071 | - | 716 / 689 / 1,118 (357) | 378 | 1 | 1 |
| or-gpt-oss-20b (`openai/gpt-oss-20b`) | openrouter (DeepInfra 377) | 84.4% (80%-88%) | 160 | 146 | $0.0253 | $0.0253 | 1,635 / 1,411 / 3,367 (377) | 379 | 2 | 2 |
| or-gpt-oss-120b (`openai/gpt-oss-120b`) | openrouter (DeepInfra 377) | 88.6% (85%-91%) | 160 | 140 | $0.0296 | $0.0296 | 6,137 / 4,342 / 15,476 (377) | 377 | 0 | 0 |

| Model | gsm8k accuracy (n) | mmlu accuracy (n) |
|---|---|---|
| gpt-oss-20b | 89.5% (133) | 80.3% (244) |
| gpt-oss-120b | 93.2% (133) | 84.0% (244) |
| or-gpt-oss-20b | 91.7% (133) | 80.3% (244) |
| or-gpt-oss-120b | 94.7% (133) | 85.2% (244) |

Question by question (first vs second model): right or wrong together, the same final answer, and the exact McNemar test of the accuracy difference.

| Pair | Both right | Both wrong | Only first right | Only second right | Same final answer | McNemar p |
|---|---|---|---|---|---|---|
| gpt-oss-20b vs gpt-oss-120b (same provider) | 298 | 31 | 17 | 31 | 324 (86%) | 0.059 |
| gpt-oss-20b vs or-gpt-oss-20b (same model, different provider) | 297 | 41 | 18 | 21 | 331 (88%) | 0.749 |
| gpt-oss-20b vs or-gpt-oss-120b (different model and provider) | 303 | 31 | 12 | 31 | 329 (87%) | 0.005 |
| gpt-oss-120b vs or-gpt-oss-20b (different model and provider) | 307 | 37 | 22 | 11 | 335 (89%) | 0.080 |
| gpt-oss-120b vs or-gpt-oss-120b (same model, different provider) | 321 | 35 | 8 | 13 | 353 (94%) | 0.383 |
| or-gpt-oss-20b vs or-gpt-oss-120b (same provider) | 308 | 33 | 10 | 26 | 335 (89%) | 0.011 |

Latency is the model's own time; answers saved before pacing waits were recorded separately are left out of it (n shows how many count). Calls, failures and rate limits cover every call the collection made for the model, after the registry's own retries.
