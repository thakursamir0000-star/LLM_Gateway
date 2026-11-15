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

gpt-oss-20b vs or-gpt-oss-20b (same model, different provider): both_right 297, both_wrong 41, only_gpt-oss-20b 18, only_or-gpt-oss-20b 21

gpt-oss-120b vs or-gpt-oss-120b (same model, different provider): both_right 321, both_wrong 35, only_gpt-oss-120b 8, only_or-gpt-oss-120b 13

Latency is the model's own time; answers saved before pacing waits were recorded separately are left out of it (n shows how many count). Calls, failures and rate limits cover every call the collection made for the model, after the registry's own retries.
