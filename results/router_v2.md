# Router v2 against v1 (2026-11-22)

Same data, split and evaluator as v1 (`scripts/train_router.py`, `scripts/evaluate_router.py`): 1,200 questions
(gsm8k, mmlu), 900 for training, 300 held out; cheap `or-gpt-oss-20b`, strong `or-gpt-oss-120b` on OpenRouter; label
"cheap is good enough" (cheap right, or strong wrong too). No API calls: the answers were collected earlier.

| | v1 (shipped candidate) | v2 |
|---|---|---|
| Features | MiniLM-L6 sentence embedding (384) | MiniLM-L6 embedding + hashed word and word-pair features (256), concatenated |
| Embedder spec | `sentence-transformers/all-MiniLM-L6-v2` | `concat:sentence-transformers/all-MiniLM-L6-v2+hashing:256` |
| Cross-validated log loss / AUC (l2 = 1) | 0.2402 / 0.649 | 0.2395 / 0.660 |
| Out-of-fold AUC | 0.638 | 0.648 |
| Threshold (chosen on training data) | 0.955 | 0.945 |
| Held-out accuracy (always-strong 88.3%, always-cheap 85.0%) | 88.0% | 87.0% |
| Held-out cost per 1k requests (always-strong $0.0282) | $0.0259 (8% saving) | $0.0249 (12% saving) |
| Sent to strong | 67% | 50% |
| Random routing at the same share | 87.2% | 86.7% |
| vs always-strong (exact McNemar) | −0.3 pts, 0 vs 1 discordant, p = 1.0 | −1.3 pts, 0 vs 4 discordant, p = 0.125 |

**Decision: keep v1 as the candidate; v2 is not promoted.** v2's small cross-validation gain did not carry over to
the held-out questions: it buys 4 more points of saving with 1 more point of accuracy loss, and it is only 0.3 points
above random routing at the same share. Neither router is significantly different from always-strong on 300
questions, and both are benchmark-only evidence. The next useful step is not a third offline model but shadow mode
on real traffic (`routing.shadow`), which measures how the classifier would route the gateway's actual requests.

Artifacts: `results/router_v2_openrouter_1200.json` (the v2 model), `results/router_v2_eval.json` (its full
evaluation); v1's are `results/router_openrouter_1200.json` and `results/router_eval.json`.
