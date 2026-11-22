# Routing, the router lab and evaluation results

Reference for how the gateway chooses a model, how the trained router was built and evaluated, and how it would be
rolled out. The [README](../README.md) has the summary.

## Routing

Two choices are made per request: the **tier** (cheap or strong model) and the **latency class** (which provider
serves it). The shipped config routes by latency class: on the measured questions the strong model was the better buy
(see [Provider comparison](#provider-comparison-measured-2026-11-15-to-2026-11-16)), and the bigger trade-off was speed against cost
across providers.

**Latency classes** (`interactive` | `batch`): a request sets its class with the `X-Gateway-Latency` header; otherwise
the app's default from `latency.apps` applies, otherwise `latency.default` (`interactive`). A release can name other
models for batch traffic; the active release `v4-latency` does:

| | Interactive (fast) | Batch (cheaper, slower) |
|---|---|---|
| strong | Groq `gpt-oss-120b` | OpenRouter `gpt-oss-120b` (DeepInfra bf16) |
| cheap | Groq `gpt-oss-20b` | OpenRouter `gpt-oss-20b` (DeepInfra bf16) |

Releases without a `batch` block (such as `v1`, Groq only) serve both classes with the same models. Pinned models
ignore the class.

**Fallback across providers**: when a model fails, the gateway tries the same tier on the other class's model (the
other provider), then both models in the other tier. Duplicate aliases are removed for releases that use the same
models for both classes. A model that hits a quota error (Groq's daily limit, OpenRouter out of credits) is skipped for
`retries.quota_cooldown_s` (300 s, or longer if the provider says so), so later requests go straight to the next
candidate. Interactive traffic that falls back to OpenRouter is slower, but answered.

| `model` value | What happens |
|---|---|
| `auto` | the configured `routing.default_mode` (`strong` in the shipped config) |
| `router` | the router's score against the threshold picks cheap or strong |
| `cascade` | cheap model first; escalate to strong when the answer check is below the threshold |
| `cheap` / `strong` | always that tier of the active (or canary) release |
| `gpt-oss-120b`, `openai/gpt-oss-120b`, ... | a configured model, pinned: no routing, no fallback |

**Thresholds**: `routing.classifier_threshold` is the classifier's authoritative runtime default. It must match the
training-selected threshold recorded in the artifact or the gateway refuses to load that classifier.
`routing.threshold` is the default for the rules router and cascade. Every router outputs P(cheap model is good
enough); requests with a score below its threshold go to the strong model. In cascade mode, cheap answers whose
confidence is below `routing.threshold` are escalated. `X-Gateway-Threshold` is an explicit, logged per-request
override. `0` = always cheap, `1` = (almost) always strong.

**Rule-based router** (`gateway/routing/rules.py`): hand-weighted signals of a hard request (long context, code,
maths, multi-step reasoning words, several questions, a long conversation). It is the baseline the trained router has
to beat.

**Classifier router** (`gateway/routing/classifier.py`): a sentence embedding of the last user message
(`sentence-transformers/all-MiniLM-L6-v2` by default) followed by logistic regression that predicts whether the cheap
model is good enough. The regression is fitted in numpy with Newton's method (a test checks it matches scikit-learn),
so serving it needs no scikit-learn. It is loaded from `models/router.json`; until that file exists, the gateway falls
back to the rules and says so in `/health`. The artifact names the cheap and strong aliases it was trained for. A
classifier request is allowed only when those aliases match the selected release and latency class; a mismatch uses
the strong model and records the reason.

The prepared OpenRouter canary keeps `routing.default_mode: strong`. After the audited artifact is deliberately
installed, only requests that explicitly use `model="router"` with `X-Gateway-Latency: batch` use it. Interactive
router requests stay on the strong model because the artifact is trained for the OpenRouter batch aliases.

**Cascade check** (`gateway/routing/cascade.py`): no extra model call, but it only catches answers that *look* wrong;
a confident wrong answer passes. The evaluation measures how much that matters.

## Router shadow mode and controlled activation

The trained classifier is never switched on in one step. Three stages, each read with `scripts/router_report.py`
(from the request log; `--json` to save it):

1. **Shadow** (`routing.shadow: true`): every routable request is still answered as configured, and the classifier
   is asked which tier it would have chosen; `requests.shadow_tier`/`shadow_score` and
   `gateway_shadow_decisions_total` record it. The report gives agreement with the tier used, the share it would
   send to the cheap model, and an estimated saving (strong answers it would have sent cheap, re-priced at the cheap
   model's list price). It cannot tell whether those cheap answers would have been good enough: compare user
   feedback per group, and sample requests for review.
2. **Canary** (`default_mode: router` or `model="router"`, with `routing.classifier_share: 0.05`): the classifier
   routes that share of router-mode requests, chosen by a hash of the request id; the rest are a control group
   answered by the strong tier (and shadow-scored). The report compares the groups' error rate, cost, latency and
   feedback. Raise the share in steps (0.05, 0.25, 1.0) only when the classifier group is no worse.
3. **Rollback** at any stage: `classifier_share: 0` (or `routing.router: rules`) and a rolling restart.

Only the model pair the classifier was trained for is ever routed or shadow-scored by it; with the shipped config
that is the OpenRouter batch pair. The shipped config keeps shadow off and `default_mode: strong`, and
`models/router.json` is not installed: turning any of this on is a deployment decision.

Retraining from feedback (Phase 6.6) is not built. The request log stores no prompt text, by design, so thumbs-up
or thumbs-down ratings can't be joined to the questions they rate. A retraining set would need an opt-in prompt
capture with its own retention and consent, and enough rated traffic to beat the 900 benchmark questions.

## Router lab: data, training, evaluation

The router is trained on which questions the cheap model gets right. The OpenRouter collection and held-out
evaluation are complete; the scripts use API keys and quota only when collecting new labels.

```bash
pip install -r requirements-lab.txt
```

1. **Questions** (no API calls): samples 400 GSM8K and 800 MMLU test questions and fixes a 75/25 train/test split
   before any model sees them. It refuses to overwrite the file, so the split can't change after answers exist.

   ```bash
   python scripts/prepare_questions.py
   ```

   If the download stops asking for `hf_xet`, set `HF_HUB_DISABLE_XET=1` (plain HTTP downloads) and rerun.

2. **Labels** (spends API quota): runs the cheap and the strong model on every question and grades the answers.
   Calls are paced under the per-minute limits in the config; every answer is appended to
   `router_data/answers.jsonl` the moment it arrives; rerunning skips everything already answered. It stops at a
   daily quota, a key problem or five errors in a row, and asks before spending quota (`--yes` skips the question).

   ```bash
   python scripts/collect_labels.py --dry-run
   ```

   ```bash
   python scripts/collect_labels.py --limit 20
   ```

   ```bash
   python scripts/collect_labels.py --yes
   ```

   Groq's free tier allows `openai/gpt-oss-120b` 8,000 tokens per minute and 200,000 per day. In a 20-question pilot
   a question took about 350-380 tokens per model (prompt plus answer, low reasoning effort), so all 1,200 questions
   need roughly 0.45M tokens per model: about three days of gpt-oss-120b's daily quota.

3. **Train** (no API calls; downloads the embedder once). Label: the cheap model is good enough when it is right, or
   when the strong model is wrong too. Regularisation is chosen by stratified 5-fold cross-validation.

   ```bash
   python scripts/train_router.py
   ```

4. **Evaluate** (no API calls): replays the held-out answers for always-cheap, always-strong, random routing (same
   share sent to strong as the router), the rules, the classifier and the cascade. It reports accuracy, cost per 1,000
   requests at list prices, the share sent to the strong model, the router's added latency, and the cheapest threshold
   within 1 percentage point of always-strong accuracy, and draws `results/cost_vs_accuracy.png`.

   ```bash
   python scripts/evaluate_router.py
   ```

5. **Compare providers** (no API calls). Another provider's models answer the same questions into their own file,
   with the provider, the host that served each answer, tokens, latency, the list-price estimate and the reported
   cost; a lock file stops two runs writing one file. For example, OpenRouter on 10 of the questions both Groq models
   answered, reading the key from another project's `.env` (never printed):

   ```bash
   python scripts/collect_labels.py --cheap or-gpt-oss-20b --strong or-gpt-oss-120b --answers router_data/answers_openrouter.jsonl --only-answered-in router_data/answers.jsonl --sample 10 --env-file ../Agentai/.env --yes
   ```

   `compare_providers.py` then compares every model on the questions they all answered: accuracy with 95%
   intervals, input/output tokens, cost per 1,000 questions, latency mean/p50/p95, failures and rate limits, and how
   often one model gives a different result on two hosts.

   ```bash
   python scripts/compare_providers.py
   ```

## Evaluation results

OpenRouter answered all 1,200 questions with both models. The collection completed with no final-call errors and
reported $0.06608 total cost, including the earlier 377-question comparison set. Dataset validation found no
integrity problems. The complete reports are in
[`results/dataset_quality.md`](../results/dataset_quality.md),
[`results/provider_comparison_openrouter_1200.md`](../results/provider_comparison_openrouter_1200.md), and
[`results/router_eval.md`](../results/router_eval.md).

On the fixed 300-question held-out split, the trained classifier reached 88.0% accuracy versus 88.3% for always
using the strong model. It sent 67% of questions to the strong model and reduced estimated cost by 8% ($0.0259
versus $0.0282 per 1,000 requests). The one-answer difference was not significant (exact McNemar p = 1.000).
Classifier scoring added 42 ms p50 and 117 ms p95 on this CPU. Its threshold is 0.955, selected from out-of-fold
training scores and mirrored by `routing.classifier_threshold`. A mismatch prevents classifier loading.

The rules baseline selected a threshold that sent every held-out question to the strong model. The cascade also
escalated every question and cost 83% more than always-strong, so neither improved this OpenRouter pair.

### Router v2 (2026-11-22, offline)

A second classifier, MiniLM embeddings plus hashed word and word-pair features, trained and evaluated exactly like
v1, scored better in cross-validation (out-of-fold AUC 0.648 vs 0.638) but not on the held-out questions: 87.0%
accuracy with a 12% saving (v1: 88.0%, 8%), only 0.3 points above random routing at its strong share, and not
significantly different from always-strong (p = 0.125). v1 stays the candidate. Details:
[`results/router_v2.md`](../results/router_v2.md).

### Provider comparison (measured 2026-11-15 to 2026-11-16)

The same open-weight models on Groq and on OpenRouter (pinned to its DeepInfra bf16 host), on the 377 identical
GSM8K + MMLU questions, low reasoning effort. The consolidated table is in
[`results/provider_comparison.md`](../results/provider_comparison.md). The full 1,200-question OpenRouter comparison is
in [`results/provider_comparison_openrouter_1200.md`](../results/provider_comparison_openrouter_1200.md).

| Model | Accuracy (95% CI) | Cost per 1,000 questions | Latency p50 / p95 |
|---|---|---|---|
| Groq gpt-oss-20b | 83.6% (79-87%) | $0.052 | 0.57 / 0.91 s |
| Groq gpt-oss-120b | 87.3% (84-90%) | $0.107 | 0.69 / 1.12 s |
| OpenRouter gpt-oss-20b | 84.4% (80-88%) | $0.025 | 1.41 / 3.37 s |
| OpenRouter gpt-oss-120b | 88.6% (85-91%) | $0.030 | 4.34 / 15.48 s |

- The host made no measurable difference to accuracy (same model, Groq vs OpenRouter: exact McNemar p = 0.75 for 20b,
  0.38 for 120b). The same model gave the same final answer on both hosts for 88% (20b) and 94% (120b) of questions.
- 120b beat 20b on OpenRouter (p = 0.011); on Groq the gap was borderline (p = 0.059).
- OpenRouter's 120b was both more accurate than Groq's 20b (p = 0.005) and 43% cheaper, but its latency was 6-14x
  higher, with a long tail.
- On OpenRouter 120b costs only 17% more than 20b, so cheap/strong routing there can save little; the larger trade-off
  is latency against cost, across providers.
- Across all 1,200 OpenRouter questions, 120b scored 88.8% and 20b scored 84.9%. The difference was significant
  (exact McNemar p < 0.001), and the models produced the same final answer on 88% of questions.

Costs are list prices (OpenRouter's reported cost matched them on every call); Groq ran on its free tier. Latency was
measured from one machine on one day; Groq latency excludes 20 early answers recorded with local pacing waits. Two
OpenRouter calls hit the host's rate limit and succeeded on a rerun.
