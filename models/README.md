# Trained router

`router.json` is written by `scripts/train_router.py`: the embedder name, the
logistic-regression weights, and how it was trained (data size,
cross-validation results), its selected threshold, and the cheap and strong
model aliases it was trained for. The gateway loads it at startup when
`routing.router` is `classifier` only when its selected threshold equals
`routing.classifier_threshold`. Until it exists or while validation fails,
the gateway uses the rule-based router and says so in `/health`.

Before each classifier route, the gateway also requires the artifact aliases
to match the selected release and latency class. A mismatch is sent to the
strong model. The audited artifact in `results/router_openrouter_1200.json`
matches the `v4-latency` batch aliases and does not match its interactive
Groq aliases. With the shipped `default_mode: strong`, an installed artifact
would therefore be used only by an explicit `model="router"` batch request.
