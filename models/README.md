# Trained router

`router.json` is written by `scripts/train_router.py`: the embedder name, the
logistic-regression weights, and how it was trained (data size,
cross-validation results). The gateway loads it at startup when
`routing.router` is `classifier`; until it exists, the gateway uses the
rule-based router and says so in `/health`.
