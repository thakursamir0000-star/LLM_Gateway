"""The trained router: a sentence embedding of the request, then logistic regression.

It predicts P(the cheap model's answer is good enough). The model is a JSON
file written by ``scripts/train_router.py``: the embedder's name, the weights
and bias, and how it was trained (data size, cross-validation results).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from gateway.routing.base import RouteScore, routing_text
from gateway.routing.embeddings import Embedder, load_embedder
from gateway.routing.logistic import predict_proba

FORMAT = "llm-gateway-router/1"


class ClassifierRouter:
    name = "classifier"

    def __init__(self, weights: np.ndarray, bias: float, embedder: Embedder, meta: dict[str, Any] | None = None):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.embedder = embedder
        self.meta = meta or {}
        cheap, strong = self.meta.get("cheap"), self.meta.get("strong")
        self.model_aliases = (cheap, strong) if isinstance(cheap, str) and isinstance(strong, str) else None
        threshold = self.meta.get("threshold")
        self.selected_threshold = float(threshold) if isinstance(threshold, int | float) else None
        self.description = f"classifier ({embedder.name}, {self.meta.get('n_train', '?')} training questions)"

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> ClassifierRouter:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: not a router file (format {data.get('format')!r}, expected {FORMAT!r})")
        if not isinstance(data.get("cheap"), str) or not isinstance(data.get("strong"), str):
            raise ValueError(f"{path}: router file must name its cheap and strong model aliases")
        threshold = data.get("threshold")
        if not isinstance(threshold, int | float) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
            raise ValueError(f"{path}: router file must have a selected threshold in [0, 1]")
        embedder = embedder or load_embedder(data["embedder"])
        weights = np.asarray(data["weights"], dtype=np.float64)
        probe = embedder.embed(["dimension check"])
        if probe.shape[1] != len(weights):
            raise ValueError(f"{path}: {len(weights)} weights but the embedder gives {probe.shape[1]} dimensions")
        meta = {k: v for k, v in data.items() if k not in ("weights", "bias")}
        return cls(weights, data["bias"], embedder, meta)

    def save(self, path: str | Path, **meta: Any) -> None:
        data = {
            "format": FORMAT,
            "embedder": self.embedder.name,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **{k: v for k, v in self.meta.items() if k not in ("format", "embedder")},
            **meta,
            "bias": self.bias,
            "weights": [round(float(w), 7) for w in self.weights],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def score_texts(self, texts: list[str]) -> np.ndarray:
        """P(cheap is good enough) for each text (batched, used by the evaluation)."""
        return predict_proba(self.embedder.embed(texts), self.weights, self.bias)

    def score(self, messages: list[dict[str, Any]]) -> RouteScore:
        p = float(self.score_texts([routing_text(messages)])[0])
        return RouteScore(round(p, 4), self.name, f"P(cheap ok)={p:.2f}")
