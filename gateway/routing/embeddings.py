"""Sentence embedders for the classifier router.

- ``sentence-transformers/...`` (any sentence-transformers model name): the
  real embedder, used for training and serving. Needs ``sentence-transformers``
  (requirements-lab.txt, or the Docker build arg WITH_EMBEDDINGS=true).
- ``hashing:<dim>``: a deterministic hashed bag of words and word pairs. No
  download, no model: used by the tests and as a light fallback.
- ``concat:<spec>+<spec>``: the parts' embeddings side by side, each weighted
  equally, then normalised (router v2: a sentence embedding plus lexical
  features).
Embeddings are L2-normalised float32 rows.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Protocol

import numpy as np

_WORD = re.compile(r"\w+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """An (n, dim) array of L2-normalised embeddings."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        if dim < 8:
            raise ValueError("hashing embedder needs dim >= 8")
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
                matrix[row, digest % self.dim] += 1.0 if digest >> 63 else -1.0
        return _normalise(matrix)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("the classifier router needs sentence-transformers: "
                              "pip install -r requirements-lab.txt") from e
        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False, batch_size=32)
        return np.asarray(vectors, dtype=np.float32)


class ConcatEmbedder:
    def __init__(self, parts: list[Embedder]):
        if len(parts) < 2:
            raise ValueError("concat embedder needs at least two parts")
        self.parts = parts
        self.name = "concat:" + "+".join(part.name for part in parts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.hstack([part.embed(texts) for part in self.parts]))


_CACHE: dict[str, Embedder] = {}
_LOCK = threading.Lock()


def load_embedder(spec: str) -> Embedder:
    """The embedder named by ``spec`` (loaded once per process)."""
    with _LOCK:
        if spec not in _CACHE:
            if spec.startswith("concat:"):
                names = spec[len("concat:"):].split("+")
                _CACHE[spec] = ConcatEmbedder([_load_one(name) for name in names])
            elif spec.startswith("hashing"):
                _, _, dim = spec.partition(":")
                _CACHE[spec] = HashingEmbedder(int(dim) if dim else 512)
            else:
                _CACHE[spec] = SentenceTransformerEmbedder(spec)
        return _CACHE[spec]


def _load_one(spec: str) -> Embedder:
    if spec.startswith("hashing"):
        _, _, dim = spec.partition(":")
        return HashingEmbedder(int(dim) if dim else 512)
    return SentenceTransformerEmbedder(spec)
