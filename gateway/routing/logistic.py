"""L2-regularised logistic regression in numpy, plus the metrics used to pick it.

Fitted with Newton's method and a backtracking line search; it minimises
``sum(log loss) + l2/2 * ||w||^2`` (the bias is not penalised), which is
scikit-learn's ``LogisticRegression(C=1/l2)``. Keeping it in numpy means the
gateway serves the router without scikit-learn; a test checks that both give
the same coefficients.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 100,
                 tol: float = 1e-8) -> tuple[np.ndarray, float]:
    """Weights and bias of an L2-regularised logistic regression."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    if n == 0:
        raise ValueError("no training rows")
    if y.min() == y.max():  # one class only: a constant, Laplace-smoothed prediction
        positives = float(y.sum())
        return np.zeros(d), float(np.log((positives + 0.5) / (n - positives + 0.5)))
    A = np.hstack([X, np.ones((n, 1))])
    reg = np.full(d + 1, float(l2))
    reg[-1] = 0.0
    theta = np.zeros(d + 1)

    def objective(t: np.ndarray) -> float:
        z = A @ t
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(reg * t * t))

    value = objective(theta)
    for _ in range(max_iter):
        p = sigmoid(A @ theta)
        grad = A.T @ (p - y) + reg * theta
        hessian = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hessian, grad)
        decrease = float(grad @ step)
        if decrease < tol:
            break
        size = 1.0
        while True:
            candidate = theta - size * step
            new_value = objective(candidate)
            if new_value <= value - 1e-4 * size * decrease or size < 1e-10:
                break
            size *= 0.5
        theta, value = candidate, new_value
    return theta[:-1], float(theta[-1])


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(np.asarray(X, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + bias)


# ── metrics ─────────────────────────────────────────────────────────

def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Area under the ROC curve (rank formula, ties averaged); None with one class."""
    y = np.asarray(y, dtype=bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return None
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    i = 0
    while i < len(p):  # average ranks over ties
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def stratified_folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Index arrays of ``k`` folds with about the same class balance each."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds: list[list[int]] = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for i, index in enumerate(idx):
            folds[i % k].append(int(index))
    return [np.array(sorted(f), dtype=int) for f in folds]


def cross_validate(X: np.ndarray, y: np.ndarray, l2_grid: list[float], k: int = 5,
                   seed: int = 0) -> list[dict[str, float | None]]:
    """Mean held-out log loss and AUC of each l2 over ``k`` stratified folds."""
    X = np.asarray(X)
    y = np.asarray(y)
    folds = stratified_folds(y, k, seed)
    results = []
    for l2 in l2_grid:
        losses, aucs = [], []
        for fold in folds:
            if len(fold) == 0:
                continue
            train = np.setdiff1d(np.arange(len(y)), fold)
            w, b = fit_logistic(X[train], y[train], l2)
            p = predict_proba(X[fold], w, b)
            losses.append(log_loss(y[fold], p))
            auc = roc_auc(y[fold], p)
            if auc is not None:
                aucs.append(auc)
        results.append({"l2": l2, "log_loss": float(np.mean(losses)),
                        "auc": float(np.mean(aucs)) if aucs else None})
    return results
