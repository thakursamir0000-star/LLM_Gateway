import json

import numpy as np
import pytest

from gateway.routing import RulesRouter, build_router, choose_tier, routing_text
from gateway.routing.cascade import check_answer, should_escalate
from gateway.routing.classifier import FORMAT, ClassifierRouter
from gateway.routing.embeddings import HashingEmbedder, load_embedder
from gateway.routing.logistic import (
    cross_validate,
    fit_logistic,
    log_loss,
    predict_proba,
    roc_auc,
    stratified_folds,
)
from tests.conftest import make_config


def user(text):
    return [{"role": "user", "content": text}]


# ── interface ───────────────────────────────────────────────────────

def test_threshold_picks_the_tier():
    assert choose_tier(0.7, 0.5) == "cheap"
    assert choose_tier(0.5, 0.5) == "cheap"
    assert choose_tier(0.49, 0.5) == "strong"
    assert choose_tier(1.0, 0.0) == "cheap" and choose_tier(0.0, 0.0) == "cheap"


def test_routing_text_is_the_last_user_message():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "first"},
                {"role": "assistant", "content": "a"}, {"role": "user", "content": [{"type": "text", "text": "last"}]}]
    assert routing_text(messages) == "last"
    assert routing_text([{"role": "system", "content": "only system"}]) == "only system"


# ── rules ───────────────────────────────────────────────────────────

def test_rules_send_simple_requests_to_cheap_and_hard_ones_to_strong():
    router = RulesRouter()
    simple = router.score(user("Translate 'good morning' into French."))
    hard = router.score(user("Step by step, prove that the integral of x^2 from 0 to 1 is 1/3, then write "
                             "```python\ndef area(): ...``` code to check it numerically. Compare both approaches."))
    assert choose_tier(simple.score, 0.5) == "cheap"
    assert choose_tier(hard.score, 0.5) == "strong"
    assert "maths" in hard.reason and "code" in hard.reason
    assert 0.0 <= hard.score < simple.score <= 1.0


def test_rules_penalise_long_context():
    short = RulesRouter().score(user("Hello there, summarise this."))
    long = RulesRouter().score(user("word " * 3000))
    assert long.score < short.score and "long context" in long.reason


# ── embeddings and logistic regression ──────────────────────────────

def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(64)
    a = embedder.embed(["the cat sat", "the cat sat", ""])
    assert a.shape == (3, 64)
    assert np.allclose(a[0], a[1]) and np.isclose(np.linalg.norm(a[0]), 1.0)
    assert np.allclose(a[2], 0.0)
    assert load_embedder("hashing:64").name == "hashing:64"


def make_data(n=400, d=6, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = np.array([2.0, -1.5, 0.0, 0.5, 0.0, 1.0])[:d]
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ true_w + 0.3)))).astype(float)
    return X, y


def test_logistic_regression_learns_a_separable_signal():
    X, y = make_data()
    w, b = fit_logistic(X, y, l2=1.0)
    p = predict_proba(X, w, b)
    assert roc_auc(y, p) > 0.8
    assert log_loss(y, p) < log_loss(y, np.full(len(y), y.mean()))
    assert w[0] > 1 and w[1] < -0.8


def test_logistic_regression_matches_scikit_learn():
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y = make_data()
    for l2 in (0.1, 1.0, 10.0):
        w, b = fit_logistic(X, y, l2=l2)
        ref = sklearn.LogisticRegression(C=1 / l2, tol=1e-10, max_iter=10_000).fit(X, y)
        assert np.allclose(w, ref.coef_[0], atol=1e-4)
        assert b == pytest.approx(ref.intercept_[0], abs=1e-4)


def test_logistic_regression_with_one_class_predicts_a_constant():
    X = np.ones((5, 3))
    w, b = fit_logistic(X, np.ones(5))
    assert np.allclose(w, 0) and 0.9 < predict_proba(X, w, b)[0] < 1


def test_auc_and_folds():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert roc_auc([1, 1], [0.2, 0.3]) is None
    y = np.array([0] * 10 + [1] * 30)
    folds = stratified_folds(y, 5)
    assert sorted(np.concatenate(folds).tolist()) == list(range(40))
    assert all(y[f].sum() == 6 for f in folds)


def test_cross_validation_prefers_reasonable_regularisation():
    X, y = make_data(n=300)
    results = cross_validate(X, y, [0.01, 1.0, 1e4], k=5)
    by_l2 = {r["l2"]: r for r in results}
    assert by_l2[1e4]["log_loss"] > by_l2[1.0]["log_loss"]  # over-regularised is worse


# ── classifier router ───────────────────────────────────────────────

def trained_router(tmp_path):
    """A tiny router trained on hashing embeddings: maths/code questions need the strong model."""
    embedder = HashingEmbedder(128)
    easy = [f"what is the capital of country number {i}" for i in range(30)]
    hard = [f"prove the theorem and derive the integral bound number {i}" for i in range(30)]
    X = embedder.embed(easy + hard)
    y = np.array([1] * 30 + [0] * 30)
    w, b = fit_logistic(X, y, l2=0.1)
    router = ClassifierRouter(w, b, embedder, {"n_train": 60})
    path = tmp_path / "router.json"
    router.save(path, cheap="small", strong="large")
    return path


def test_classifier_router_round_trip(tmp_path):
    path = trained_router(tmp_path)
    data = json.loads(path.read_text())
    assert data["format"] == FORMAT and data["embedder"] == "hashing:128" and data["cheap"] == "small"
    router = ClassifierRouter.load(path)
    easy = router.score(user("what is the capital of country number 99"))
    hard = router.score(user("prove the theorem and derive the integral bound number 99"))
    assert easy.score > 0.5 > hard.score
    assert easy.router == "classifier" and easy.reason.startswith("P(cheap ok)=")
    assert router.score_texts(["a", "b"]).shape == (2,)


def test_classifier_rejects_a_mismatched_embedder(tmp_path):
    path = trained_router(tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        ClassifierRouter.load(path, embedder=HashingEmbedder(64))


def test_build_router_falls_back_to_rules(tmp_path):
    missing = make_config(routing={"router": "classifier", "classifier_path": str(tmp_path / "none.json")})
    router = build_router(missing)
    assert router.name == "rules" and "no trained classifier" in router.description

    broken = tmp_path / "broken.json"
    broken.write_text("{}")
    router = build_router(make_config(routing={"router": "classifier", "classifier_path": str(broken)}))
    assert router.name == "rules" and "failed to load" in router.description

    good = trained_router(tmp_path)
    assert build_router(make_config(routing={"router": "classifier", "classifier_path": str(good)})).name == \
        "classifier"
    assert build_router(make_config(routing={"router": "rules"})).name == "rules"


# ── cascade check ───────────────────────────────────────────────────

@pytest.mark.parametrize("answer, finish, expected", [
    ("", "stop", "empty answer"),
    ("The answer is", "length", "truncated"),
    ("It's 4.", "content_filter", "blocked"),
    ("I'm not sure, but maybe 12.", "stop", "hedged"),
    ("I cannot determine that from the text.", "stop", "hedged"),
    ("\n".join(["The same long line repeated again"] * 5), "stop", "repeats"),
    ("The answer is 42 because 6 times 7 is 42.", "stop", "no warning signs"),
])
def test_check_answer(answer, finish, expected):
    assert expected in check_answer("What is 6*7?", answer, finish).reason


def test_short_answer_to_a_long_question_is_less_confident():
    check = check_answer("x" * 400, "42", "stop")
    assert check.reason.startswith("very short")
    assert should_escalate(check, 0.8) and not should_escalate(check, 0.5)


def test_escalation_follows_the_threshold():
    good = check_answer("q", "A full and confident answer here.")
    hedged = check_answer("q", "I don't know.")
    assert not should_escalate(good, 0.5) and should_escalate(hedged, 0.5)
    assert should_escalate(good, 0.95)  # a very high threshold escalates everything
    assert not should_escalate(hedged, 0.0)  # threshold 0 never escalates
