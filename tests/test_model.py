"""Smoke tests for ExplainableXGB: fit/predict/explain, and the exact
prediction-decomposition property this method is built around.
"""

from __future__ import annotations

import numpy as np
import pytest

from explainablexgb import ExplainableXGB, ExplainableXGBMulticlass


def _make_binary_data(seed: int = 0, n: int = 800, p: int = 6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 0] * X[:, 2] + 0.2 * rng.normal(size=n)
    y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(int)
    names = [f"x{i}" for i in range(p)]
    return X, y, names


def test_fit_predict_shapes():
    X, y, names = _make_binary_data()
    model = ExplainableXGB(
        max_main_effects=6,
        max_interactions=3,
        xgb_params={"n_estimators": 40, "learning_rate": 0.1},
    )
    model.fit(X, y, feature_names=names)

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_exact_decomposition():
    """The model's raw score must equal base_score + sum of every term's
    contribution, for every accepted tree -- this is the central claim of
    the method, verified numerically rather than assumed."""
    X, y, names = _make_binary_data(seed=1)
    model = ExplainableXGB(
        max_main_effects=6,
        max_interactions=3,
        xgb_params={"n_estimators": 60, "learning_rate": 0.1},
    )
    model.fit(X, y, feature_names=names)

    raw_score, contributions = model._predict_margin_and_contribs(X)
    reconstructed = model.base_score_ + contributions.sum(axis=1)
    assert np.allclose(raw_score, reconstructed, atol=1e-8)

    # Local explanations must sum to the same raw score.
    row = X[:1]
    local = model.explain_local(row)
    total = sum(item["total_contribution"] for item in local["positive_contributions"])
    total += sum(item["total_contribution"] for item in local["negative_contributions"])
    assert total == pytest.approx(local["raw_score"] - local["base_score"], abs=1e-8)


def test_lambda_complexity_reduces_interactions():
    """A positive lambda_complexity must never accept more interactions than
    lambda_complexity=0 on the same data."""
    X, y, names = _make_binary_data(seed=2)
    kwargs = dict(
        max_main_effects=6,
        max_interactions=5,
        xgb_params={"n_estimators": 60, "learning_rate": 0.1},
    )
    lax = ExplainableXGB(lambda_complexity=0.0, **kwargs).fit(X, y, feature_names=names)
    strict = ExplainableXGB(lambda_complexity=1e6, **kwargs).fit(X, y, feature_names=names)
    assert len(strict._selected_interactions) <= len(lax._selected_interactions)
    assert len(strict._selected_interactions) == 0


def test_global_explanation_structure():
    X, y, names = _make_binary_data(seed=3)
    model = ExplainableXGB(
        max_main_effects=6, max_interactions=3,
        xgb_params={"n_estimators": 40, "learning_rate": 0.1},
    )
    model.fit(X, y, feature_names=names)
    global_exp = model.explain_global()
    assert "main_effects" in global_exp
    assert "interactions" in global_exp
    assert "extracted_rules" in global_exp


def test_multiclass_ovr():
    rng = np.random.default_rng(4)
    n, p = 600, 5
    X = rng.normal(size=(n, p))
    logits = np.stack(
        [0.5 * X[:, 0], -0.5 * X[:, 1], 0.5 * X[:, 0] - 0.5 * X[:, 1]], axis=1
    )
    y = np.argmax(logits + 0.1 * rng.normal(size=logits.shape), axis=1)
    names = [f"x{i}" for i in range(p)]

    model = ExplainableXGBMulticlass(
        max_main_effects=5, max_interactions=2,
        xgb_params={"n_estimators": 30, "learning_rate": 0.1},
    )
    model.fit(X, y, feature_names=names)

    proba = model.predict_proba(X)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1, 2})
