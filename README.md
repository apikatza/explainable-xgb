# ExplainableXGB

An inherently interpretable additive model built directly from XGBoost's own native staged-boosting
and interaction-constraint primitives — no custom tree learner, no modification to XGBoost's C++ core.

`ExplainableXGB` trains XGBoost in two restricted stages:

1. **Main-effect stage** — depth-1 trees only, so every tree's contribution is by construction a
   univariate function of a single feature.
2. **Interaction stage** — continued boosting, with each stage constrained via XGBoost's native
   `interaction_constraints` to a single selected feature pair.

Because every accepted tree is grown under one of these two constraints, the model's prediction is an
*exact* sum of main-effect and pairwise-interaction terms — the same functional form as GA²M models
(e.g. Explainable Boosting Machines), but produced directly by XGBoost's own training loop rather than a
separate cyclic-boosting algorithm.

Candidate interaction pairs are ranked by real split gain from a short, disposable exploratory booster
(not a static heuristic), and a candidate is accepted only if its gain exceeds an explicit
`lambda_complexity` threshold — a discrete penalised selection rule, similar in spirit to forward
stepwise regression with an AIC/C<sub>p</sub>-style penalty, giving direct control over the
complexity/accuracy trade-off of the resulting model.

## Install

```bash
pip install .
# or, for development:
pip install -e ".[test]"
```

Requires `numpy`, `scipy` and `xgboost` (>=2.0).

## Quickstart

```python
from explainablexgb import ExplainableXGB

model = ExplainableXGB(
    max_main_effects=100,
    max_interactions=10,
    max_depth_interaction=3,
    lambda_complexity=0.0,       # >0 trades interaction count for a real gain threshold
    xgb_params={"n_estimators": 400, "learning_rate": 0.05, "objective": "binary:logistic"},
)
model.fit(X_train, y_train, feature_names=feature_names)

model.predict_proba(X_test)
model.explain_global()          # main effects + interactions, ranked by importance
model.explain_local(X_test[:1]) # exact per-row score decomposition
```

The configuration above (`n_estimators=400`, `max_main_effects=100`, `max_depth_interaction=3`) is the
capacity-matched setting the accompanying paper evaluates as its main protocol; a smaller, faster
configuration (e.g. `n_estimators=120`, `max_main_effects=30`, `max_depth_interaction=2`) still trains a
valid model with the same exact-decomposition guarantee, just at lower capacity — see "Why this exists"
below for what that trade-off costs.

Multiclass tasks (one-vs-rest, softmax over per-class raw margins):

```python
from explainablexgb import ExplainableXGBMulticlass

model = ExplainableXGBMulticlass(max_main_effects=65, max_interactions=8, max_depth_interaction=3)
model.fit(X_train, y_train, feature_names=feature_names)
model.predict_proba(X_test)
```

## Why this exists

Gradient-boosted tree ensembles are strong predictors but not directly interpretable. GA²M-style
additive-plus-interaction models (e.g. Explainable Boosting Machines, EBM) trade some flexibility for an
exactly decomposable structure, fit with a cyclic, multi-pass, term-by-term training procedure.
`ExplainableXGB` asks whether that same structure can be produced with no custom training algorithm at
all — using only XGBoost's own native `interaction_constraints` and staged boosting, and whether a
larger single-pass training budget can substitute for EBM's repeated revisiting of every term. At a
sufficiently large budget, it can, on several benchmarks: `ExplainableXGB` significantly exceeds EBM on
some datasets and metrics, ties it on others, and remains behind on others — a real, if partial and
metric-dependent, predictive improvement, not a uniform win. That gain is not free: it costs several
times more training and inference time than an unconstrained XGBoost model, and on at least one dataset
it trades away cross-validation explanation stability that a smaller-capacity configuration has. See the
accompanying paper for the full evaluation, including exactly where this trade-off does and does not pay
off.

## Testing

```bash
pytest tests/
```

## Citation

This package builds directly on XGBoost's native training primitives (`interaction_constraints`,
staged boosting via `xgb_model` continuation) — if you use it, please cite XGBoost itself alongside this
repository (see `CITATION.cff`):

- Amaia Pikatza-Huerga and Asier Gonzalez-Santocildes. *ExplainableXGB*.
  https://github.com/apikatza/explainable-xgb
- Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. KDD 2016.
  https://doi.org/10.1145/2939672.2939785

## Licence

Apache License 2.0 — see `LICENSE`.
