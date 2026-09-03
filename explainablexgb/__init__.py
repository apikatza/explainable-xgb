"""ExplainableXGB: an inherently interpretable additive model built directly
from XGBoost's own native staged-boosting and interaction-constraint
primitives -- no custom tree learner, no modification to XGBoost's C++ core.

    from explainablexgb import ExplainableXGB

    model = ExplainableXGB(max_main_effects=30, max_interactions=10)
    model.fit(X_train, y_train, feature_names=feature_names)
    model.predict_proba(X_test)
    model.explain_global()
    model.explain_local(X_test[:1])

See README.md for the full API and the accompanying paper for the method.
"""

from .model import (
    AggregatedFeatureEffect,
    ExplainableXGB,
    ExplainableXGBClassifier,
    ExplainableXGBMulticlass,
    ExplainableXGBOvR,
    TreeTerm,
)

__all__ = [
    "ExplainableXGB",
    "ExplainableXGBClassifier",
    "ExplainableXGBMulticlass",
    "ExplainableXGBOvR",
    "TreeTerm",
    "AggregatedFeatureEffect",
]

__version__ = "0.1.0"
