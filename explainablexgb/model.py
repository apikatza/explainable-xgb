"""ExplainableXGB: an inherently interpretable additive model built from
XGBoost's own native staged-boosting and interaction-constraint primitives.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from xgboost import DMatrix, train


@dataclass(frozen=True)
class TreeTerm:
    tree_id: int
    term_type: str
    features: Tuple[str, ...]
    raw_tree: Any
    weight: float
    direction: str
    domain_label: Optional[str] = None


@dataclass(frozen=True)
class AggregatedFeatureEffect:
    feature: str
    label: str
    total_importance: float
    total_contribution: Optional[float]
    direction: str
    shape: str
    shape_confidence: float
    monotonic_confidence: float
    effect_amplitude: float
    num_sign_changes: int
    thresholds: Mapping[str, Any]
    n_terms: int
    summary: str


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def normalise_pair(pair: Iterable[str]) -> Tuple[str, str]:
    a, b = tuple(pair)
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def feature_label(feature: str, metadata: Mapping[str, Mapping[str, Any]]) -> str:
    return str(metadata.get(feature, {}).get("label", feature))


def smooth_effect_values(values: Sequence[float], strength: str = "moderate") -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return [float(v) for v in arr]
    fractions = {"none": 0.0, "light": 0.10, "moderate": 0.18, "strong": 0.28}
    fraction = fractions.get(str(strength), fractions["moderate"])
    if fraction <= 0:
        return [float(v) for v in arr]
    window = max(3, int(round(arr.size * fraction)))
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window < 3:
        return [float(v) for v in arr]
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(v) for v in smoothed]


def analyse_effect_shape(
    effect_curve: Mapping[str, Sequence[float]],
    *,
    shape_smoothing_strength: str = "moderate",
) -> Dict[str, Any]:
    values = np.asarray(effect_curve.get("contributions", []), dtype=float)
    empty = {
        "direction": "flat",
        "shape": "flat",
        "shape_confidence": 1.0,
        "monotonic_confidence": 1.0,
        "effect_amplitude": 0.0,
        "num_sign_changes": 0,
        "smoothed_curve": [float(v) for v in values],
    }
    if values.size < 2 or not np.isfinite(values).any():
        return empty

    smoothed = np.asarray(
        smooth_effect_values(values, strength=shape_smoothing_strength), dtype=float
    )
    amplitude = float(np.nanmax(smoothed) - np.nanmin(smoothed))
    max_abs = float(np.nanmax(np.abs(smoothed)))
    low_amplitude = amplitude <= max(1e-8, 0.05 * max(1.0, max_abs))
    tolerance = max(1e-12, 0.05 * amplitude)
    diffs = np.diff(smoothed)
    filtered = np.where(np.abs(diffs) < tolerance, 0.0, diffs)
    signs = np.sign(filtered)
    nonzero = signs[signs != 0]
    num_sign_changes = int(np.sum(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0

    if filtered.size == 0 or low_amplitude or nonzero.size == 0:
        return {
            **empty,
            "effect_amplitude": amplitude,
            "num_sign_changes": num_sign_changes,
            "smoothed_curve": [float(v) for v in smoothed],
        }

    positive_ratio = float(np.mean(filtered > 0))
    negative_ratio = float(np.mean(filtered < 0))
    flat_ratio = float(np.mean(filtered == 0))
    nonzero_count = max(1, int(np.sum(filtered != 0)))
    positive_meaningful_ratio = float(np.sum(filtered > 0) / nonzero_count)
    negative_meaningful_ratio = float(np.sum(filtered < 0) / nonzero_count)
    dominant = max(positive_meaningful_ratio, negative_meaningful_ratio)
    monotonic_confidence = min(1.0, dominant * (1.0 - 0.25 * flat_ratio))
    if positive_ratio > 0.70 or (positive_meaningful_ratio > 0.70 and num_sign_changes <= 1):
        direction = "increases_prediction"
    elif negative_ratio > 0.70 or (negative_meaningful_ratio > 0.70 and num_sign_changes <= 1):
        direction = "decreases_prediction"
    else:
        direction = "non_monotonic"

    abs_diffs = np.abs(filtered[filtered != 0])
    median_change = float(np.median(abs_diffs)) if abs_diffs.size else 0.0
    large_change = np.abs(filtered) >= max(tolerance, 2.0 * median_change)
    if np.sum(large_change) >= 1 and np.mean(np.abs(filtered) <= tolerance) > 0.35:
        first_large = int(np.argmax(large_change))
        tail = filtered[first_large + 1 :]
        if tail.size and float(np.mean(np.abs(tail) <= tolerance)) > 0.65:
            shape = "saturation"
            shape_confidence = 0.85
        else:
            shape = "threshold"
            shape_confidence = 0.80
    elif direction in {"increases_prediction", "decreases_prediction"}:
        nonzero_steps = int(np.sum(filtered != 0))
        if nonzero_steps <= max(2, int(0.5 * filtered.size)):
            shape = "stepwise"
            shape_confidence = 0.85
        else:
            shape = "linear"
            shape_confidence = max(0.65, monotonic_confidence)
    else:
        shape = "stepwise" if num_sign_changes <= 2 else "non_monotonic"
        shape_confidence = 0.65 if shape == "stepwise" else 0.55

    # Exotic shapes are deliberately last and require persistent, deep evidence.
    min_segment = max(1, int(math.ceil(0.20 * smoothed.size)))
    if shape in {"stepwise", "non_monotonic"} and smoothed.size >= 5:
        valley = int(np.argmin(smoothed))
        if min_segment <= valley <= smoothed.size - min_segment - 1:
            left = filtered[:valley]
            right = filtered[valley:]
            left_down = float(np.mean(left < 0)) if left.size else 0.0
            right_up = float(np.mean(right > 0)) if right.size else 0.0
            left_depth = float(np.nanmax(smoothed[: valley + 1]) - smoothed[valley])
            right_depth = float(np.nanmax(smoothed[valley:]) - smoothed[valley])
            u_depth = min(left_depth, right_depth)
            confidence = min(left_down, right_up, u_depth / max(tolerance, amplitude))
            if left_down >= 0.60 and right_up >= 0.60 and u_depth >= 0.20 * amplitude and confidence >= 0.60:
                direction = "u_shape"
                shape = "u_shape"
                shape_confidence = max(0.80, min(1.0, confidence))

        peak = int(np.argmax(smoothed))
        if shape not in {"u_shape"} and min_segment <= peak <= smoothed.size - min_segment - 1:
            left = filtered[:peak]
            right = filtered[peak:]
            left_up = float(np.mean(left > 0)) if left.size else 0.0
            right_down = float(np.mean(right < 0)) if right.size else 0.0
            left_depth = float(smoothed[peak] - np.nanmin(smoothed[: peak + 1]))
            right_depth = float(smoothed[peak] - np.nanmin(smoothed[peak:]))
            inv_depth = min(left_depth, right_depth)
            confidence = min(left_up, right_down, inv_depth / max(tolerance, amplitude))
            if left_up >= 0.60 and right_down >= 0.60 and inv_depth >= 0.20 * amplitude and confidence >= 0.60:
                direction = "inverted_u_shape"
                shape = "inverted_u_shape"
                shape_confidence = max(0.80, min(1.0, confidence))

    if shape in {"u_shape", "inverted_u_shape"} and shape_confidence < 0.60:
        shape = "non_monotonic"

    return {
        "direction": direction,
        "shape": shape,
        "shape_confidence": float(shape_confidence),
        "monotonic_confidence": float(monotonic_confidence),
        "effect_amplitude": amplitude,
        "num_sign_changes": num_sign_changes,
        "smoothed_curve": [float(v) for v in smoothed],
    }


def infer_effect_direction(effect_curve: Mapping[str, Sequence[float]]) -> str:
    return str(analyse_effect_shape(effect_curve)["direction"])


def infer_effect_shape(effect_curve: Mapping[str, Sequence[float]]) -> str:
    return str(analyse_effect_shape(effect_curve)["shape"])


def generate_global_summary(global_explanation: Mapping[str, Any]) -> str:
    effects = list(global_explanation.get("main_effects", []))[:3]
    interactions = list(global_explanation.get("interactions", []))[:2]
    if not effects and not interactions:
        return "El modelo no ha identificado efectos agregados dominantes."
    labels = [effect.get("label", effect.get("feature", "")) for effect in effects]
    text = "El modelo utiliza principalmente " + ", ".join(labels) + "."
    increasing = [
        effect.get("label", effect.get("feature", ""))
        for effect in effects
        if effect.get("direction") == "increases_prediction"
    ]
    if increasing:
        text += " La prediccion aumenta especialmente con "
        text += ", ".join(increasing) + "."
    if interactions:
        pairs = [
            " + ".join(interaction.get("labels", interaction.get("features", [])))
            for interaction in interactions
        ]
        text += " Tambien usa interacciones como " + ", ".join(pairs) + "."
    return text


def generate_local_summary(local_explanation: Mapping[str, Any]) -> str:
    probability = float(local_explanation.get("predicted_probability", 0.0))
    positive = list(local_explanation.get("positive_contributions", []))[:2]
    negative = list(local_explanation.get("negative_contributions", []))[:2]
    text = (
        "Esta observacion tiene probabilidad estimada de clase positiva "
        f"del {probability:.1%}."
    )
    if positive:
        terms = [item.get("label", item.get("term", "")) for item in positive]
        text += " La prediccion aumenta principalmente por "
        text += ", ".join(terms) + "."
    if negative:
        terms = [item.get("label", item.get("term", "")) for item in negative]
        text += " La prediccion disminuye principalmente por "
        text += ", ".join(terms) + "."
    return text


class ExplainableXGB:
    """Train XGBoost as an additive main-effect plus pair-interaction model.

    Unlike the earlier wrapper, this estimator does not train an unconstrained
    model and then reject trees. It constructs the model in restricted stages:

    - main-effect stage: depth-1 trees only;
    - interaction stage: continued boosters, each constrained to one feature
      pair using XGBoost's native ``interaction_constraints``.

    Prediction and explanation are both computed from the resulting staged
    trees, so local explanations are the exact score decomposition.
    """

    def __init__(
        self,
        *,
        max_main_effects: int = 30,
        max_interactions: int = 10,
        max_depth_main: int = 1,
        max_depth_interaction: int = 2,
        top_k: int = 5,
        shape_smoothing_strength: str = "moderate",
        lambda_complexity: float = 0.0,
        xgb_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.max_main_effects = max_main_effects
        self.max_interactions = max_interactions
        self.max_depth_main = max_depth_main
        self.max_depth_interaction = max_depth_interaction
        self.top_k = top_k
        self.shape_smoothing_strength = shape_smoothing_strength
        self.lambda_complexity = lambda_complexity
        self.xgb_params = dict(xgb_params or {})

        self.booster_: Any = None
        self.terms_: List[TreeTerm] = []
        self.rejected_terms_: List[TreeTerm] = []
        self.feature_metadata_: Dict[str, Dict[str, Any]] = {}
        self.domain_constraints_: Dict[str, Any] = {}
        self.feature_names_: List[str] = []
        self.original_feature_names_: List[str] = []
        self._kept_feature_indices: Optional[List[int]] = None
        self.base_score_: float = 0.0
        self._term_importance: Dict[int, float] = {}
        self._training_matrix: Optional[np.ndarray] = None
        self._selected_interactions: List[Tuple[str, str]] = []
        self._stage_plan: List[Tuple[str, Optional[Tuple[str, str]], int]] = []

    def fit(
        self,
        X: Any,
        y: Any,
        feature_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        domain_constraints: Optional[Mapping[str, Any]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "ExplainableXGB":
        self.feature_metadata_ = {
            str(k): dict(v) for k, v in (feature_metadata or {}).items()
        }
        self.domain_constraints_ = dict(domain_constraints or {})
        matrix, names = self._as_matrix(X, fit=True, feature_names=feature_names)
        matrix, names = self._filter_excluded_features(matrix, names)
        self.feature_names_ = names
        self.original_feature_names_ = list(feature_names or names)
        self._training_matrix = matrix
        y_array = np.asarray(y)
        dtrain = DMatrix(matrix, label=y_array, feature_names=self.feature_names_)

        total_rounds = int(self.xgb_params.get("n_estimators", 120))
        main_rounds = max(1, min(self.max_main_effects, total_rounds))
        interaction_rounds = max(0, total_rounds - main_rounds)
        base_params = self._base_params()

        main_params = dict(base_params)
        main_params["max_depth"] = self.max_depth_main
        self.booster_ = train(
            main_params,
            dtrain,
            num_boost_round=main_rounds,
            verbose_eval=False,
        )
        self._stage_plan = [("main", None, main_rounds)]

        self._selected_interactions = self._select_interaction_pairs(y_array)
        if self._selected_interactions and interaction_rounds:
            rounds_left = interaction_rounds
            for idx, pair in enumerate(self._selected_interactions):
                stages_left = len(self._selected_interactions) - idx
                rounds = max(1, math.ceil(rounds_left / stages_left))
                rounds = min(rounds, rounds_left)
                pair_params = dict(base_params)
                pair_params["max_depth"] = self.max_depth_interaction
                pair_params["interaction_constraints"] = [list(pair)]
                self.booster_ = train(
                    pair_params,
                    dtrain,
                    num_boost_round=rounds,
                    xgb_model=self.booster_,
                    verbose_eval=False,
                )
                self._stage_plan.append(("interaction", pair, rounds))
                rounds_left -= rounds
                if rounds_left <= 0:
                    break

        self.base_score_ = self._extract_base_score(self.booster_)
        self._build_terms_from_booster(matrix)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._is_regression_objective():
            raise ValueError("predict_proba is not available for regression objectives.")
        raw_score, _ = self._predict_margin_and_contribs(X)
        positive = sigmoid(raw_score)
        return np.vstack((1.0 - positive, positive)).T

    def predict(self, X: Any) -> np.ndarray:
        if self._is_regression_objective():
            raw_score, _ = self._predict_margin_and_contribs(X)
            return raw_score
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

    def explain_global(self) -> Dict[str, Any]:
        self._check_fitted()
        main_effects = self._aggregated_global_main_effects()
        interactions = self._aggregated_global_interactions()
        main_effects.sort(key=lambda item: item["importance"], reverse=True)
        interactions.sort(key=lambda item: item["importance"], reverse=True)
        explanation = {
            "base_score": float(self.base_score_),
            "main_effects": main_effects,
            "interactions": interactions,
            "extracted_rules": self.extract_rules(),
            "selected_interactions": [
                list(pair) for pair in self._selected_interactions
            ],
        }
        explanation["global_summary"] = generate_global_summary(explanation)
        explanation["monotonicity_report"] = self._monotonicity_report()
        return explanation

    def explain_local(self, X_row: Any) -> Dict[str, Any]:
        raw_score, keys, aggregated = self._aggregated_contribs(X_row)
        row = self._as_matrix(X_row, fit=False)[0][0]
        score = float(raw_score[0])
        positive_items = []
        negative_items = []
        for idx, key in enumerate(keys):
            contribution = float(aggregated[0, idx])
            item = self._aggregated_local_item(key, row, contribution)
            if contribution >= 0:
                positive_items.append(item)
            else:
                negative_items.append(item)
        positive_items.sort(
            key=lambda item: abs(item["total_contribution"]), reverse=True
        )
        negative_items.sort(
            key=lambda item: abs(item["total_contribution"]), reverse=True
        )
        ranked = sorted(
            positive_items + negative_items,
            key=lambda item: abs(item["total_contribution"]),
            reverse=True,
        )
        explanation = {
            "predicted_probability": float(sigmoid(np.asarray([score]))[0]),
            "raw_score": score,
            "base_score": float(self.base_score_),
            "positive_contributions": positive_items,
            "negative_contributions": negative_items,
            "top_reasons": [item["explanation"] for item in ranked[: self.top_k]],
            "counterfactual_hints": [],
        }
        explanation["summary"] = generate_local_summary(explanation)
        explanation["local_summary"] = explanation["summary"]
        explanation["exact_decomposition"] = True
        return explanation

    def explain_batch(self, X: Any) -> List[Dict[str, Any]]:
        matrix, _ = self._as_matrix(X, fit=False)
        return [self.explain_local(matrix[i : i + 1]) for i in range(matrix.shape[0])]

    def _aggregated_contribs(
        self, X: Any
    ) -> Tuple[np.ndarray, List[Tuple[str, Tuple[str, ...]]], np.ndarray]:
        raw_score, internal = self._predict_margin_and_contribs(X)
        key_to_values: Dict[Tuple[str, Tuple[str, ...]], np.ndarray] = {}
        for idx, term in enumerate(self.terms_):
            key = self._term_key(term)
            if key is None:
                continue
            key_to_values.setdefault(key, np.zeros(internal.shape[0]))
            key_to_values[key] += internal[:, idx]
        keys = list(key_to_values)
        if not keys:
            return raw_score, [], np.zeros((internal.shape[0], 0))
        aggregated = np.vstack([key_to_values[key] for key in keys]).T
        return raw_score, keys, aggregated

    def _filter_excluded_features(
        self, matrix: np.ndarray, names: Sequence[str]
    ) -> Tuple[np.ndarray, List[str]]:
        excluded = set(self.domain_constraints_.get("excluded_features", []))
        if not excluded:
            self._kept_feature_indices = list(range(len(names)))
            return matrix, list(names)
        kept = []
        kept_names = []
        for idx, name in enumerate(names):
            label = feature_label(name, self.feature_metadata_)
            if name in excluded or label in excluded:
                continue
            kept.append(idx)
            kept_names.append(name)
        self._kept_feature_indices = kept
        return matrix[:, kept], kept_names

    def _apply_feature_filter(self, matrix: np.ndarray) -> np.ndarray:
        if self._kept_feature_indices is None:
            return matrix
        if matrix.shape[1] == len(self._kept_feature_indices):
            return matrix
        return matrix[:, self._kept_feature_indices]

    def _term_key(
        self, term: TreeTerm
    ) -> Optional[Tuple[str, Tuple[str, ...]]]:
        if term.term_type == "main" and len(term.features) == 1:
            return ("main", (term.features[0],))
        if term.term_type == "interaction" and len(term.features) == 2:
            return ("interaction", normalise_pair(term.features))
        if term.term_type == "bias":
            return ("bias", ("constant_term",))
        return None

    def _grouped_main_terms(self) -> Dict[str, List[TreeTerm]]:
        groups: Dict[str, List[TreeTerm]] = {}
        for term in self.terms_:
            if term.term_type == "main" and len(term.features) == 1:
                groups.setdefault(term.features[0], []).append(term)
        return groups

    def _grouped_interaction_terms(self) -> Dict[Tuple[str, str], List[TreeTerm]]:
        groups: Dict[Tuple[str, str], List[TreeTerm]] = {}
        for term in self.terms_:
            if term.term_type == "interaction" and len(term.features) == 2:
                groups.setdefault(normalise_pair(term.features), []).append(term)
        return groups

    def _aggregated_global_main_effects(self) -> List[Dict[str, Any]]:
        if self._training_matrix is None:
            return []
        _, keys, aggregated = self._aggregated_contribs(self._training_matrix)
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        grouped = self._grouped_main_terms()
        effects = []
        for feature, terms in grouped.items():
            key = ("main", (feature,))
            if key not in key_to_idx:
                continue
            curve = self._feature_effect_curve(feature)
            shape_analysis = analyse_effect_shape(
                curve, shape_smoothing_strength=self.shape_smoothing_strength
            )
            direction = str(shape_analysis["direction"])
            shape = str(shape_analysis["shape"])
            thresholds = self._threshold_summary_for_features((feature,))
            label = feature_label(feature, self.feature_metadata_)
            importance = float(np.mean(np.abs(aggregated[:, key_to_idx[key]])))
            effect = AggregatedFeatureEffect(
                feature=feature,
                label=label,
                total_importance=importance,
                total_contribution=None,
                direction=direction,
                shape=shape,
                shape_confidence=float(shape_analysis["shape_confidence"]),
                monotonic_confidence=float(shape_analysis["monotonic_confidence"]),
                effect_amplitude=float(shape_analysis["effect_amplitude"]),
                num_sign_changes=int(shape_analysis["num_sign_changes"]),
                thresholds=thresholds,
                n_terms=len(terms),
                summary=self._effect_summary(label, direction, shape, thresholds),
            )
            effects.append(
                {
                    "feature": effect.feature,
                    "label": effect.label,
                    "importance": effect.total_importance,
                    "total_importance": effect.total_importance,
                    "total_contribution": effect.total_contribution,
                    "direction": effect.direction,
                    "shape": effect.shape,
                    "shape_confidence": effect.shape_confidence,
                    "monotonic_confidence": effect.monotonic_confidence,
                    "effect_amplitude": effect.effect_amplitude,
                    "num_sign_changes": effect.num_sign_changes,
                    "smoothed_curve": shape_analysis["smoothed_curve"],
                    "thresholds": effect.thresholds,
                    "n_terms": effect.n_terms,
                    "summary": effect.summary,
                }
            )
        return effects

    def _aggregated_global_interactions(self) -> List[Dict[str, Any]]:
        if self._training_matrix is None:
            return []
        _, keys, aggregated = self._aggregated_contribs(self._training_matrix)
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        interactions = []
        for pair, terms in self._grouped_interaction_terms().items():
            key = ("interaction", pair)
            if key not in key_to_idx:
                continue
            labels = [feature_label(f, self.feature_metadata_) for f in pair]
            importance = float(np.mean(np.abs(aggregated[:, key_to_idx[key]])))
            interactions.append(
                {
                    "features": list(pair),
                    "labels": labels,
                    "importance": importance,
                    "total_importance": importance,
                    "n_terms": len(terms),
                    "thresholds": self._threshold_summary_for_features(pair),
                    "summary": (
                        f"{' + '.join(labels)} aparece como interaccion "
                        "aprendida por el modelo."
                    ),
                }
            )
        return interactions

    def _aggregated_local_item(
        self,
        key: Tuple[str, Tuple[str, ...]],
        row: np.ndarray,
        contribution: float,
    ) -> Dict[str, Any]:
        term_type, features = key
        if term_type == "bias":
            label = "Ajuste constante"
            value: Any = None
            term_name = "constant_term"
        elif len(features) == 1:
            feature = features[0]
            label = feature_label(feature, self.feature_metadata_)
            value = self._clean_value(row[self._feature_index(feature)])
            term_name = feature
        else:
            labels = [feature_label(f, self.feature_metadata_) for f in features]
            label = " + ".join(labels)
            value = {
                feature: self._clean_value(row[self._feature_index(feature)])
                for feature in features
            }
            term_name = " + ".join(features)
        sign = "aumenta" if contribution >= 0 else "reduce"
        return {
            "term": term_name,
            "label": label,
            "term_type": term_type,
            "value": value,
            "total_contribution": contribution,
            "n_internal_terms": self._n_terms_for_key(key),
            "explanation": f"{label} {sign} la prediccion positiva.",
        }

    def _n_terms_for_key(self, key: Tuple[str, Tuple[str, ...]]) -> int:
        term_type, features = key
        if term_type == "main":
            return len(self._grouped_main_terms().get(features[0], []))
        if term_type == "interaction":
            return len(self._grouped_interaction_terms().get(features, []))
        return 1

    def _feature_effect_curve(
        self, feature: str, n_grid: int = 50
    ) -> Dict[str, List[float]]:
        if self._training_matrix is None:
            return {"values": [], "contributions": []}
        return self._feature_effect_curve_from_matrix(
            feature, self._training_matrix, n_grid
        )

    def compute_feature_effect_curve(
        self,
        feature: str,
        X_reference: Optional[Any] = None,
        grid_size: int = 50,
    ) -> Dict[str, Any]:
        matrix = self._training_matrix
        if X_reference is not None:
            matrix = self._as_matrix(X_reference, fit=False)[0]
        if matrix is None:
            raise ValueError("A fitted model or X_reference is required.")
        feature_id = self._resolve_feature(feature)
        curve = self._feature_effect_curve_from_matrix(feature_id, matrix, grid_size)
        contributions = np.asarray(curve["contributions"], dtype=float)
        centered = contributions - float(np.mean(contributions))
        shape_analysis = analyse_effect_shape(
            curve, shape_smoothing_strength=self.shape_smoothing_strength
        )
        direction = str(shape_analysis["direction"])
        shape = str(shape_analysis["shape"])
        thresholds = self._threshold_summary_for_features((feature_id,))
        label = feature_label(feature_id, self.feature_metadata_)
        return {
            "feature": feature_id,
            "label": label,
            "x_values": curve["values"],
            "contributions": curve["contributions"],
            "centered_contributions": [float(v) for v in centered],
            "direction": direction,
            "shape": shape,
            "shape_confidence": float(shape_analysis["shape_confidence"]),
            "monotonic_confidence": float(shape_analysis["monotonic_confidence"]),
            "effect_amplitude": float(shape_analysis["effect_amplitude"]),
            "num_sign_changes": int(shape_analysis["num_sign_changes"]),
            "smoothed_curve": shape_analysis["smoothed_curve"],
            "thresholds": thresholds,
            "saturation_points": [
                thresholds["saturation_point"]
            ]
            if thresholds["saturation_point"] is not None
            else [],
            "summary": self._effect_summary(label, direction, shape, thresholds),
        }

    def _feature_effect_curve_from_matrix(
        self, feature: str, matrix: np.ndarray, n_grid: int
    ) -> Dict[str, List[float]]:
        idx = self._feature_index(feature)
        col = matrix[:, idx]
        unique = np.unique(col[~np.isnan(col)])
        if unique.size == 0:
            return {"values": [], "contributions": []}
        if unique.size <= n_grid:
            grid = unique
        else:
            grid = np.quantile(col[~np.isnan(col)], np.linspace(0.0, 1.0, n_grid))
            grid = np.unique(grid)
        base = np.nanmedian(matrix, axis=0)
        rows = np.repeat(base.reshape(1, -1), grid.size, axis=0)
        rows[:, idx] = grid
        contributions = np.zeros(grid.size)
        for term in self._grouped_main_terms().get(feature, []):
            contributions += self._eval_tree_batch(term.raw_tree, rows)
        return {
            "values": [float(v) for v in grid],
            "contributions": [float(v) for v in contributions],
        }

    def _thresholds_for_features(self, features: Sequence[str]) -> List[float]:
        thresholds = []
        feature_set = set(features)
        for term in self.terms_:
            if not set(term.features).intersection(feature_set):
                continue
            thresholds.extend(self._thresholds_in_tree(term.raw_tree, feature_set))
        unique = sorted({round(float(value), 6) for value in thresholds})
        return unique[:8]

    def _threshold_summary_for_features(
        self, features: Sequence[str]
    ) -> Dict[str, Any]:
        all_points = self._thresholds_for_features(features)
        return {
            "first_threshold": all_points[0] if all_points else None,
            "major_threshold": all_points[len(all_points) // 2] if all_points else None,
            "saturation_point": all_points[-1] if len(all_points) >= 3 else None,
            "all_split_points": all_points,
        }

    def _thresholds_in_tree(
        self, node: Mapping[str, Any], features: set
    ) -> List[float]:
        if "leaf" in node:
            return []
        values = []
        if str(node["split"]) in features:
            values.append(float(node["split_condition"]))
        for child in node.get("children", []):
            values.extend(self._thresholds_in_tree(child, features))
        return values

    def _effect_summary(
        self, label: str, direction: str, shape: str, thresholds: Mapping[str, Any]
    ) -> str:
        direction_text = {
            "increases_prediction": "aumenta la prediccion positiva",
            "decreases_prediction": "reduce la prediccion positiva",
            "u_shape": "muestra un patron en U",
            "inverted_u_shape": "muestra un patron en U invertida",
            "flat": "tiene un efecto casi plano",
        }.get(direction, "tiene un efecto no monotono")
        first = thresholds.get("first_threshold")
        saturation = thresholds.get("saturation_point")
        if first is not None and shape in {"threshold", "saturation"}:
            text = f"{label} {direction_text} principalmente a partir de {first:g}"
            if saturation is not None:
                text += f" y se estabiliza alrededor de {saturation:g}"
            return text + "."
        return f"{label} {direction_text} con forma {shape}."

    def extract_rules(self) -> List[Dict[str, Any]]:
        if self._training_matrix is None:
            return []
        rules = self._main_effect_rules() + self._interaction_rules()
        rules.sort(key=lambda item: item["importance"], reverse=True)
        return rules[: self.top_k]

    def _main_effect_rules(self) -> List[Dict[str, Any]]:
        rules = []
        assert self._training_matrix is not None
        for effect in self._aggregated_global_main_effects():
            thresholds = effect.get("thresholds", {})
            threshold = thresholds.get("first_threshold")
            if threshold is None:
                continue
            feature = effect["feature"]
            idx = self._feature_index(feature)
            mask = self._training_matrix[:, idx] > threshold
            if not np.any(mask):
                continue
            _, keys, aggregated = self._aggregated_contribs(self._training_matrix)
            key_to_idx = {key: i for i, key in enumerate(keys)}
            key = ("main", (feature,))
            if key not in key_to_idx:
                continue
            effect_value = float(np.mean(aggregated[mask, key_to_idx[key]]))
            rules.append(
                {
                    "rule": f"{effect['label']} > {threshold:g}",
                    "effect": effect_value,
                    "support": int(np.sum(mask)),
                    "direction": "increases_prediction"
                    if effect_value >= 0
                    else "decreases_prediction",
                    "importance": abs(effect_value) * math.log1p(int(np.sum(mask))),
                }
            )
        return rules

    def _resolve_feature(self, feature: str) -> str:
        if feature in self.feature_names_:
            return feature
        for feature_id in self.feature_names_:
            if feature_label(feature_id, self.feature_metadata_) == feature:
                return feature_id
        raise ValueError(f"Unknown feature: {feature}")

    def _interaction_rules(self) -> List[Dict[str, Any]]:
        rules = []
        assert self._training_matrix is not None
        _, keys, aggregated = self._aggregated_contribs(self._training_matrix)
        key_to_idx = {key: i for i, key in enumerate(keys)}
        for interaction in self._aggregated_global_interactions():
            features = tuple(interaction["features"])
            key = ("interaction", features)
            if key not in key_to_idx:
                continue
            parts = []
            mask = np.ones(self._training_matrix.shape[0], dtype=bool)
            for feature in features:
                idx = self._feature_index(feature)
                threshold = float(np.nanmedian(self._training_matrix[:, idx]))
                parts.append(
                    f"{feature_label(feature, self.feature_metadata_)} > {threshold:g}"
                )
                mask &= self._training_matrix[:, idx] > threshold
            if not np.any(mask):
                continue
            effect_value = float(np.mean(aggregated[mask, key_to_idx[key]]))
            rules.append(
                {
                    "rule": " AND ".join(parts),
                    "effect": effect_value,
                    "support": int(np.sum(mask)),
                    "direction": "increases_prediction"
                    if effect_value >= 0
                    else "decreases_prediction",
                    "importance": abs(effect_value) * math.log1p(int(np.sum(mask))),
                }
            )
        return rules

    def _monotonicity_report(self) -> Dict[str, Dict[str, Any]]:
        report: Dict[str, Dict[str, Any]] = {}
        constraints = self.domain_constraints_.get("monotonic", {})
        for feature_name, constraint in constraints.items():
            try:
                feature = self._resolve_feature(str(feature_name))
            except ValueError:
                continue
            curve = self.compute_feature_effect_curve(feature)
            values = np.asarray(curve["contributions"], dtype=float)
            if values.size < 2:
                violations = 0
                rate = 0.0
            else:
                diffs = np.diff(values)
                tolerance = 0.05 * max(1e-12, float(np.max(np.abs(values))))
                if constraint == "increasing":
                    violations = int(np.sum(diffs < -tolerance))
                elif constraint == "decreasing":
                    violations = int(np.sum(diffs > tolerance))
                else:
                    violations = 0
                rate = violations / max(1, len(diffs))
            report[feature] = {
                "constraint": constraint,
                "violations": violations,
                "violation_rate": rate,
            }
        return report

    def _base_params(self) -> Dict[str, Any]:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 2026,
            "nthread": 1,
        }
        for key, value in self.xgb_params.items():
            if key not in {"n_estimators", "max_depth", "n_jobs"}:
                params[key] = value
        monotone = self._monotone_constraints()
        if monotone:
            params["monotone_constraints"] = monotone
        return params

    def _is_regression_objective(self) -> bool:
        objective = str(self.xgb_params.get("objective", "binary:logistic"))
        return objective.startswith("reg:")

    def _monotone_constraints(self) -> Tuple[int, ...]:
        constraints = dict(self.domain_constraints_.get("monotonic", {}))
        for name, meta in self.feature_metadata_.items():
            if "monotonic" in meta:
                constraints.setdefault(name, meta["monotonic"])
        mapping = {"increasing": 1, "decreasing": -1, "none": 0, None: 0}
        return tuple(
            mapping.get(constraints.get(name), 0) for name in self.feature_names_
        )

    def _select_interaction_pairs(self, y: np.ndarray) -> List[Tuple[str, str]]:
        allowed = self.domain_constraints_.get("allowed_interactions")
        forbidden = {
            normalise_pair(item)
            for item in self.domain_constraints_.get("forbidden_interactions", [])
        }
        if allowed:
            candidates = [
                normalise_pair(pair)
                for pair in allowed
                if all(str(feature) in self.feature_names_ for feature in pair)
            ]
            return [pair for pair in candidates if pair not in forbidden][
                : self.max_interactions
            ]

        if self.max_interactions <= 0:
            return []

        ranked = self._rank_features_for_interactions(y)
        ranked = ranked[: max(2, min(len(ranked), self.max_main_effects))]
        candidates = []
        for pair in itertools.combinations(ranked, 2):
            normalised = normalise_pair(pair)
            if normalised not in forbidden:
                candidates.append(normalised)
        if not candidates:
            return []

        gains = self._exploratory_pair_gains(candidates, y)
        candidates.sort(key=lambda pair: gains.get(pair, 0.0), reverse=True)

        if self.lambda_complexity <= 0.0:
            return candidates[: self.max_interactions]

        selected: List[Tuple[str, str]] = []
        for pair in candidates:
            if len(selected) >= self.max_interactions:
                break
            if gains.get(pair, 0.0) <= self.lambda_complexity:
                break
            selected.append(pair)
        return selected

    def _exploratory_pair_gains(
        self, candidates: Sequence[Tuple[str, str]], y: np.ndarray
    ) -> Dict[Tuple[str, str], float]:
        """Rank candidate interaction pairs by real training-loss-reduction gain.

        Grows a short-lived depth-2 exploratory booster on top of the
        main-effect stage, restricted (via ``interaction_constraints``) to the
        candidate feature set, so every split realises one of the candidate
        pairs. The exploratory booster is discarded afterwards and never
        merged into ``self.booster_`` or ``self._stage_plan``; only the
        per-pair split gain it reveals is kept. This replaces a static
        variance-of-product proxy with an actual loss-reduction statistic --
        comparable in spirit to the RSS-gain criterion GA2M/EBM's FAST
        algorithm uses for interaction detection -- and gives
        ``lambda_complexity`` a real quantity (gain, in loss units) to
        threshold against.
        """
        assert self.booster_ is not None and self._training_matrix is not None
        candidate_features = sorted({f for pair in candidates for f in pair})
        if len(candidate_features) < 2:
            return {}
        explore_params = dict(self._base_params())
        explore_params["max_depth"] = self.max_depth_interaction
        explore_params["interaction_constraints"] = [candidate_features]
        explore_params["subsample"] = 1.0
        explore_params["colsample_bytree"] = 1.0
        dtrain = DMatrix(
            self._training_matrix, label=np.asarray(y), feature_names=self.feature_names_
        )
        explore_rounds = min(60, max(20, 2 * len(candidates) // 3))
        explore_booster = train(
            explore_params,
            dtrain,
            num_boost_round=explore_rounds,
            xgb_model=self.booster_,
            verbose_eval=False,
        )
        gains: Dict[Tuple[str, str], float] = {pair: 0.0 for pair in candidates}
        candidate_set = set(candidates)
        for tree_json in explore_booster.get_dump(dump_format="json", with_stats=True):
            self._accumulate_pair_gain(json.loads(tree_json), gains, candidate_set)
        return gains

    def _accumulate_pair_gain(
        self,
        node: Mapping[str, Any],
        gains: Dict[Tuple[str, str], float],
        candidate_set: set,
        ancestors: Tuple[str, ...] = (),
    ) -> None:
        if "leaf" in node:
            return
        feature = str(node["split"])
        gain = float(node.get("gain", 0.0))
        for ancestor in ancestors:
            pair = normalise_pair((ancestor, feature))
            if pair in candidate_set:
                gains[pair] = gains.get(pair, 0.0) + gain
        for child in node.get("children", []):
            self._accumulate_pair_gain(
                child, gains, candidate_set, ancestors + (feature,)
            )

    def _rank_features_for_interactions(self, y: np.ndarray) -> List[str]:
        term_scores: Dict[str, float] = {}
        if self.booster_ is not None:
            for tree in self.booster_.get_dump(dump_format="json"):
                parsed = json.loads(tree)
                features = tuple(sorted(self._tree_features(parsed)))
                if len(features) == 1:
                    term_scores[features[0]] = (
                        term_scores.get(features[0], 0.0)
                        + self._mean_abs_leaf(parsed)
                    )
        if len(term_scores) >= 2:
            return [
                feature
                for feature, _ in sorted(
                    term_scores.items(), key=lambda item: item[1], reverse=True
                )
            ]

        assert self._training_matrix is not None
        y_centered = y - np.mean(y)
        scores = []
        for idx, feature in enumerate(self.feature_names_):
            col = self._training_matrix[:, idx]
            col_centered = col - np.nanmean(col)
            denom = np.linalg.norm(col_centered) * np.linalg.norm(y_centered)
            score = 0.0
            if denom != 0:
                score = abs(float(np.dot(col_centered, y_centered) / denom))
            scores.append((feature, score))
        return [
            feature
            for feature, _ in sorted(scores, key=lambda item: item[1], reverse=True)
        ]

    def _build_terms_from_booster(self, matrix: np.ndarray) -> None:
        raw_trees = [
            json.loads(tree) for tree in self.booster_.get_dump(dump_format="json")
        ]
        terms: List[TreeTerm] = []
        invalid: List[TreeTerm] = []
        stage_by_tree = self._stage_by_tree()

        for tree_id, tree in enumerate(raw_trees):
            expected_type, expected_pair = stage_by_tree[tree_id]
            features = tuple(sorted(self._tree_features(tree)))
            term_type = self._term_type_for_stage(expected_type, features)
            term = TreeTerm(
                tree_id=tree_id,
                term_type=term_type,
                features=features,
                raw_tree=tree,
                weight=self._mean_abs_leaf(tree),
                direction=self._direction_from_constraints(features),
                domain_label=None,
            )
            if self._violates_stage(term, expected_type, expected_pair):
                invalid.append(term)
            else:
                terms.append(term)
                values = self._eval_tree_batch(term.raw_tree, matrix)
                self._term_importance[term.tree_id] = float(np.mean(np.abs(values)))
        self.terms_ = terms
        self.rejected_terms_ = invalid

    def _stage_by_tree(self) -> List[Tuple[str, Optional[Tuple[str, str]]]]:
        stages: List[Tuple[str, Optional[Tuple[str, str]]]] = []
        for stage_type, pair, rounds in self._stage_plan:
            stages.extend([(stage_type, pair)] * rounds)
        return stages

    def _term_type_for_stage(self, expected_type: str, features: Sequence[str]) -> str:
        if not features:
            return "bias"
        if len(features) == 1:
            return "main"
        if expected_type == "interaction" and len(features) == 2:
            return "interaction"
        return "invalid"

    def _violates_stage(
        self,
        term: TreeTerm,
        expected_type: str,
        expected_pair: Optional[Tuple[str, str]],
    ) -> bool:
        if term.term_type == "invalid":
            return True
        if expected_type == "main":
            return (
                len(term.features) > 1
                or self._tree_depth(term.raw_tree) > self.max_depth_main
            )
        if self._tree_depth(term.raw_tree) > self.max_depth_interaction:
            return True
        if expected_pair is None or len(term.features) < 2:
            return False
        return not set(term.features).issubset(set(expected_pair))

    def _predict_margin_and_contribs(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        matrix, _ = self._as_matrix(X, fit=False)
        contributions = np.zeros((matrix.shape[0], len(self.terms_)))
        for idx, term in enumerate(self.terms_):
            contributions[:, idx] = self._eval_tree_batch(term.raw_tree, matrix)
        return self.base_score_ + np.sum(contributions, axis=1), contributions

    def _eval_tree_batch(
        self, tree: Mapping[str, Any], matrix: np.ndarray
    ) -> np.ndarray:
        result = np.empty(matrix.shape[0], dtype=float)
        self._eval_tree_vectorized(tree, matrix, np.arange(matrix.shape[0]), result)
        return result

    def _eval_tree_vectorized(
        self,
        node: Mapping[str, Any],
        matrix: np.ndarray,
        rows: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """Evaluate one dumped tree for a batch of rows at once.

        Same node-by-node semantics as evaluating row by row (missing values
        follow ``node["missing"]``, ties go to ``node["no"]``), but each
        recursive call partitions the current row *index* array by boolean
        mask instead of Python-looping over rows, so a tree with depth d costs
        O(d) numpy operations over the batch rather than O(n_rows) Python
        function calls per tree.
        """
        if rows.size == 0:
            return
        if "leaf" in node:
            out[rows] = float(node["leaf"])
            return
        idx = self._feature_index(str(node["split"]))
        values = matrix[rows, idx]
        children = {child["nodeid"]: child for child in node["children"]}
        is_nan = np.isnan(values)
        goes_yes = (~is_nan) & (values < float(node["split_condition"]))
        if node["missing"] == node["yes"]:
            goes_yes = goes_yes | is_nan
        self._eval_tree_vectorized(children[node["yes"]], matrix, rows[goes_yes], out)
        self._eval_tree_vectorized(children[node["no"]], matrix, rows[~goes_yes], out)

    def _as_matrix(
        self,
        X: Any,
        *,
        fit: bool,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if sparse.issparse(X):
            matrix = X.toarray()
        elif hasattr(X, "to_numpy"):
            matrix = X.to_numpy(dtype=np.float64)
        else:
            matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("X must be a 2-dimensional table or a single row.")
        if fit:
            names = list(feature_names or [f"f{i}" for i in range(matrix.shape[1])])
            return matrix, [f"f{i}" for i in range(len(names))]
        matrix = self._apply_feature_filter(matrix)
        if matrix.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Expected {len(self.feature_names_)} features, got {matrix.shape[1]}."
            )
        return matrix, self.feature_names_

    def _feature_index(self, feature: str) -> int:
        if feature.startswith("f") and feature[1:].isdigit():
            return int(feature[1:])
        return self.feature_names_.index(feature)

    def _tree_features(self, node: Mapping[str, Any]) -> set:
        if "leaf" in node:
            return set()
        features = {str(node["split"])}
        for child in node.get("children", []):
            features.update(self._tree_features(child))
        return features

    def _tree_depth(self, node: Mapping[str, Any]) -> int:
        if "leaf" in node:
            return 0
        return 1 + max(self._tree_depth(child) for child in node.get("children", []))

    def _leaf_values(self, node: Mapping[str, Any]) -> List[float]:
        if "leaf" in node:
            return [float(node["leaf"])]
        values: List[float] = []
        for child in node.get("children", []):
            values.extend(self._leaf_values(child))
        return values

    def _mean_abs_leaf(self, node: Mapping[str, Any]) -> float:
        leaves = self._leaf_values(node)
        return float(np.mean(np.abs(leaves))) if leaves else 0.0

    def _direction_from_constraints(self, features: Sequence[str]) -> str:
        if len(features) != 1:
            return "mixed"
        monotone = self.domain_constraints_.get("monotonic", {}).get(features[0])
        if monotone == "increasing":
            return "prediction_increasing"
        if monotone == "decreasing":
            return "prediction_decreasing"
        return "mixed"

    def _public_direction(self, direction: str) -> str:
        return {
            "prediction_increasing": "increases_prediction",
            "prediction_decreasing": "decreases_prediction",
        }.get(direction, "mixed")

    def _extract_base_score(self, booster: Any) -> float:
        config = json.loads(booster.save_config())
        value = config["learner"]["learner_model_param"].get("base_score", 0.5)
        if isinstance(value, str) and value.startswith("["):
            value = value.strip("[]")
        if self._is_regression_objective():
            return float(value)
        probability = min(max(float(value), 1e-12), 1.0 - 1e-12)
        return float(math.log(probability / (1.0 - probability)))

    def _clean_value(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _check_fitted(self) -> None:
        if self.booster_ is None:
            raise ValueError("ExplainableXGB is not fitted.")


ExplainableXGBClassifier = ExplainableXGB


class ExplainableXGBMulticlass:
    """One-vs-rest ensemble of :class:`ExplainableXGB` binary models for multiclass tasks.

    ``ExplainableXGB`` is intrinsically binary: a single booster, a single
    ``base_score_``, and a sigmoid link for probabilities. Rather than
    rewriting the staged main-effect/interaction boosting algorithm to be
    natively multiclass, this wrapper adds multiclass support the same way
    scikit-learn adds it to many binary-only estimators: a genuine
    One-vs-Rest (OvR) ensemble.

    For ``K`` classes it trains ``K`` independent ``ExplainableXGB`` models,
    one per class, each fit on a binary target ``y_k = (y == k)`` with a
    ``binary:logistic`` objective. Each of the ``K`` sub-models keeps its own
    exact additive decomposition (main effects + pairwise interactions), so
    the "exact decomposition" property of ``ExplainableXGB`` is preserved
    *per class* -- there are simply ``K`` independent sets of terms instead
    of one.

    Predicted class probabilities are obtained by taking a softmax over the
    ``K`` raw margins (the pre-sigmoid scores) returned by each class's
    sub-model, so the ``K`` probabilities sum to 1 per row. This is the
    standard way to turn ``K`` independent one-vs-rest margins into a proper
    probability distribution (equivalent to what LogisticRegression /
    OneVsRestClassifier-style softmax-of-margins does), and is preferred over
    naively re-normalising the ``K`` independent sigmoids, which does not
    have as clean a probabilistic interpretation when the K one-vs-rest
    problems are fit independently.

    This class does not change the core staged-boosting algorithm, the
    interaction-selection heuristic, or the complexity control of
    ``ExplainableXGB`` in any way -- it is purely an OvR composition layer
    needed to be able to evaluate the estimator on multiclass datasets.
    """

    def __init__(
        self,
        *,
        n_classes: Optional[int] = None,
        max_main_effects: int = 30,
        max_interactions: int = 10,
        max_depth_main: int = 1,
        max_depth_interaction: int = 2,
        top_k: int = 5,
        shape_smoothing_strength: str = "moderate",
        lambda_complexity: float = 0.0,
        xgb_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # ``n_classes`` is accepted for API convenience / logging but the
        # authoritative source of truth is always ``np.unique(y)`` at fit
        # time -- see ``fit()``.
        self.n_classes = n_classes
        self._model_kwargs: Dict[str, Any] = dict(
            max_main_effects=max_main_effects,
            max_interactions=max_interactions,
            max_depth_main=max_depth_main,
            max_depth_interaction=max_depth_interaction,
            top_k=top_k,
            shape_smoothing_strength=shape_smoothing_strength,
            lambda_complexity=lambda_complexity,
            xgb_params=xgb_params,
        )
        self.classifiers_: List[ExplainableXGB] = []
        self.classes_: np.ndarray = np.array([])
        self.feature_names_: List[str] = []

    def fit(
        self,
        X: Any,
        y: Any,
        feature_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        domain_constraints: Optional[Mapping[str, Any]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "ExplainableXGBMulticlass":
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.n_classes = len(self.classes_)
        self.classifiers_ = []
        for class_value in self.classes_:
            y_binary = (y_array == class_value).astype(int)
            clf = ExplainableXGB(**self._model_kwargs)
            clf.fit(
                X,
                y_binary,
                feature_metadata=feature_metadata,
                domain_constraints=domain_constraints,
                feature_names=feature_names,
            )
            self.classifiers_.append(clf)
        self.feature_names_ = (
            list(self.classifiers_[0].feature_names_) if self.classifiers_ else []
        )
        return self

    def _raw_margins(self, X: Any) -> np.ndarray:
        """Return the (n_rows, n_classes) matrix of pre-sigmoid raw scores."""
        margins = [clf._predict_margin_and_contribs(X)[0] for clf in self.classifiers_]
        return np.vstack(margins).T

    def predict_proba(self, X: Any) -> np.ndarray:
        margins = self._raw_margins(X)
        shifted = margins - np.max(margins, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def explain_global(self) -> Dict[str, Any]:
        """Aggregate per-class explanations into a single global view.

        Each class keeps a fully independent decomposition (available under
        ``per_class`` below); the top-level ``main_effects``/``interactions``
        lists merge those per-class importances by averaging each feature's
        (or pair's) importance across the classes in which it was selected.
        """
        per_class: List[Dict[str, Any]] = []
        for class_value, clf in zip(self.classes_, self.classifiers_):
            exp = clf.explain_global()
            exp = dict(exp)
            exp["class"] = self._clean_class(class_value)
            per_class.append(exp)

        main_importance: Dict[str, List[float]] = {}
        main_label: Dict[str, str] = {}
        for exp in per_class:
            for effect in exp["main_effects"]:
                main_importance.setdefault(effect["feature"], []).append(effect["importance"])
                main_label.setdefault(effect["feature"], effect["label"])
        merged_main = [
            {
                "feature": feature,
                "label": main_label.get(feature, feature),
                "importance": float(np.mean(values)),
                "n_classes_present": len(values),
            }
            for feature, values in main_importance.items()
        ]
        merged_main.sort(key=lambda item: item["importance"], reverse=True)

        inter_importance: Dict[Tuple[str, str], List[float]] = {}
        inter_labels: Dict[Tuple[str, str], List[str]] = {}
        for exp in per_class:
            for interaction in exp["interactions"]:
                key = normalise_pair(interaction["features"])
                inter_importance.setdefault(key, []).append(interaction["importance"])
                inter_labels.setdefault(key, interaction["labels"])
        merged_interactions = [
            {
                "features": list(key),
                "labels": inter_labels.get(key, list(key)),
                "importance": float(np.mean(values)),
                "n_classes_present": len(values),
            }
            for key, values in inter_importance.items()
        ]
        merged_interactions.sort(key=lambda item: item["importance"], reverse=True)

        return {
            "n_classes": len(self.classifiers_),
            "classes": [self._clean_class(c) for c in self.classes_],
            "main_effects": merged_main,
            "interactions": merged_interactions,
            "per_class": per_class,
            "multiclass_strategy": (
                "one-vs-rest: one independent ExplainableXGB per class, "
                "predict_proba via softmax over raw margins"
            ),
        }

    def explain_local(self, X_row: Any) -> Dict[str, Any]:
        per_class = []
        for class_value, clf in zip(self.classes_, self.classifiers_):
            exp = clf.explain_local(X_row)
            exp = dict(exp)
            exp["class"] = self._clean_class(class_value)
            per_class.append(exp)
        proba = self.predict_proba(X_row)[0]
        predicted_idx = int(np.argmax(proba))
        return {
            "predicted_class": self._clean_class(self.classes_[predicted_idx]),
            "predicted_probabilities": {
                str(self._clean_class(c)): float(p) for c, p in zip(self.classes_, proba)
            },
            "per_class": per_class,
            "multiclass_strategy": (
                "one-vs-rest: one independent ExplainableXGB per class, "
                "predict_proba via softmax over raw margins"
            ),
        }

    def extract_rules(self) -> List[Dict[str, Any]]:
        """Merge the top rules extracted independently from each class model.

        Each rule string is tagged with the class it was extracted for
        (``"[class=<value>] <rule>"``), since the same feature/threshold rule
        can carry a different sign/importance for different classes. Rules
        are pooled across all classes and re-sorted by importance.
        """
        combined: List[Dict[str, Any]] = []
        for class_value, clf in zip(self.classes_, self.classifiers_):
            for rule in clf.extract_rules():
                tagged = dict(rule)
                tagged["class"] = self._clean_class(class_value)
                tagged["rule"] = f"[class={self._clean_class(class_value)}] {rule['rule']}"
                combined.append(tagged)
        combined.sort(key=lambda item: item["importance"], reverse=True)
        return combined

    @property
    def terms_(self) -> List[TreeTerm]:
        return [term for clf in self.classifiers_ for term in clf.terms_]

    @property
    def rejected_terms_(self) -> List[TreeTerm]:
        return [term for clf in self.classifiers_ for term in clf.rejected_terms_]

    @property
    def base_score_(self) -> float:
        if not self.classifiers_:
            return 0.0
        return float(np.mean([clf.base_score_ for clf in self.classifiers_]))

    @staticmethod
    def _clean_class(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value


ExplainableXGBOvR = ExplainableXGBMulticlass
