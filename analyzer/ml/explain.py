"""
analyzer/ml/explain.py — Phase 8B SHAP-Based Model Explainability

Provides:
- Global SHAP feature importance summary (across all test samples).
- Individual PR explanation (top contributing factors for a single prediction).

The explanations are derived purely from SHAP values and model output.
They are NEVER hard-coded or fabricated.

SHAP interprets the model as a black-box; explanations reflect actual
feature contributions learned by CatBoost during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy SHAP import
try:
    import shap  # type: ignore[import-untyped]
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False
    shap = None  # noqa: F841


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FeatureContribution:
    """A single feature's SHAP contribution for one prediction."""

    feature_name: str
    value: float        # The actual feature value for this sample
    shap_value: float   # SHAP contribution (positive = toward high_risk=1)
    direction: str      # "increases_risk" | "decreases_risk" | "neutral"


@dataclass
class PRExplanation:
    """SHAP explanation for a single PR prediction."""

    pr_id: Optional[Any]
    predicted_probability: float
    predicted_label: int
    label_text: str               # "HIGH RISK" or "LOW RISK"
    contributions: List[FeatureContribution]
    top_risk_factors: List[str]
    top_protective_factors: List[str]
    base_value: float             # SHAP expected value (baseline)


@dataclass
class GlobalSHAPSummary:
    """Global SHAP summary across all samples."""

    mean_abs_shap: Dict[str, float]         # feature → mean |SHAP|
    feature_importance_rank: List[str]      # sorted by mean |SHAP|
    shap_values: Optional[np.ndarray]       # shape (n_samples, n_features)
    feature_names: List[str]
    n_samples: int
    expected_value: float


# ---------------------------------------------------------------------------
# SHAP explanation functions
# ---------------------------------------------------------------------------


def _require_shap() -> None:
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is required. pip install shap")


def generate_shap_explanations(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 1000,
) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values for a CatBoost model.

    Uses TreeExplainer which is exact and efficient for tree-based models.

    Args:
        model: Trained CatBoostClassifier.
        X: Feature matrix (n_samples, n_features). Will be subsampled
           to max_samples if larger.
        feature_names: Optional feature names.
        max_samples: Maximum samples to explain (performance bound).

    Returns:
        (shap_values, expected_value) where:
            shap_values: Array of shape (n_samples, n_features), log-odds scale.
            expected_value: SHAP base value (model expected output).
    """
    _require_shap()

    # Subsample if too large
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_explain = X[idx]
        logger.info(
            f"SHAP: subsampled {len(X)} → {max_samples} samples for explanation."
        )
    else:
        X_explain = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # CatBoost with binary classification returns a single array
    if isinstance(shap_values, list):
        # Some versions return [neg_class_shap, pos_class_shap]
        shap_values = shap_values[1]

    expected_value = float(explainer.expected_value)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = float(explainer.expected_value[1])

    return shap_values, expected_value


def global_shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 1000,
) -> GlobalSHAPSummary:
    """
    Generate a global SHAP feature importance summary.

    Args:
        model: Trained CatBoostClassifier.
        X: Feature matrix.
        feature_names: Optional feature names (length == n_features).
        max_samples: SHAP computation sample limit.

    Returns:
        GlobalSHAPSummary with feature importance ranking.
    """
    _require_shap()

    # Subsample
    n_orig = len(X)
    if n_orig > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_orig, size=max_samples, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    shap_values, expected_value = generate_shap_explanations(
        model, X_explain, feature_names=feature_names, max_samples=max_samples
    )

    n_features = shap_values.shape[1] if len(shap_values.shape) > 1 else 0
    fnames = feature_names or [f"feature_{i}" for i in range(n_features)]

    # Mean absolute SHAP per feature
    mean_abs = {}
    for j, fname in enumerate(fnames):
        mean_abs[fname] = float(np.mean(np.abs(shap_values[:, j])))

    # Rank by importance
    ranked = sorted(mean_abs, key=lambda f: mean_abs[f], reverse=True)

    return GlobalSHAPSummary(
        mean_abs_shap=mean_abs,
        feature_importance_rank=ranked,
        shap_values=shap_values,
        feature_names=fnames,
        n_samples=len(X_explain),
        expected_value=expected_value,
    )


def individual_pr_explanation(
    model: Any,
    X_single: np.ndarray,
    feature_names: Optional[List[str]] = None,
    pr_id: Optional[Any] = None,
    threshold: float = 0.5,
    top_n: int = 10,
) -> PRExplanation:
    """
    Generate a SHAP explanation for a single PR.

    Args:
        model: Trained CatBoostClassifier.
        X_single: Feature vector for one PR — shape (1, n_features) or (n_features,).
        feature_names: Optional feature names.
        pr_id: Optional PR identifier for display.
        threshold: Classification threshold.
        top_n: Number of top factors to report.

    Returns:
        PRExplanation with human-readable contributing factors.
    """
    _require_shap()

    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    proba = float(model.predict_proba(X_single)[0, 1])
    predicted_label = int(proba >= threshold)
    label_text = "HIGH RISK" if predicted_label == 1 else "LOW RISK"

    shap_values, expected_value = generate_shap_explanations(
        model, X_single, feature_names=feature_names, max_samples=1
    )

    n_features = X_single.shape[1]
    fnames = feature_names or [f"feature_{i}" for i in range(n_features)]
    sv = shap_values[0]  # shape (n_features,)

    contributions = []
    for j, fname in enumerate(fnames):
        sv_j = float(sv[j])
        fval = float(X_single[0, j])
        if sv_j > 0.001:
            direction = "increases_risk"
        elif sv_j < -0.001:
            direction = "decreases_risk"
        else:
            direction = "neutral"
        contributions.append(FeatureContribution(
            feature_name=fname,
            value=fval,
            shap_value=sv_j,
            direction=direction,
        ))

    # Sort by absolute SHAP value
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    top_risk = [
        f"{c.feature_name} (value={c.value:.3f}, SHAP={c.shap_value:+.4f})"
        for c in contributions[:top_n]
        if c.direction == "increases_risk"
    ]
    top_protective = [
        f"{c.feature_name} (value={c.value:.3f}, SHAP={c.shap_value:+.4f})"
        for c in contributions[:top_n]
        if c.direction == "decreases_risk"
    ]

    return PRExplanation(
        pr_id=pr_id,
        predicted_probability=round(proba, 4),
        predicted_label=predicted_label,
        label_text=label_text,
        contributions=contributions,
        top_risk_factors=top_risk,
        top_protective_factors=top_protective,
        base_value=expected_value,
    )


def format_explanation(explanation: PRExplanation, max_factors: int = 5) -> str:
    """
    Format a PRExplanation as a human-readable string.

    Args:
        explanation: PRExplanation from individual_pr_explanation().
        max_factors: Maximum number of factors to list.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        f"PR Prediction: {explanation.label_text}",
        f"  Probability: {explanation.predicted_probability:.4f}",
        f"  SHAP base value (expected): {explanation.base_value:.4f}",
        "",
    ]

    risk_factors = explanation.top_risk_factors[:max_factors]
    protective_factors = explanation.top_protective_factors[:max_factors]

    if risk_factors:
        lines.append("  Major risk-increasing factors:")
        for f in risk_factors:
            lines.append(f"    + {f}")
    else:
        lines.append("  No major risk-increasing factors identified.")

    lines.append("")

    if protective_factors:
        lines.append("  Major risk-decreasing factors:")
        for f in protective_factors:
            lines.append(f"    - {f}")
    else:
        lines.append("  No major risk-decreasing factors identified.")

    return "\n".join(lines)


def print_global_importance(
    summary: GlobalSHAPSummary,
    top_n: int = 20,
) -> str:
    """
    Format a GlobalSHAPSummary as a human-readable feature importance table.

    Args:
        summary: GlobalSHAPSummary from global_shap_summary().
        top_n: Number of top features to show.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        f"Global SHAP Feature Importance ({summary.n_samples} samples)",
        f"  {'Rank':<5} {'Feature':<45} {'Mean |SHAP|':>12}",
        "  " + "-" * 62,
    ]
    for rank, fname in enumerate(summary.feature_importance_rank[:top_n], 1):
        mean_abs = summary.mean_abs_shap[fname]
        lines.append(f"  {rank:<5} {fname:<45} {mean_abs:>12.6f}")
    return "\n".join(lines)
