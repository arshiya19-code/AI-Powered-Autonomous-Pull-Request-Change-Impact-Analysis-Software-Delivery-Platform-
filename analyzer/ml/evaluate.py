"""
analyzer/ml/evaluate.py — Phase 8B Model Evaluation

Reports:
- accuracy, precision, recall, F1
- ROC-AUC, PR-AUC
- confusion matrix
- repository-aware CV (mean ± std AUC)
- threshold analysis (precision, recall, F1 at multiple thresholds)
- calibration summary
- high-risk class metrics specifically

Never reports synthetic or assumed metric values.
All values come from actual model predictions on held-out data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
try:
    from sklearn.metrics import (  # type: ignore[import-untyped]
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, Pool  # type: ignore[import-untyped]
    _CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    Pool = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Metrics at a single classification threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    n_predicted_positive: int
    n_actual_positive: int


@dataclass
class EvaluationResult:
    """Full evaluation result for a trained model."""

    # Overall metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0

    # High-risk class metrics (label=1)
    high_risk_precision: float = 0.0
    high_risk_recall: float = 0.0
    high_risk_f1: float = 0.0

    # Confusion matrix [[TN, FP], [FN, TP]]
    confusion_matrix: List[List[int]] = field(default_factory=list)
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    # Threshold analysis
    threshold_results: List[ThresholdResult] = field(default_factory=list)
    optimal_threshold: float = 0.5

    # CV results
    cv_mean_auc: float = 0.0
    cv_std_auc: float = 0.0
    cv_fold_aucs: List[float] = field(default_factory=list)
    cv_n_folds: int = 0

    # Calibration
    calibration_mean_error: float = 0.0

    # Metadata
    n_test: int = 0
    n_positive: int = 0
    n_negative: int = 0
    pos_fraction: float = 0.0
    evaluation_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        d: Dict[str, Any] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "high_risk_precision": self.high_risk_precision,
            "high_risk_recall": self.high_risk_recall,
            "high_risk_f1": self.high_risk_f1,
            "confusion_matrix": self.confusion_matrix,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "optimal_threshold": self.optimal_threshold,
            "cv_mean_auc": self.cv_mean_auc,
            "cv_std_auc": self.cv_std_auc,
            "cv_fold_aucs": self.cv_fold_aucs,
            "cv_n_folds": self.cv_n_folds,
            "calibration_mean_error": self.calibration_mean_error,
            "n_test": self.n_test,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "pos_fraction": self.pos_fraction,
            "evaluation_threshold": self.evaluation_threshold,
            "threshold_results": [
                {
                    "threshold": tr.threshold,
                    "precision": tr.precision,
                    "recall": tr.recall,
                    "f1": tr.f1,
                    "n_predicted_positive": tr.n_predicted_positive,
                    "n_actual_positive": tr.n_actual_positive,
                }
                for tr in self.threshold_results
            ],
        }
        return d

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Model Evaluation Summary",
            "=" * 60,
            f"  Test samples      : {self.n_test}"
            f"  (+{self.n_positive} / -{self.n_negative},"
            f" pos_frac={self.pos_fraction:.3f})",
            "",
            f"  Accuracy          : {self.accuracy:.4f}",
            f"  ROC-AUC           : {self.roc_auc:.4f}",
            f"  PR-AUC            : {self.pr_auc:.4f}",
            "",
            f"  Precision (macro) : {self.precision:.4f}",
            f"  Recall    (macro) : {self.recall:.4f}",
            f"  F1        (macro) : {self.f1:.4f}",
            "",
            "  High-Risk Class:",
            f"    Precision : {self.high_risk_precision:.4f}",
            f"    Recall    : {self.high_risk_recall:.4f}",
            f"    F1        : {self.high_risk_f1:.4f}",
            "",
            f"  Confusion Matrix (TN={self.tn}, FP={self.fp},"
            f" FN={self.fn}, TP={self.tp})",
            "",
            f"  CV AUC (repo-aware): {self.cv_mean_auc:.4f}"
            f" ± {self.cv_std_auc:.4f}  (n_folds={self.cv_n_folds})",
            "",
            "  Threshold analysis:",
        ]
        for tr in self.threshold_results:
            lines.append(
                f"    threshold={tr.threshold:.2f}"
                f"  P={tr.precision:.3f}"
                f"  R={tr.recall:.3f}"
                f"  F1={tr.f1:.3f}"
                f"  predicted_pos={tr.n_predicted_positive}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required. pip install scikit-learn")


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    thresholds_to_test: Optional[List[float]] = None,
) -> EvaluationResult:
    """
    Evaluate a trained CatBoost model on a test set.

    Args:
        model: Trained CatBoostClassifier.
        X_test: Feature matrix (n_samples, n_features).
        y_test: True labels (n_samples,).
        feature_names: Optional feature names.
        threshold: Default classification threshold.
        thresholds_to_test: List of thresholds for threshold analysis.

    Returns:
        EvaluationResult with all metrics.
    """
    _require_sklearn()
    if thresholds_to_test is None:
        thresholds_to_test = [0.30, 0.40, 0.50, 0.60, 0.70]

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    n = len(y_test)
    n_pos = int(np.sum(y_test))
    n_neg = n - n_pos

    result = EvaluationResult(
        n_test=n,
        n_positive=n_pos,
        n_negative=n_neg,
        pos_fraction=round(n_pos / max(n, 1), 4),
        evaluation_threshold=threshold,
    )

    result.accuracy = float(accuracy_score(y_test, y_pred))
    result.precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    result.recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    result.f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # High-risk class (label = 1)
    result.high_risk_precision = float(
        precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    )
    result.high_risk_recall = float(
        recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    )
    result.high_risk_f1 = float(
        f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    )

    # ROC-AUC
    if n_pos > 0 and n_neg > 0:
        result.roc_auc = float(roc_auc_score(y_test, proba))
        result.pr_auc = float(average_precision_score(y_test, proba))
    else:
        logger.warning("Cannot compute ROC-AUC: only one class present in y_test.")
        result.roc_auc = float("nan")
        result.pr_auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        result.tn, result.fp, result.fn, result.tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    result.confusion_matrix = cm.tolist()

    # Threshold analysis
    result.threshold_results = threshold_analysis(y_test, proba, thresholds_to_test)

    # Optimal threshold (max F1 for high-risk class)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, proba)
    f1_arr = np.where(
        (prec_arr + rec_arr) == 0, 0.0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr)
    )
    if len(thr_arr) > 0:
        best_idx = int(np.argmax(f1_arr[:-1]))
        result.optimal_threshold = float(thr_arr[best_idx])
    else:
        result.optimal_threshold = threshold

    return result


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[ThresholdResult]:
    """
    Compute precision, recall, F1 at multiple classification thresholds.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        thresholds: List of thresholds to evaluate.

    Returns:
        List of ThresholdResult, one per threshold.
    """
    _require_sklearn()
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    results = []
    n_actual_pos = int(np.sum(y_true))
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        n_pred_pos = int(np.sum(y_pred))
        results.append(ThresholdResult(
            threshold=thr,
            precision=prec,
            recall=rec,
            f1=f1,
            n_predicted_positive=n_pred_pos,
            n_actual_positive=n_actual_pos,
        ))
    return results


def cross_validate_by_repo(
    model_params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> EvaluationResult:
    """
    Run repository-aware cross-validation and return AUC statistics.

    Args:
        model_params: CatBoost hyperparameter dict.
        X, y, groups: Feature matrix, labels, repo groups.
        feature_names: Optional feature names.
        n_splits: Number of CV folds.
        random_state: Random seed.

    Returns:
        EvaluationResult with cv_mean_auc, cv_std_auc, cv_fold_aucs.
    """
    _require_sklearn()
    if not _CATBOOST_AVAILABLE or CatBoostClassifier is None:
        raise ImportError("catboost is required for cross_validate_by_repo.")

    from analyzer.ml.train import RepositoryAwareSplitter, compute_class_weights

    splitter = RepositoryAwareSplitter(n_splits=n_splits)
    splits = splitter.split(X, y, groups)
    class_weights = compute_class_weights(y)

    fold_aucs: List[float] = []
    fold_f1s: List[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        model = CatBoostClassifier(**{**model_params, "verbose": False})
        weights = [class_weights[int(yi)] for yi in y[train_idx]]
        train_pool = Pool(
            X[train_idx], y[train_idx],
            feature_names=feature_names,
            weight=weights,
        )
        val_pool = Pool(X[val_idx], y[val_idx], feature_names=feature_names)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        proba = model.predict_proba(X[val_idx])[:, 1]
        y_val = y[val_idx]
        n_pos = int(np.sum(y_val))
        n_neg = len(y_val) - n_pos

        if n_pos > 0 and n_neg > 0:
            auc = float(roc_auc_score(y_val, proba))
        else:
            logger.warning(f"Fold {fold_i}: only one class in val set — skipping AUC.")
            auc = float("nan")

        y_pred = (proba >= 0.5).astype(int)
        f1 = float(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
        fold_aucs.append(auc)
        fold_f1s.append(f1)
        logger.info(f"Fold {fold_i}: AUC={auc:.4f}  F1={f1:.4f}")

    valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
    result = EvaluationResult()
    result.cv_fold_aucs = fold_aucs
    result.cv_n_folds = len(splits)
    result.cv_mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    result.cv_std_auc = float(np.std(valid_aucs)) if valid_aucs else float("nan")
    result.cv_n_folds = len(splits)
    return result


def calibration_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration statistics (mean absolute calibration error).

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Dict with fraction_of_positives, mean_predicted_value, mean_abs_error.
    """
    _require_sklearn()
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    mean_abs_error = float(np.mean(np.abs(frac_pos - mean_pred)))
    return {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "mean_abs_calibration_error": round(mean_abs_error, 4),
        "n_bins": n_bins,
    }


def save_evaluation_result(
    result: EvaluationResult,
    output_path: Path,
) -> None:
    """Save EvaluationResult to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Evaluation result saved to {output_path}")
