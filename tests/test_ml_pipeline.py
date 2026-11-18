"""
tests/test_ml_pipeline.py — End-to-end ML pipeline tests (mocked, no network)

Tests the full pipeline: feature matrix → CatBoost → evaluation → SHAP
using synthetic in-memory data (NOT AIDev data — these are pipeline validation
tests, not dataset tests).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from analyzer.ml.preprocessing import (
    CANONICAL_FEATURES,
    build_feature_matrix,
    get_repo_groups,
    run_quality_checks,
)
from analyzer.ml.labels import assign_high_risk_label, compute_class_balance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_rows(n: int = 50, n_positive: int = 15) -> List[Dict[str, Any]]:
    """Create n synthetic rows for pipeline testing."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n):
        label = 1 if i < n_positive else 0
        row: Dict[str, Any] = {
            "pr_id": i + 1,
            "repo_id": (i % 10) + 100,
            "repo_full_name": f"owner/repo{(i % 10)}",
            "high_risk": label,
        }
        # Generate realistic-ish feature values
        for j, feat in enumerate(CANONICAL_FEATURES):
            if label == 1:
                row[feat] = float(rng.uniform(5.0, 20.0))
            else:
                row[feat] = float(rng.uniform(0.0, 5.0))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


class TestPreprocessingPipeline:
    def test_build_matrix_from_synthetic_data(self) -> None:
        rows = _make_feature_rows(50)
        X, y, names = build_feature_matrix(rows)
        assert X.shape == (50, 42)
        assert y.shape == (50,)
        assert names == CANONICAL_FEATURES

    def test_no_nan_in_matrix(self) -> None:
        rows = _make_feature_rows(50)
        X, y, names = build_feature_matrix(rows)
        assert not np.any(np.isnan(X))

    def test_group_array_shape(self) -> None:
        rows = _make_feature_rows(50)
        groups = get_repo_groups(rows)
        assert len(groups) == 50

    def test_quality_check_passes(self) -> None:
        rows = _make_feature_rows(50, n_positive=15)
        report = run_quality_checks(rows)
        assert report.schema_ok
        assert report.n_positive == 15

    def test_class_balance_correct(self) -> None:
        rows = _make_feature_rows(50, n_positive=15)
        report = run_quality_checks(rows)
        assert report.n_positive == 15
        assert report.n_negative == 35
        assert abs(report.pos_fraction - 0.30) < 0.01


# ---------------------------------------------------------------------------
# CatBoost training pipeline
# ---------------------------------------------------------------------------


class TestCatBoostTraining:
    @pytest.fixture
    def training_data(self):  # type: ignore[no-untyped-def]
        rows = _make_feature_rows(100, n_positive=30)
        X, y, names = build_feature_matrix(rows)
        groups = get_repo_groups(rows)
        return X, y, names, groups

    def test_train_catboost(self, training_data) -> None:  # type: ignore[no-untyped-def]
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            pytest.skip("catboost not installed")

        from analyzer.ml.train import train_catboost

        X, y, names, groups = training_data
        result = train_catboost(
            X, y, groups,
            feature_names=names,
            params={"iterations": 20, "verbose": False, "allow_writing_files": False},
        )
        assert result.model is not None
        assert 0.0 <= result.val_auc <= 1.0
        assert 0.0 <= result.train_auc <= 1.0
        assert result.n_train > 0
        assert result.n_val > 0

    def test_class_weights_computed_from_data(self, training_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.train import compute_class_weights

        X, y, names, groups = training_data
        weights = compute_class_weights(y)
        assert 0 in weights
        assert 1 in weights
        assert weights[0] == 1.0
        # Negative class (n=70) / Positive class (n=30) ≈ 2.33
        assert weights[1] > 1.0

    def test_repository_splitter(self, training_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.train import RepositoryAwareSplitter

        X, y, names, groups = training_data
        splitter = RepositoryAwareSplitter(n_splits=5)
        splits = splitter.split(X, y, groups)
        assert len(splits) > 0
        for train_idx, val_idx in splits:
            # No repo appears in both train and val
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            assert len(train_groups & val_groups) == 0, \
                "Repository leakage detected: same repo in train AND val!"

    def test_single_split_no_repo_overlap(self, training_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.train import RepositoryAwareSplitter

        X, y, names, groups = training_data
        splitter = RepositoryAwareSplitter()
        train_idx, val_idx = splitter.get_single_split(X, y, groups)
        train_repos = set(groups[train_idx])
        val_repos = set(groups[val_idx])
        assert len(train_repos & val_repos) == 0


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


class TestEvaluationPipeline:
    @pytest.fixture
    def trained_model_and_data(self):  # type: ignore[no-untyped-def]
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            pytest.skip("catboost not installed")

        from analyzer.ml.train import train_catboost

        rows = _make_feature_rows(100, n_positive=30)
        X, y, names = build_feature_matrix(rows)
        groups = get_repo_groups(rows)
        result = train_catboost(
            X, y, groups,
            feature_names=names,
            params={"iterations": 20, "verbose": False, "allow_writing_files": False},
        )
        # Use full dataset as test set for simplicity in unit tests
        return result.model, X, y, names

    def test_evaluate_model(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.evaluate import evaluate_model

        model, X, y, names = trained_model_and_data
        result = evaluate_model(model, X, y, feature_names=names)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.roc_auc <= 1.0
        assert 0.0 <= result.high_risk_recall <= 1.0
        assert result.n_test == len(y)

    def test_threshold_analysis(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.evaluate import evaluate_model

        model, X, y, names = trained_model_and_data
        result = evaluate_model(model, X, y, thresholds_to_test=[0.3, 0.5, 0.7])
        assert len(result.threshold_results) == 3
        thresholds = [r.threshold for r in result.threshold_results]
        assert 0.3 in thresholds
        assert 0.5 in thresholds
        assert 0.7 in thresholds

    def test_confusion_matrix_shape(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.evaluate import evaluate_model

        model, X, y, names = trained_model_and_data
        result = evaluate_model(model, X, y)
        assert result.tn + result.fp + result.fn + result.tp == len(y)

    def test_evaluation_result_to_dict(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.evaluate import evaluate_model

        model, X, y, names = trained_model_and_data
        result = evaluate_model(model, X, y)
        d = result.to_dict()
        assert "roc_auc" in d
        assert "high_risk_recall" in d
        assert "confusion_matrix" in d

    def test_evaluation_summary_string(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.evaluate import evaluate_model

        model, X, y, names = trained_model_and_data
        result = evaluate_model(model, X, y)
        summary = result.summary()
        assert "ROC-AUC" in summary
        assert "High-Risk" in summary


# ---------------------------------------------------------------------------
# SHAP pipeline
# ---------------------------------------------------------------------------


class TestSHAPPipeline:
    @pytest.fixture
    def trained_model_and_data(self):  # type: ignore[no-untyped-def]
        try:
            from catboost import CatBoostClassifier
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("catboost or shap not installed")

        from analyzer.ml.train import train_catboost

        rows = _make_feature_rows(80, n_positive=25)
        X, y, names = build_feature_matrix(rows)
        groups = get_repo_groups(rows)
        result = train_catboost(
            X, y, groups,
            feature_names=names,
            params={"iterations": 20, "verbose": False, "allow_writing_files": False},
        )
        return result.model, X, y, names

    def test_global_shap_summary(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.explain import global_shap_summary

        model, X, y, names = trained_model_and_data
        summary = global_shap_summary(model, X, feature_names=names, max_samples=50)
        assert len(summary.mean_abs_shap) == 42
        assert len(summary.feature_importance_rank) == 42
        assert summary.n_samples <= 50
        # All SHAP values should be non-negative (mean absolute)
        for feat, val in summary.mean_abs_shap.items():
            assert val >= 0.0, f"Mean |SHAP| for {feat} is negative: {val}"

    def test_individual_explanation(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        from analyzer.ml.explain import individual_pr_explanation, format_explanation

        model, X, y, names = trained_model_and_data
        explanation = individual_pr_explanation(
            model, X[0:1], feature_names=names, pr_id=42
        )
        assert explanation.predicted_label in (0, 1)
        assert explanation.label_text in ("HIGH RISK", "LOW RISK")
        assert 0.0 <= explanation.predicted_probability <= 1.0
        assert len(explanation.contributions) == 42

        formatted = format_explanation(explanation)
        assert explanation.label_text in formatted

    def test_shap_contributions_sum_to_prediction(self, trained_model_and_data) -> None:  # type: ignore[no-untyped-def]
        """SHAP values + base_value should approximately equal log-odds of prediction."""
        from analyzer.ml.explain import individual_pr_explanation

        model, X, y, names = trained_model_and_data
        explanation = individual_pr_explanation(model, X[0:1], feature_names=names)
        total_shap = sum(c.shap_value for c in explanation.contributions)
        # base + sum(shap) ≈ log-odds of prediction (relaxed tolerance)
        approx_log_odds = explanation.base_value + total_shap
        import math
        prob_from_shap = 1.0 / (1.0 + math.exp(-approx_log_odds))
        # Should be close to the predicted probability
        assert abs(prob_from_shap - explanation.predicted_probability) < 0.1
