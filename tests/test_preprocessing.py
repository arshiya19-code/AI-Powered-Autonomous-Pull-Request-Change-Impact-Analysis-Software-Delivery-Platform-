"""
tests/test_preprocessing.py — Unit tests for analyzer/ml/preprocessing.py

All tests use in-memory fixtures — no AIDev download required.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from analyzer.ml.preprocessing import (
    CANONICAL_FEATURES,
    DatasetQualityReport,
    build_feature_matrix,
    get_repo_groups,
    run_quality_checks,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    pr_id: int = 1,
    repo_id: int = 100,
    high_risk: int = 0,
    **overrides: Any,
) -> Dict[str, Any]:
    """Create a complete 42-feature row dict for testing."""
    row: Dict[str, Any] = {
        "pr_id": pr_id,
        "repo_id": repo_id,
        "repo_full_name": "owner/repo",
        "high_risk": high_risk,
    }
    # All 42 features default to 1.0
    for feat in CANONICAL_FEATURES:
        row[feat] = 1.0
    row.update(overrides)
    return row


def _make_rows(n: int, n_positive: int = 5) -> List[Dict[str, Any]]:
    """Create n complete rows with n_positive labeled 1."""
    rows = []
    for i in range(n):
        label = 1 if i < n_positive else 0
        rows.append(_make_row(pr_id=i + 1, repo_id=(i % 10) + 100, high_risk=label))
    return rows


# ---------------------------------------------------------------------------
# CANONICAL_FEATURES sanity
# ---------------------------------------------------------------------------


class TestCanonicalFeatures:
    def test_exactly_42_features(self) -> None:
        assert len(CANONICAL_FEATURES) == 42

    def test_no_duplicates(self) -> None:
        assert len(CANONICAL_FEATURES) == len(set(CANONICAL_FEATURES))

    def test_starts_with_cat_a(self) -> None:
        assert CANONICAL_FEATURES[0] == "files_changed"
        assert CANONICAL_FEATURES[1] == "lines_added"
        assert CANONICAL_FEATURES[2] == "lines_deleted"
        assert CANONICAL_FEATURES[3] == "total_lines_changed"

    def test_ends_with_cat_c(self) -> None:
        assert CANONICAL_FEATURES[-2] == "impact_score"
        assert CANONICAL_FEATURES[-1] == "impact_level_numeric"


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_schema(self) -> None:
        rows = _make_rows(5)
        errors = validate_schema(rows)
        assert errors == []

    def test_empty_dataset(self) -> None:
        errors = validate_schema([])
        assert len(errors) > 0

    def test_missing_label(self) -> None:
        rows = [_make_row()]
        del rows[0]["high_risk"]
        errors = validate_schema(rows)
        assert any("high_risk" in e for e in errors)

    def test_missing_feature(self) -> None:
        rows = [_make_row()]
        del rows[0]["files_changed"]
        errors = validate_schema(rows)
        assert any("files_changed" in e for e in errors)

    def test_leakage_field_detected(self) -> None:
        rows = [_make_row()]
        rows[0]["merged_at"] = "2024-01-01"
        errors = validate_schema(rows)
        assert any("LEAKAGE" in e.upper() or "merged_at" in e for e in errors)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    def test_shape(self) -> None:
        rows = _make_rows(10, n_positive=3)
        X, y, names = build_feature_matrix(rows)
        assert X.shape == (10, 42)
        assert y.shape == (10,)
        assert len(names) == 42

    def test_feature_order(self) -> None:
        rows = _make_rows(3)
        X, y, names = build_feature_matrix(rows)
        assert names == CANONICAL_FEATURES

    def test_label_extraction(self) -> None:
        rows = _make_rows(10, n_positive=4)
        X, y, names = build_feature_matrix(rows)
        assert int(np.sum(y)) == 4

    def test_empty_returns_zero_rows(self) -> None:
        X, y, names = build_feature_matrix([])
        assert X.shape[0] == 0
        assert y.shape[0] == 0
        assert len(names) == 42

    def test_nan_imputed_with_median(self) -> None:
        rows = _make_rows(10)
        # Set files_changed to specific values for first 9 rows
        for i in range(9):
            rows[i]["files_changed"] = float(i + 1)
        # Last row: NaN → should be imputed with median
        rows[9]["files_changed"] = float("nan")
        X, y, names = build_feature_matrix(rows)
        # The 10th row should NOT be 0 (median imputation)
        fc_idx = names.index("files_changed")
        assert not math.isnan(X[9, fc_idx])
        # Imputed value should be the median of 1-9 = 5.0
        assert X[9, fc_idx] == pytest.approx(5.0)

    def test_all_nan_feature_imputed_to_zero(self) -> None:
        """If an entire feature is NaN (no non-null values), impute to 0."""
        rows = _make_rows(3)
        for r in rows:
            r["files_changed"] = float("nan")
        X, y, names = build_feature_matrix(rows)
        fc_idx = names.index("files_changed")
        assert all(X[i, fc_idx] == 0.0 for i in range(3))

    def test_none_values_imputed(self) -> None:
        rows = _make_rows(5)
        rows[0]["impact_score"] = None  # type: ignore[assignment]
        X, y, names = build_feature_matrix(rows)
        is_idx = names.index("impact_score")
        # Should be imputed with median (1.0 from other rows)
        assert not math.isnan(X[0, is_idx])

    def test_dtype_float64(self) -> None:
        rows = _make_rows(5)
        X, y, names = build_feature_matrix(rows)
        assert X.dtype == np.float64

    def test_label_dtype_int32(self) -> None:
        rows = _make_rows(5)
        X, y, names = build_feature_matrix(rows)
        assert y.dtype == np.int32


# ---------------------------------------------------------------------------
# run_quality_checks
# ---------------------------------------------------------------------------


class TestRunQualityChecks:
    def test_good_dataset_passes(self) -> None:
        rows = _make_rows(20, n_positive=5)
        report = run_quality_checks(rows)
        assert report.schema_ok
        assert report.n_rows == 20
        assert report.n_positive == 5
        assert report.n_negative == 15

    def test_empty_dataset_fails(self) -> None:
        report = run_quality_checks([])
        assert not report.passed

    def test_leakage_detected(self) -> None:
        rows = _make_rows(10)
        for r in rows:
            r["merged_at"] = "2024-01-01"
        report = run_quality_checks(rows)
        assert len(report.leakage_detected) > 0

    def test_duplicate_pr_ids_detected(self) -> None:
        rows = _make_rows(10)
        # Make all pr_ids the same
        for r in rows:
            r["pr_id"] = 999
        report = run_quality_checks(rows)
        assert report.n_duplicates > 0

    def test_class_balance_warning_on_severe_imbalance(self) -> None:
        # 1 positive out of 100 → ratio = 0.01 < 0.1
        rows = _make_rows(100, n_positive=1)
        report = run_quality_checks(rows, min_class_balance_ratio=0.1)
        assert not report.class_balance_ok
        assert any("imbalance" in w.lower() for w in report.warnings)

    def test_missing_fraction_tracked(self) -> None:
        rows = _make_rows(10)
        for r in rows[:5]:
            r["impact_score"] = float("nan")
        report = run_quality_checks(rows)
        assert report.missing_fraction_per_feature["impact_score"] == pytest.approx(0.5)

    def test_quality_report_summary_contains_key_info(self) -> None:
        rows = _make_rows(10, n_positive=3)
        report = run_quality_checks(rows)
        summary = report.summary()
        assert "Rows" in summary
        assert "Label" in summary


# ---------------------------------------------------------------------------
# get_repo_groups
# ---------------------------------------------------------------------------


class TestGetRepoGroups:
    def test_same_repo_same_group(self) -> None:
        rows = [
            {"repo_id": 1, "pr_id": 1},
            {"repo_id": 1, "pr_id": 2},
            {"repo_id": 2, "pr_id": 3},
        ]
        groups = get_repo_groups(rows)
        assert groups[0] == groups[1]  # same repo
        assert groups[0] != groups[2]  # different repo

    def test_output_shape(self) -> None:
        rows = [{"repo_id": i} for i in range(10)]
        groups = get_repo_groups(rows)
        assert len(groups) == 10

    def test_missing_repo_id_handled(self) -> None:
        rows = [{"pr_id": 1}, {"pr_id": 2}]  # no repo_id
        groups = get_repo_groups(rows)
        assert len(groups) == 2
        # All unknown repos should be the same group
        assert groups[0] == groups[1]
