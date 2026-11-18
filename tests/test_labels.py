"""
tests/test_labels.py — Unit tests for analyzer/ml/labels.py

All tests use fixtures/mocks — no AIDev download required.
"""

from __future__ import annotations

import pytest
from analyzer.ml.labels import (
    LEAKAGE_FIELDS,
    assign_high_risk_label,
    assert_no_label_leakage,
    compute_class_balance,
    label_from_review_rows,
    validate_no_label_leakage,
)


# ---------------------------------------------------------------------------
# assign_high_risk_label
# ---------------------------------------------------------------------------


class TestAssignHighRiskLabel:
    def test_changes_requested_returns_1(self) -> None:
        assert assign_high_risk_label(["CHANGES_REQUESTED"]) == 1

    def test_approved_only_returns_0(self) -> None:
        assert assign_high_risk_label(["APPROVED"]) == 0

    def test_mixed_returns_1(self) -> None:
        assert assign_high_risk_label(["APPROVED", "CHANGES_REQUESTED", "COMMENTED"]) == 1

    def test_empty_returns_0(self) -> None:
        assert assign_high_risk_label([]) == 0

    def test_none_values_ignored(self) -> None:
        assert assign_high_risk_label([None, None]) == 0  # type: ignore[list-item]

    def test_empty_string_ignored(self) -> None:
        assert assign_high_risk_label([""]) == 0

    def test_case_insensitive(self) -> None:
        # Comparison is case-insensitive: lowercase must match the same as uppercase
        assert assign_high_risk_label(["changes_requested"]) == 1

    def test_changes_requested_uppercase(self) -> None:
        assert assign_high_risk_label(["CHANGES_REQUESTED"]) == 1

    def test_multiple_changes_requested(self) -> None:
        assert assign_high_risk_label(
            ["CHANGES_REQUESTED", "CHANGES_REQUESTED"]
        ) == 1

    def test_commented_only_returns_0(self) -> None:
        assert assign_high_risk_label(["COMMENTED", "COMMENTED"]) == 0

    def test_dismissed_only_returns_0(self) -> None:
        assert assign_high_risk_label(["DISMISSED"]) == 0


# ---------------------------------------------------------------------------
# label_from_review_rows
# ---------------------------------------------------------------------------


class TestLabelFromReviewRows:
    def test_with_changes_requested_row(self) -> None:
        rows = [{"state": "APPROVED"}, {"state": "CHANGES_REQUESTED"}]
        assert label_from_review_rows(rows) == 1

    def test_all_approved(self) -> None:
        rows = [{"state": "APPROVED"}, {"state": "COMMENTED"}]
        assert label_from_review_rows(rows) == 0

    def test_empty_rows(self) -> None:
        assert label_from_review_rows([]) == 0

    def test_missing_state_key(self) -> None:
        rows = [{"user": "alice"}]
        assert label_from_review_rows(rows) == 0

    def test_none_state(self) -> None:
        rows = [{"state": None}]
        assert label_from_review_rows(rows) == 0  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# validate_no_label_leakage
# ---------------------------------------------------------------------------


class TestValidateNoLabelLeakage:
    def test_clean_feature_dict(self) -> None:
        features = {
            "files_changed": 3.0,
            "lines_added": 100.0,
            "impact_score": 0.7,
        }
        found = validate_no_label_leakage(features)
        assert found == []

    def test_detects_merged_at(self) -> None:
        features = {"merged_at": "2024-01-01", "files_changed": 3.0}
        found = validate_no_label_leakage(features)
        assert "merged_at" in found

    def test_detects_closed_at(self) -> None:
        features = {"closed_at": "2024-01-01"}
        found = validate_no_label_leakage(features)
        assert "closed_at" in found

    def test_detects_review_state(self) -> None:
        features = {"review_state": "CHANGES_REQUESTED"}
        found = validate_no_label_leakage(features)
        assert len(found) > 0

    def test_multiple_leakage_fields(self) -> None:
        features = {"merged_at": "x", "closed_at": "y"}
        found = validate_no_label_leakage(features)
        assert len(found) >= 2


class TestAssertNoLabelLeakage:
    def test_raises_on_leakage(self) -> None:
        with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
            assert_no_label_leakage({"merged_at": "2024-01-01"})

    def test_no_raise_on_clean(self) -> None:
        assert_no_label_leakage({"files_changed": 3.0, "impact_score": 0.5})


# ---------------------------------------------------------------------------
# compute_class_balance
# ---------------------------------------------------------------------------


class TestComputeClassBalance:
    def test_balanced(self) -> None:
        labels = [0, 1, 0, 1]
        result = compute_class_balance(labels)
        assert result["n_positive"] == 2
        assert result["n_negative"] == 2
        assert abs(result["pos_fraction"] - 0.5) < 0.01
        assert result["suggested_weight_ratio"] == pytest.approx(1.0)

    def test_imbalanced(self) -> None:
        labels = [0] * 75 + [1] * 25
        result = compute_class_balance(labels)
        assert result["n_positive"] == 25
        assert result["n_negative"] == 75
        assert abs(result["pos_fraction"] - 0.25) < 0.01
        assert result["suggested_weight_ratio"] == pytest.approx(3.0, rel=0.01)

    def test_empty(self) -> None:
        result = compute_class_balance([])
        assert result["n"] == 0
        assert result["suggested_weight_ratio"] == 1.0

    def test_all_positive(self) -> None:
        result = compute_class_balance([1, 1, 1])
        assert result["n_positive"] == 3
        assert result["n_negative"] == 0

    def test_all_negative(self) -> None:
        result = compute_class_balance([0, 0, 0])
        assert result["n_positive"] == 0


# ---------------------------------------------------------------------------
# LEAKAGE_FIELDS constant
# ---------------------------------------------------------------------------


class TestLeakageFields:
    def test_contains_key_fields(self) -> None:
        required = {"merged_at", "closed_at", "state"}
        assert required.issubset(set(LEAKAGE_FIELDS))

    def test_does_not_contain_safe_features(self) -> None:
        safe = ["files_changed", "impact_score", "lines_added"]
        for s in safe:
            assert s not in LEAKAGE_FIELDS
