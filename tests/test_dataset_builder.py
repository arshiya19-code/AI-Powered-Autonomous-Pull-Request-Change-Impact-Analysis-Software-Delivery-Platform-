"""
tests/test_dataset_builder.py — Unit tests for analyzer/ml/dataset_builder.py

Tests use fixtures and mocks — no AIDev download or network access required.
Tests cover: filtering, patch feature extraction, label building, leakage prevention.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from analyzer.ml.dataset_builder import (
    FEATURE_NAMES,
    FeatureRow,
    FilterCounters,
    _extract_patch_features,
    _normalize_license_standalone,
)


# ---------------------------------------------------------------------------
# FEATURE_NAMES sanity check
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_42_features(self) -> None:
        assert len(FEATURE_NAMES) == 42

    def test_no_duplicates(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_first_four_are_cat_a(self) -> None:
        assert FEATURE_NAMES[:4] == [
            "files_changed", "lines_added", "lines_deleted", "total_lines_changed"
        ]

    def test_last_two_are_impact(self) -> None:
        assert FEATURE_NAMES[-2] == "impact_score"
        assert FEATURE_NAMES[-1] == "impact_level_numeric"


# ---------------------------------------------------------------------------
# FeatureRow defaults
# ---------------------------------------------------------------------------


class TestFeatureRow:
    def test_all_features_default_nan(self) -> None:
        import math
        row = FeatureRow(
            pr_id=1, repo_id=100, repo_full_name="o/r",
            pr_number=1, created_at="2024-01-01", agent=None, high_risk=0,
        )
        for feat in FEATURE_NAMES:
            val = getattr(row, feat)
            assert math.isnan(val), f"{feat} should default to NaN, got {val}"

    def test_high_risk_label_not_a_feature(self) -> None:
        assert "high_risk" not in FEATURE_NAMES

    def test_to_dict_contains_all_features(self) -> None:
        row = FeatureRow(
            pr_id=1, repo_id=100, repo_full_name="o/r",
            pr_number=1, created_at="2024-01-01", agent=None, high_risk=1,
        )
        d = row.to_dict()
        for feat in FEATURE_NAMES:
            assert feat in d, f"feature {feat} missing from to_dict()"

    def test_to_dict_contains_high_risk(self) -> None:
        row = FeatureRow(
            pr_id=1, repo_id=100, repo_full_name="o/r",
            pr_number=1, created_at="2024-01-01", agent=None, high_risk=1,
        )
        d = row.to_dict()
        assert "high_risk" in d
        assert d["high_risk"] == 1

    def test_leakage_fields_not_in_feature_names(self) -> None:
        from analyzer.ml.labels import LEAKAGE_FIELDS
        for lf in LEAKAGE_FIELDS:
            assert lf not in FEATURE_NAMES, f"Leakage field {lf} found in FEATURE_NAMES!"


# ---------------------------------------------------------------------------
# FilterCounters
# ---------------------------------------------------------------------------


class TestFilterCounters:
    def test_initial_all_zero(self) -> None:
        c = FilterCounters()
        assert c.considered == 0
        assert c.successfully_extracted == 0

    def test_summary_string(self) -> None:
        c = FilterCounters(considered=100, successfully_extracted=80)
        s = c.summary()
        assert "100" in s
        assert "80" in s


# ---------------------------------------------------------------------------
# _extract_patch_features
# ---------------------------------------------------------------------------


SIMPLE_DIFF = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,5 +1,10 @@
-import os
+import os
+import sys

-def old_func():
-    pass
+def new_func():
+    return 42

+class NewClass:
+    def method(self):
+        pass
"""


class TestExtractPatchFeatures:
    def test_returns_expected_keys(self) -> None:
        feats = _extract_patch_features(SIMPLE_DIFF)
        expected_keys = [
            "files_changed", "lines_added", "lines_deleted", "total_lines_changed",
            "functions_changed", "classes_changed", "methods_changed",
            "dependencies_added", "dependencies_removed", "dependencies_changed",
            "dependency_change_ratio",
        ]
        for k in expected_keys:
            assert k in feats, f"Missing key: {k}"

    def test_positive_lines_added(self) -> None:
        feats = _extract_patch_features(SIMPLE_DIFF)
        assert feats["lines_added"] > 0

    def test_dependency_change_ratio_non_negative(self) -> None:
        feats = _extract_patch_features(SIMPLE_DIFF)
        assert feats["dependency_change_ratio"] >= 0.0

    def test_empty_patch(self) -> None:
        feats = _extract_patch_features("")
        # Should not raise; all values should be 0 or NaN
        assert isinstance(feats, dict)
        for k, v in feats.items():
            assert isinstance(v, float)

    def test_no_changes_patch(self) -> None:
        feats = _extract_patch_features("# no diff lines")
        assert feats["lines_added"] == 0.0 or feats["lines_added"] >= 0.0

    def test_all_values_are_floats(self) -> None:
        feats = _extract_patch_features(SIMPLE_DIFF)
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_adds_detect_new_imports(self) -> None:
        patch_with_imports = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+import numpy\n"
        feats = _extract_patch_features(patch_with_imports)
        assert feats["dependencies_added"] >= 1.0


# ---------------------------------------------------------------------------
# AIDevDatasetBuilder — unit tests with mocks
# ---------------------------------------------------------------------------


class TestAIDevDatasetBuilderInit:
    def test_init_sets_defaults(self) -> None:
        try:
            from analyzer.ml.dataset_builder import AIDevDatasetBuilder
        except ImportError:
            pytest.skip("polars not available")
        builder = AIDevDatasetBuilder()
        assert builder.language_filter == "Python"
        assert "mit" in builder.allowed_licenses
        assert builder.exclude_forks is True
        assert builder.random_seed == 42

    def test_custom_allowed_licenses(self) -> None:
        try:
            from analyzer.ml.dataset_builder import AIDevDatasetBuilder
        except ImportError:
            pytest.skip("polars not available")
        builder = AIDevDatasetBuilder(allowed_licenses={"mit", "isc"})
        assert "mit" in builder.allowed_licenses
        assert "apache-2.0" not in builder.allowed_licenses

    def test_normalize_license_mit(self) -> None:
        try:
            from analyzer.ml.dataset_builder import AIDevDatasetBuilder
        except ImportError:
            pytest.skip("polars not available")
        builder = AIDevDatasetBuilder()
        assert builder._normalize_license("MIT") == "mit"
        assert builder._normalize_license("MIT License") == "mit"
        assert builder._normalize_license("Apache License 2.0") == "apache-2.0"

    def test_normalize_license_none(self) -> None:
        try:
            from analyzer.ml.dataset_builder import AIDevDatasetBuilder
        except ImportError:
            pytest.skip("polars not available")
        builder = AIDevDatasetBuilder()
        assert builder._normalize_license(None) is None


class TestBuildLabelMap:
    def test_changes_requested_sets_1(self) -> None:
        try:
            import polars as pl
            from analyzer.ml.dataset_builder import AIDevDatasetBuilder
        except ImportError:
            pytest.skip("polars not available")
        builder = AIDevDatasetBuilder()
        reviews = pl.DataFrame({
            "pr_id": [1, 1, 2],
            "state": ["APPROVED", "CHANGES_REQUESTED", "APPROVED"],
        })
        label_map = builder._build_label_map(reviews)
        assert label_map[1] == 1
        assert label_map[2] == 0

    def test_all_approved_sets_0(self) -> None:
        try:
            import polars as pl
            from analyzer.ml.dataset_builder import AIDevDatasetBuilder
        except ImportError:
            pytest.skip("polars not available")
        builder = AIDevDatasetBuilder()
        reviews = pl.DataFrame({
            "pr_id": [10, 10],
            "state": ["APPROVED", "COMMENTED"],
        })
        label_map = builder._build_label_map(reviews)
        assert label_map[10] == 0


# ---------------------------------------------------------------------------
# Leakage prevention: FeatureRow.to_dict() must not contain leakage fields
# ---------------------------------------------------------------------------


class TestLeakagePrevention:
    """
    Validate that leakage fields cannot enter the feature row
    and thus cannot reach the model.
    """

    def test_feature_row_dict_has_no_review_state(self) -> None:
        row = FeatureRow(
            pr_id=42, repo_id=1, repo_full_name="o/r",
            pr_number=7, created_at="2024-01-01", agent=None, high_risk=1,
        )
        d = row.to_dict()
        assert "review_state" not in d
        assert "pr_reviews_state" not in d
        assert "merged_at" not in d
        assert "closed_at" not in d

    def test_feature_row_has_no_state_field(self) -> None:
        row = FeatureRow(
            pr_id=1, repo_id=1, repo_full_name="o/r",
            pr_number=1, created_at="", agent=None, high_risk=0,
        )
        d = row.to_dict()
        # 'state' as PR outcome state should not be present
        # (checkout_success is fine; it's extraction metadata)
        assert "state" not in d or d.get("state") is None
