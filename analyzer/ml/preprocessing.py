"""
analyzer/ml/preprocessing.py — Phase 8B Feature Matrix Construction

Responsibilities:
- Validate that the dataset has the expected 42-feature schema.
- Separate features from label and metadata.
- Enforce no-leakage contract.
- Handle missing values per category strategy (from dataset/config.py).
- Return a CatBoost-compatible numpy feature matrix and label vector.
- Preserve exact feature order as defined in dataset/config.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected feature names in canonical order (42 features)
# Matches ALL_FEATURES from dataset/config.py exactly.
# ---------------------------------------------------------------------------

CANONICAL_FEATURES: List[str] = [
    # Category A — direct from AIDev metadata
    "files_changed",
    "lines_added",
    "lines_deleted",
    "total_lines_changed",
    # Category B — derived from patch text
    "functions_changed",
    "classes_changed",
    "methods_changed",
    "dependencies_added",
    "dependencies_removed",
    "dependencies_changed",
    # Category B/C — partial
    "dependency_change_ratio",
    # Category C — requires repository checkout
    "direct_nodes_changed",
    "impacted_nodes",
    "impacted_modules",
    "impacted_functions",
    "impacted_classes",
    "impacted_methods",
    "maximum_impact_depth",
    "impact_ratio",
    "average_impact_depth",
    "changed_modules",
    "cross_module_impact_count",
    "cross_module_ratio",
    "average_changed_node_degree",
    "maximum_changed_node_degree",
    "average_changed_node_dependents",
    "maximum_changed_node_dependents",
    "average_changed_node_dependencies",
    "maximum_changed_node_dependencies",
    "import_propagation_count",
    "from_import_propagation_count",
    "call_propagation_count",
    "inheritance_propagation_count",
    "decoration_propagation_count",
    "total_propagation_edges",
    "critical_node_count",
    "nodes_at_depth_0",
    "nodes_at_depth_1",
    "nodes_at_depth_2",
    "nodes_at_depth_3_plus",
    "impact_score",
    "impact_level_numeric",
]

assert len(CANONICAL_FEATURES) == 42, (
    f"CANONICAL_FEATURES must have 42 entries, got {len(CANONICAL_FEATURES)}"
)

# Leakage fields (must never appear in feature matrix)
_LEAKAGE_FIELDS = frozenset({
    "merged_at", "closed_at", "state", "pr_reviews_state", "review_state",
    "pr_state", "review_submitted_at", "review_user", "review_body",
    "high_risk",  # label itself — separate from features
})

# Metadata columns allowed in dataset but not in feature matrix
_METADATA_COLUMNS = frozenset({
    "pr_id", "repo_id", "repo_full_name", "pr_url", "pr_number",
    "created_at", "agent", "checkout_success", "analysis_mode",
    "extraction_error",
})


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------


@dataclass
class DatasetQualityReport:
    """Summary of dataset quality checks."""

    n_rows: int = 0
    n_features: int = 0
    n_missing_per_feature: Dict[str, int] = field(default_factory=dict)
    missing_fraction_per_feature: Dict[str, float] = field(default_factory=dict)
    features_above_missing_threshold: List[str] = field(default_factory=list)
    n_duplicates: int = 0
    n_positive: int = 0
    n_negative: int = 0
    pos_fraction: float = 0.0
    class_balance_ok: bool = True
    schema_ok: bool = True
    schema_errors: List[str] = field(default_factory=list)
    leakage_detected: List[str] = field(default_factory=list)
    passed: bool = True
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Dataset Quality Report",
            f"  Rows: {self.n_rows}",
            f"  Features: {self.n_features}",
            f"  Label: +{self.n_positive} / -{self.n_negative}"
            f" (pos_frac={self.pos_fraction:.3f})",
            f"  Duplicates: {self.n_duplicates}",
            f"  Schema OK: {self.schema_ok}",
            f"  Passed: {self.passed}",
        ]
        if self.schema_errors:
            lines.append(f"  Schema errors: {self.schema_errors}")
        if self.leakage_detected:
            lines.append(f"  ⚠ LEAKAGE: {self.leakage_detected}")
        if self.features_above_missing_threshold:
            lines.append(
                f"  High-missing features: {self.features_above_missing_threshold}"
            )
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------


def validate_schema(
    rows: List[Dict[str, Any]],
) -> List[str]:
    """
    Validate that all rows have the 42 expected feature columns and
    the label column 'high_risk'. Returns list of error messages.

    Args:
        rows: List of feature dicts (one per PR).

    Returns:
        List of schema error messages (empty = schema OK).
    """
    errors: List[str] = []
    if not rows:
        errors.append("Dataset is empty — no rows to validate.")
        return errors

    sample = rows[0]
    present = set(sample.keys())

    # Check label
    if "high_risk" not in present:
        errors.append("Missing required label column 'high_risk'.")

    # Check features
    missing_features = [f for f in CANONICAL_FEATURES if f not in present]
    if missing_features:
        errors.append(f"Missing feature columns: {missing_features}")

    # Check for unexpected leakage fields
    leaked = [f for f in _LEAKAGE_FIELDS if f in present and f != "high_risk"]
    if leaked:
        errors.append(f"LEAKAGE FIELDS FOUND: {leaked}")

    return errors


def _impute_missing(
    value: Any,
    feature: str,
    medians: Dict[str, float],
    cat_c_features: Optional[List[str]] = None,
) -> float:
    """
    Impute a single missing/NaN value.

    Strategy:
    - Cat C features: median imputation (pass None → medians[feature])
    - Other features: 0.0 (or median if available)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        median = medians.get(feature)
        if median is not None:
            return float(median)
        return 0.0
    return float(value)


def build_feature_matrix(
    rows: List[Dict[str, Any]],
    max_missing_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a (X, y, feature_names) triple from a list of feature dicts.

    Args:
        rows: List of dicts, each containing the 42 features + 'high_risk'.
        max_missing_fraction: Features above this missing rate trigger a warning.

    Returns:
        (X, y, feature_names) where:
            X: float64 array of shape (n_samples, 42)
            y: int32 array of shape (n_samples,)
            feature_names: List[str] of length 42 in canonical order
    """
    if not rows:
        return (
            np.zeros((0, len(CANONICAL_FEATURES)), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            CANONICAL_FEATURES,
        )

    n = len(rows)

    # Compute per-feature medians for imputation (from non-null values)
    medians: Dict[str, float] = {}
    for feat in CANONICAL_FEATURES:
        values = [
            float(r[feat])
            for r in rows
            if feat in r and r[feat] is not None
            and not (isinstance(r[feat], float) and np.isnan(r[feat]))
        ]
        if values:
            medians[feat] = float(np.median(values))
        else:
            medians[feat] = 0.0

    # Build arrays
    X = np.full((n, len(CANONICAL_FEATURES)), np.nan, dtype=np.float64)
    y = np.zeros(n, dtype=np.int32)

    for i, row in enumerate(rows):
        for j, feat in enumerate(CANONICAL_FEATURES):
            raw = row.get(feat)
            X[i, j] = _impute_missing(raw, feat, medians)
        y[i] = int(row.get("high_risk", 0))

    return X, y, CANONICAL_FEATURES


def run_quality_checks(
    rows: List[Dict[str, Any]],
    max_missing_fraction: float = 0.5,
    min_class_balance_ratio: float = 0.1,
) -> DatasetQualityReport:
    """
    Run full quality checks on the dataset.

    Args:
        rows: List of feature dicts.
        max_missing_fraction: Threshold for flagging high-missing features.
        min_class_balance_ratio: Minimum (minority / majority) ratio.

    Returns:
        DatasetQualityReport with pass/fail status.
    """
    report = DatasetQualityReport(n_rows=len(rows), n_features=len(CANONICAL_FEATURES))

    # Schema check
    schema_errors = validate_schema(rows)
    report.schema_errors = schema_errors
    report.schema_ok = len(schema_errors) == 0

    # Leakage check — check all keys across all rows
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    leaked = [f for f in _LEAKAGE_FIELDS if f in all_keys and f != "high_risk"]
    report.leakage_detected = leaked

    if not rows:
        report.passed = False
        return report

    n = len(rows)

    # Missing fraction per feature
    for feat in CANONICAL_FEATURES:
        n_missing = sum(
            1 for r in rows
            if r.get(feat) is None
            or (isinstance(r.get(feat), float) and np.isnan(float(r[feat])))
        )
        frac = n_missing / n
        report.n_missing_per_feature[feat] = n_missing
        report.missing_fraction_per_feature[feat] = round(frac, 4)
        if frac > max_missing_fraction:
            report.features_above_missing_threshold.append(feat)

    # Duplicates (by pr_id if present)
    pr_ids = [r.get("pr_id") for r in rows if r.get("pr_id") is not None]
    report.n_duplicates = len(pr_ids) - len(set(pr_ids))

    # Class balance
    labels = [int(r.get("high_risk", 0)) for r in rows]
    report.n_positive = sum(labels)
    report.n_negative = n - report.n_positive
    report.pos_fraction = report.n_positive / n

    minority = min(report.n_positive, report.n_negative)
    majority = max(report.n_positive, report.n_negative)
    ratio = minority / majority if majority > 0 else 0.0
    report.class_balance_ok = ratio >= min_class_balance_ratio
    if not report.class_balance_ok:
        report.warnings.append(
            f"Severe class imbalance: minority/majority = {ratio:.3f} "
            f"(threshold {min_class_balance_ratio})"
        )

    # Warnings
    if report.n_duplicates > 0:
        report.warnings.append(f"{report.n_duplicates} duplicate PR IDs found.")
    if leaked:
        report.warnings.append(f"LEAKAGE FIELDS IN DATASET: {leaked}")
    if not report.schema_ok:
        report.warnings.extend(schema_errors)

    # Overall pass/fail
    report.passed = (
        report.schema_ok
        and not leaked
        and report.class_balance_ok
        and report.n_duplicates == 0
        and not report.features_above_missing_threshold
    )
    # Relax: missing features above threshold = warning only, not fatal
    # (checkout failures can cause legitimate missing Cat C features)
    report.passed = (
        report.schema_ok
        and not leaked
        and report.class_balance_ok
        and report.n_duplicates == 0
    )

    return report


def get_repo_groups(rows: List[Dict[str, Any]]) -> np.ndarray:
    """
    Return an integer group array for GroupKFold (one group per repo).

    Args:
        rows: List of feature dicts (each with 'repo_id' key).

    Returns:
        Integer numpy array of length n_rows, with unique repo_id per group.
    """
    # Map repo_id to integer group index
    repo_ids = [r.get("repo_id", "unknown") for r in rows]
    unique_repos = {rid: i for i, rid in enumerate(dict.fromkeys(repo_ids))}
    return np.array([unique_repos[rid] for rid in repo_ids], dtype=np.int64)
