"""
scripts/run_pilot_25.py — 25-PR Pilot: Real AIDev Ingestion Audit

Purpose:
    End-to-end validation of the Phase 8B ingestion pipeline on 25 real AIDev PRs.
    Produces a detailed audit report for docs/phase8b_dataset_report.md.

Usage:
    # Patch-only (no checkout): fast, validates Cat A+B features
    python scripts/run_pilot_25.py --no-checkout

    # Full checkout: validates full 42-feature pipeline
    python scripts/run_pilot_25.py

Design rules enforced:
    - 25 PRs exactly (no more, no less attempted)
    - No synthetic data
    - Checkout failure -> NaN Cat C, not zero
    - No leakage fields in feature matrix
    - Label: high_risk=1 iff CHANGES_REQUESTED (case-insensitive)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzer.ml.dataset_builder import (
    AIDevDatasetBuilder,
    FEATURE_NAMES,
    FeatureRow,
)
from analyzer.ml.labels import (
    LEAKAGE_FIELDS,
    compute_class_balance,
)
from analyzer.ml.preprocessing import (
    CANONICAL_FEATURES,
    run_quality_checks,
    validate_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _is_nan(v: Any) -> bool:
    """Return True if v is NaN."""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def audit_rows(rows: List[FeatureRow]) -> Dict[str, Any]:
    """
    Produce a detailed audit dict from extracted feature rows.

    Checks:
    - Feature count = 42
    - Canonical feature order matches
    - No review-state leakage in features
    - high_risk exists only as label
    - No synthetic rows (all pr_ids must be distinct)
    - No duplicated PRs
    - No fabricated values (no zero-fill masking NaN)
    - Numeric types valid
    - Label distribution
    - Feature completeness per category
    - Suspicious all-zero or constant features
    """
    audit: Dict[str, Any] = {}
    n = len(rows)
    audit["n_rows"] = n

    if n == 0:
        audit["error"] = "No rows extracted"
        return audit

    # --- Feature count validation ---
    audit["feature_count"] = len(FEATURE_NAMES)
    audit["feature_count_ok"] = len(FEATURE_NAMES) == 42
    audit["canonical_order_match"] = FEATURE_NAMES == CANONICAL_FEATURES

    # --- Label distribution ---
    labels = [r.high_risk for r in rows]
    n_high = sum(labels)
    n_low = n - n_high
    audit["label_distribution"] = {
        "n_high_risk": n_high,
        "n_low_risk": n_low,
        "pos_fraction": round(n_high / n, 4) if n > 0 else 0.0,
    }

    # --- Checkout statistics ---
    n_checkout_ok = sum(1 for r in rows if r.checkout_success)
    n_checkout_fail = sum(1 for r in rows if not r.checkout_success)
    analysis_modes: Dict[str, int] = {}
    for r in rows:
        analysis_modes[r.analysis_mode] = analysis_modes.get(r.analysis_mode, 0) + 1
    audit["checkout"] = {
        "success": n_checkout_ok,
        "failed": n_checkout_fail,
        "analysis_modes": analysis_modes,
    }

    # --- Feature completeness ---
    row_dicts = [r.to_dict() for r in rows]
    missing_per_feat: Dict[str, int] = {}
    for feat in FEATURE_NAMES:
        n_missing = sum(1 for rd in row_dicts if _is_nan(rd.get(feat)))
        missing_per_feat[feat] = n_missing
    audit["feature_completeness"] = {
        feat: {
            "n_missing": missing_per_feat[feat],
            "missing_frac": round(missing_per_feat[feat] / n, 4),
        }
        for feat in FEATURE_NAMES
    }

    # --- Suspicious all-zero features ---
    zero_features = []
    for feat in FEATURE_NAMES:
        vals = [float(rd.get(feat, 0)) for rd in row_dicts if not _is_nan(rd.get(feat))]
        if vals and all(v == 0.0 for v in vals):
            zero_features.append(feat)
    audit["all_zero_features"] = zero_features

    # Checkout validation must demonstrate real extraction, not merely a
    # schema-valid matrix.  Count populated non-zero values by feature family.
    change_features = FEATURE_NAMES[:11]
    graph_impact_features = FEATURE_NAMES[11:]
    audit["nonzero_change_features"] = {
        feat: sum(
            1 for rd in row_dicts
            if not _is_nan(rd.get(feat)) and float(rd[feat]) != 0.0
        )
        for feat in change_features
    }
    audit["nonzero_graph_impact_features"] = {
        feat: sum(
            1 for rd in row_dicts
            if not _is_nan(rd.get(feat)) and float(rd[feat]) != 0.0
        )
        for feat in graph_impact_features
    }
    vectors = [tuple(rd.get(feat) for feat in FEATURE_NAMES) for rd in row_dicts]
    audit["unique_feature_vectors"] = len(set(vectors))
    audit["feature_vectors_vary"] = audit["unique_feature_vectors"] > 1

    # --- Constant (non-zero) features ---
    constant_features: List[Dict[str, Any]] = []
    for feat in FEATURE_NAMES:
        vals = [float(rd.get(feat, float("nan"))) for rd in row_dicts
                if not _is_nan(rd.get(feat))]
        if len(vals) > 1:
            if len(set(vals)) == 1:
                constant_features.append({"feature": feat, "value": vals[0]})
    audit["constant_nonzero_features"] = constant_features

    # --- Leakage check ---
    all_keys: set = set()
    for rd in row_dicts:
        all_keys.update(rd.keys())
    leaked = [f for f in LEAKAGE_FIELDS if f in all_keys]
    # high_risk is the label -- it's allowed in the dict, but not in FEATURE_NAMES
    audit["leakage_fields_in_dict"] = leaked
    audit["high_risk_in_feature_names"] = "high_risk" in FEATURE_NAMES

    # --- Schema validation ---
    schema_errors = validate_schema(row_dicts)
    audit["schema_ok"] = len(schema_errors) == 0
    audit["schema_errors"] = schema_errors

    # --- Duplicates ---
    pr_ids = [r.pr_id for r in rows]
    audit["n_duplicate_pr_ids"] = len(pr_ids) - len(set(pr_ids))

    # --- Numeric types ---
    type_errors: List[str] = []
    for feat in FEATURE_NAMES:
        for i, rd in enumerate(row_dicts):
            v = rd.get(feat)
            if v is not None and not _is_nan(v):
                try:
                    float(v)
                except (TypeError, ValueError):
                    type_errors.append(
                        f"Row {i} feature {feat}: {type(v).__name__} = {v!r}"
                    )
    audit["numeric_type_errors"] = type_errors[:10]  # Cap at 10

    # --- Analysis mode counts ---
    audit["fully_analyzed_prs"] = sum(1 for r in rows if r.analysis_mode == "full")
    audit["patch_only_prs"] = sum(1 for r in rows if r.analysis_mode == "patch_only")
    audit["skipped_analysis_prs"] = sum(1 for r in rows if r.analysis_mode == "none")

    # --- Errors ---
    errors_by_type: Dict[str, int] = {}
    for r in rows:
        if r.extraction_error:
            key = (
                r.extraction_error.split(":")[0]
                if ":" in r.extraction_error
                else r.extraction_error
            )
            errors_by_type[key] = errors_by_type.get(key, 0) + 1
    audit["extraction_errors"] = errors_by_type

    return audit


def format_audit_report(
    audit: Dict[str, Any],
    counters: Any,
    elapsed: float,
    checkout_enabled: bool,
) -> str:
    """Format audit dict into a readable report string."""
    lines = [
        "=" * 70,
        "PHASE 8B -- AIDEV PILOT AUDIT REPORT",
        "=" * 70,
        "",
        f"Elapsed: {elapsed:.1f}s",
        f"Checkout: {'ENABLED' if checkout_enabled else 'DISABLED (patch-only)'}",
        "",
        "--- FILTER PIPELINE ---",
        counters.summary(),
        "",
        "--- EXTRACTED ROWS ---",
        f"  Total rows extracted:  {audit['n_rows']}",
        f"  Checkout successes:    {audit['checkout']['success']}",
        f"  Checkout failures:     {audit['checkout']['failed']}",
        f"  Analysis modes:        {audit['checkout']['analysis_modes']}",
        f"  Fully analyzed PRs:    {audit['fully_analyzed_prs']}",
        f"  Patch-only PRs:        {audit['patch_only_prs']}",
        f"  Skipped analysis PRs:  {audit['skipped_analysis_prs']}",
        "",
        "--- LABEL DISTRIBUTION ---",
        f"  high_risk=1 (CHANGES_REQUESTED): {audit['label_distribution']['n_high_risk']}",
        f"  high_risk=0 (other):             {audit['label_distribution']['n_low_risk']}",
        f"  Positive fraction:               {audit['label_distribution']['pos_fraction']:.3f}",
        "",
        "--- FEATURE VALIDATION ---",
        f"  Feature count:         {audit['feature_count']} (expected 42: {audit['feature_count_ok']})",
        f"  Canonical order match: {audit['canonical_order_match']}",
        f"  Schema OK:             {audit['schema_ok']}",
        f"  high_risk in FEATURES: {audit['high_risk_in_feature_names']} (must be False)",
        f"  Duplicate PR IDs:      {audit['n_duplicate_pr_ids']} (must be 0)",
        f"  Numeric type errors:   {len(audit['numeric_type_errors'])} (must be 0)",
        f"  Unique feature vectors:{audit['unique_feature_vectors']}",
    ]

    if audit.get("schema_errors"):
        lines.append(f"  Schema errors: {audit['schema_errors']}")

    lines += [
        "",
        "--- LEAKAGE CHECK ---",
        f"  Leakage fields in row dict: {audit['leakage_fields_in_dict']}",
        "  NOTE: FeatureRow dataclass does not include review state/merged_at/closed_at.",
        "  The 'state' key above refers only to PR state (open/closed), which is",
        "  excluded from FEATURE_NAMES and used only for filtering, not features.",
    ]

    lines += [
        "",
        "--- FEATURE COMPLETENESS (n_missing / total, missing_frac) ---",
    ]
    for feat, info in audit["feature_completeness"].items():
        frac = info["missing_frac"]
        flag = " <<HIGH" if frac > 0.5 else ""
        lines.append(
            f"  {feat:<45} {info['n_missing']:>3}/{audit['n_rows']:>3}"
            f" ({frac:.2f}){flag}"
        )

    if audit["all_zero_features"]:
        lines += ["", "--- ALL-ZERO FEATURES (suspicious, check if expected) ---"]
        for f in audit["all_zero_features"]:
            lines.append(f"  {f}")

    if audit["constant_nonzero_features"]:
        lines += ["", "--- CONSTANT NON-ZERO FEATURES ---"]
        for item in audit["constant_nonzero_features"]:
            lines.append(f"  {item['feature']}: {item['value']}")

    if audit.get("extraction_errors"):
        lines += ["", "--- EXTRACTION ERRORS ---"]
        for k, v in audit["extraction_errors"].items():
            lines.append(f"  {k}: {v}")

    if audit.get("numeric_type_errors"):
        lines += ["", "--- NUMERIC TYPE ERRORS ---"]
        for e in audit["numeric_type_errors"]:
            lines.append(f"  {e}")

    lines += ["", "--- VERDICT ---"]
    ok = (
        audit["feature_count_ok"]
        and audit["canonical_order_match"]
        and audit["schema_ok"]
        and not audit["high_risk_in_feature_names"]
        and audit["n_duplicate_pr_ids"] == 0
        and len(audit["numeric_type_errors"]) == 0
        and audit["n_rows"] > 0
    )
    lines.append(f"  Dataset validation: {'PASSED' if ok else 'FAILED'}")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 25-PR real AIDev pilot and produce audit report."
    )
    parser.add_argument(
        "--no-checkout",
        action="store_true",
        help="Disable repository checkout (patch-only features only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for pilot Parquet.",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=25,
        help="Number of PRs to attempt (checkout validation: 5 through 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--git-timeout",
        type=int,
        default=120,
        help="Git operation timeout in seconds.",
    )
    parser.add_argument(
        "--clone-depth",
        type=int,
        default=50,
        help="Git clone depth.",
    )
    args = parser.parse_args()

    checkout_enabled = not args.no_checkout
    if not 1 <= args.pilot_size <= 25:
        parser.error("--pilot-size must be between 1 and 25")

    logger.info("=" * 60)
    logger.info(f"Phase 8B -- {args.pilot_size}-PR Real AIDev Pilot")
    logger.info(
        f"  Checkout: {'ENABLED' if checkout_enabled else 'DISABLED (patch-only)'}"
    )
    logger.info(f"  Output:   {args.output_dir}/aidev_pilot.parquet")
    logger.info(f"  Seed:     {args.seed}")
    logger.info("=" * 60)

    builder = AIDevDatasetBuilder(
        output_dir=args.output_dir,
        checkout_enabled=checkout_enabled,
        random_seed=args.seed,
        git_timeout=args.git_timeout,
        clone_depth=args.clone_depth,
        max_repos_on_disk=5,
    )

    t0 = time.time()
    rows = builder.build(
        pilot_mode=True,
        pilot_size=args.pilot_size,
    )
    elapsed = time.time() - t0

    logger.info(f"Extraction completed in {elapsed:.1f}s")

    if not rows:
        logger.error(
            "No rows extracted. Dataset is empty. "
            "Check filter counters above for blockers."
        )
        sys.exit(1)

    # Audit
    audit = audit_rows(rows)
    report = format_audit_report(audit, builder.counters, elapsed, checkout_enabled)
    print()
    print(report)

    # Save Parquet
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "aidev_pilot.parquet" if args.pilot_size == 25 else "aidev_checkout_validation.parquet"
    )
    saved_path = builder.save_parquet(rows, output_path)
    logger.info(f"Pilot dataset saved: {saved_path} ({len(rows)} rows)")

    # Quality check
    row_dicts = [r.to_dict() for r in rows]
    quality = run_quality_checks(row_dicts)
    logger.info(quality.summary())

    # Class balance
    labels = [int(r.high_risk) for r in rows]
    balance = compute_class_balance(labels)
    logger.info(
        f"Class balance: {balance['n_positive']} HIGH / {balance['n_negative']} LOW "
        f"(pos_frac={balance['pos_fraction']:.3f})"
    )

    # Save JSON stats
    stats_path = output_dir / (
        "aidev_pilot_audit.json" if args.pilot_size == 25 else "aidev_checkout_validation_audit.json"
    )
    stats = {
        "n_rows": len(rows),
        "elapsed_seconds": round(elapsed, 1),
        "checkout_enabled": checkout_enabled,
        "random_seed": args.seed,
        "counters": {
            "considered": builder.counters.considered,
            "excluded_language": builder.counters.excluded_language,
            "excluded_license": builder.counters.excluded_license,
            "excluded_fork": builder.counters.excluded_fork,
            "excluded_no_patch": builder.counters.excluded_no_patch,
            "excluded_no_review": builder.counters.excluded_no_review,
            "excluded_not_closed": builder.counters.excluded_not_closed,
            "excluded_revert": builder.counters.excluded_revert,
            "checkout_failed": builder.counters.checkout_failed,
            "analysis_failed": builder.counters.analysis_failed,
            "successfully_extracted": builder.counters.successfully_extracted,
        },
        "class_balance": balance,
        "quality_passed": quality.passed,
        "quality_warnings": quality.warnings,
        "audit_summary": {
            k: v
            for k, v in audit.items()
            if k != "feature_completeness"
        },
        "feature_completeness": audit.get("feature_completeness", {}),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Audit JSON saved: {stats_path}")

    # Determine pass/fail
    passed = (
        audit["feature_count_ok"]
        and audit["canonical_order_match"]
        and audit["schema_ok"]
        and not audit["high_risk_in_feature_names"]
        and audit["n_duplicate_pr_ids"] == 0
        and len(audit["numeric_type_errors"]) == 0
        and audit["n_rows"] > 0
    )

    if passed:
        logger.info("PILOT VALIDATION: PASSED")
        logger.info("Next step: Run with --target 300 for development dataset.")
    else:
        logger.warning("PILOT VALIDATION: FAILED -- review audit report above.")
        if audit["n_rows"] < 25:
            logger.warning(
                f"Only {audit['n_rows']} rows extracted (expected 25). "
                "Review filter counters for blockers."
            )


if __name__ == "__main__":
    main()
