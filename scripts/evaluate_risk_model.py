"""
scripts/evaluate_risk_model.py — Phase 8B Model Evaluation Script

Usage:
    python scripts/evaluate_risk_model.py \\
        --model models/risk_model.cbm \\
        --dataset data/processed/aidev_features_3000.parquet \\
        --shap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained CatBoost risk model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to saved CatBoost model (.cbm).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to Parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs"),
        help="Output directory for evaluation report (default: docs/).",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Generate SHAP global importance summary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    if not args.dataset.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Load model
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.error("catboost is required. pip install catboost")
        sys.exit(1)

    import numpy as np
    import pyarrow.parquet as pq

    model = CatBoostClassifier()
    model.load_model(str(args.model))
    logger.info(f"Model loaded: {args.model}")

    # Load dataset
    table = pq.read_table(str(args.dataset))
    rows = table.to_pylist()
    logger.info(f"Dataset loaded: {len(rows)} rows")

    from analyzer.ml.preprocessing import build_feature_matrix, get_repo_groups, CANONICAL_FEATURES

    X, y, feature_names = build_feature_matrix(rows)
    groups = get_repo_groups(rows)

    logger.info(f"Test set: {len(y)} samples, {int(np.sum(y))} positive")

    # Evaluate
    from analyzer.ml.evaluate import evaluate_model, save_evaluation_result

    result = evaluate_model(
        model, X, y,
        feature_names=feature_names,
        threshold=args.threshold,
    )
    logger.info(result.summary())

    # Save JSON result
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = args.output_dir / "phase8b_eval_result.json"
    save_evaluation_result(result, eval_path)
    logger.info(f"Evaluation result saved: {eval_path}")

    # SHAP global importance
    if args.shap:
        try:
            from analyzer.ml.explain import global_shap_summary, print_global_importance

            logger.info("Generating SHAP global importance...")
            summary = global_shap_summary(model, X, feature_names=feature_names)
            importance_str = print_global_importance(summary)
            logger.info(f"\n{importance_str}")

            # Save
            shap_path = args.output_dir / "phase8b_shap_importance.json"
            with open(shap_path, "w") as f:
                json.dump({
                    "mean_abs_shap": summary.mean_abs_shap,
                    "feature_importance_rank": summary.feature_importance_rank,
                    "n_samples": summary.n_samples,
                    "expected_value": summary.expected_value,
                }, f, indent=2)
            logger.info(f"SHAP importance saved: {shap_path}")
        except Exception as exc:
            logger.warning(f"SHAP failed: {exc}")

    # Print high-risk recall specifically
    logger.info("")
    logger.info("=" * 50)
    logger.info("KEY METRIC: High-Risk Recall (most important for deployment risk detection)")
    logger.info(f"  High-Risk Recall : {result.high_risk_recall:.4f}")
    logger.info(f"  ROC-AUC          : {result.roc_auc:.4f}")
    logger.info(f"  PR-AUC           : {result.pr_auc:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
