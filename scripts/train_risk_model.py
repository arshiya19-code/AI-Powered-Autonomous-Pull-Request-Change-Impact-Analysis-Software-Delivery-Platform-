"""
scripts/train_risk_model.py — Phase 8B CatBoost Training Script

Usage:
    python scripts/train_risk_model.py --dataset data/processed/aidev_features_3000.parquet
    python scripts/train_risk_model.py --dataset data/processed/aidev_pilot.parquet --tune
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
        description="Train CatBoost risk model on AIDev feature dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to Parquet dataset (from build_aidev_dataset.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model (default: models/).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="risk_model",
        help="Model filename prefix (default: risk_model).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run grid-search hyperparameter tuning before training.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run repository-aware cross-validation after training.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Load dataset
    try:
        import pyarrow.parquet as pq
        import numpy as np
    except ImportError:
        logger.error("pyarrow and numpy are required. pip install pyarrow numpy")
        sys.exit(1)

    logger.info(f"Loading dataset from {args.dataset}...")
    table = pq.read_table(str(args.dataset))
    rows = table.to_pylist()
    logger.info(f"Loaded {len(rows)} rows")

    if len(rows) < 10:
        logger.error("Dataset too small for training. Need at least 10 rows.")
        sys.exit(1)

    # Build feature matrix
    from analyzer.ml.preprocessing import build_feature_matrix, get_repo_groups, run_quality_checks, CANONICAL_FEATURES

    quality = run_quality_checks(rows)
    logger.info(quality.summary())
    if not quality.passed:
        logger.warning("Quality checks did not all pass — proceeding anyway.")

    X, y, feature_names = build_feature_matrix(rows)
    groups = get_repo_groups(rows)

    logger.info(f"Feature matrix: {X.shape}  Labels: {y.shape}")
    logger.info(f"  Positive (high_risk=1): {int(np.sum(y))}")
    logger.info(f"  Negative (high_risk=0): {int(np.sum(y == 0))}")

    # Optional hyperparameter tuning
    params = None
    if args.tune:
        logger.info("Running hyperparameter tuning (grid search)...")
        from analyzer.ml.train import HyperparameterTuner
        tuner = HyperparameterTuner(n_splits=args.n_splits, random_state=args.seed)
        params = tuner.tune(X, y, groups, feature_names=feature_names)
        logger.info(f"Best params: {params}")

    # Train
    logger.info("Training CatBoost model...")
    from analyzer.ml.train import train_catboost
    result = train_catboost(
        X, y, groups,
        feature_names=feature_names,
        params=params,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    logger.info(f"Train AUC: {result.train_auc:.4f}  Val AUC: {result.val_auc:.4f}")
    logger.info(f"Class weights used: {result.class_weights}")

    if result.model_path:
        logger.info(f"Model saved: {result.model_path}")

    # Optional cross-validation
    if args.cv:
        logger.info(f"Running {args.n_splits}-fold repository-aware CV...")
        from analyzer.ml.evaluate import cross_validate_by_repo
        cv_result = cross_validate_by_repo(
            result.params, X, y, groups,
            feature_names=feature_names,
            n_splits=args.n_splits,
        )
        logger.info(
            f"CV AUC: {cv_result.cv_mean_auc:.4f} ± {cv_result.cv_std_auc:.4f}"
            f"  ({cv_result.cv_n_folds} folds)"
        )
        # Save CV result
        cv_path = args.output_dir / f"{args.model_name}_cv.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(cv_path, "w") as f:
            json.dump({
                "cv_mean_auc": cv_result.cv_mean_auc,
                "cv_std_auc": cv_result.cv_std_auc,
                "cv_fold_aucs": cv_result.cv_fold_aucs,
                "n_folds": cv_result.cv_n_folds,
            }, f, indent=2)
        logger.info(f"CV results saved: {cv_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
