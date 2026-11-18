"""
analyzer/ml/train.py — Phase 8B CatBoost Training

Design decisions:
- CatBoost only (no XGBoost / LightGBM).
- Repository-aware cross-validation (GroupKFold) to prevent repo leakage.
- Class weights are computed from actual dataset distribution, not hard-coded.
- Hyperparameter tuning is optional (centralised in config).
- The trained model and metadata are saved to disk.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — avoid ImportError at module load time when catboost/sklearn
# are not yet installed (tests can mock them).
# ---------------------------------------------------------------------------

try:
    from catboost import CatBoostClassifier, Pool  # type: ignore[import-untyped]
    _CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    Pool = None

try:
    from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    GroupKFold = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Result of a single CatBoost training run."""

    model: Any  # CatBoostClassifier instance
    feature_names: List[str]
    train_auc: float
    val_auc: float
    n_train: int
    n_val: int
    class_weights: Dict[int, float]
    params: Dict[str, Any]
    model_path: Optional[Path] = None
    metadata_path: Optional[Path] = None


@dataclass
class CVResult:
    """Result of cross-validation training."""

    fold_aucs: List[float] = field(default_factory=list)
    fold_f1s: List[float] = field(default_factory=list)
    mean_auc: float = 0.0
    std_auc: float = 0.0
    mean_f1: float = 0.0
    std_f1: float = 0.0
    n_folds: int = 0


# ---------------------------------------------------------------------------
# Repository-aware splitter
# ---------------------------------------------------------------------------


class RepositoryAwareSplitter:
    """
    Wraps sklearn GroupKFold to enforce repository-level train/val splits.

    Ensures that all PRs from a given repository appear in either train OR
    validation, never both. This prevents the model from exploiting
    repository-specific patterns as shortcuts to label prediction.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate repository-aware train/validation index splits.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label vector (n_samples,).
            groups: Integer group (repo) array (n_samples,).

        Returns:
            List of (train_idx, val_idx) tuples.
        """
        if not _SKLEARN_AVAILABLE or GroupKFold is None:
            raise ImportError(
                "scikit-learn is required for RepositoryAwareSplitter. "
                "Install with: pip install scikit-learn"
            )
        n_unique_groups = len(np.unique(groups))
        actual_splits = min(self.n_splits, n_unique_groups)
        if actual_splits < 2:
            raise ValueError(
                f"Not enough unique repositories ({n_unique_groups}) for "
                f"{self.n_splits}-fold cross-validation. Need at least 2."
            )
        gkf = GroupKFold(n_splits=actual_splits)
        return list(gkf.split(X, y, groups))

    def get_single_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        val_fraction: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single train/validation split respecting repository boundaries.

        Args:
            X, y, groups: As in split().
            val_fraction: Approximate fraction of data to use for validation.

        Returns:
            (train_idx, val_idx)
        """
        n_splits = max(2, round(1.0 / val_fraction))
        if not _SKLEARN_AVAILABLE or GroupKFold is None:
            raise ImportError("scikit-learn is required.")
        n_unique_groups = len(np.unique(groups))
        actual_splits = min(n_splits, n_unique_groups)
        if actual_splits < 2:
            raise ValueError(
                f"Not enough unique repos ({n_unique_groups}) for a "
                f"{actual_splits}-fold split."
            )
        gkf = GroupKFold(n_splits=actual_splits)
        splits = list(gkf.split(X, y, groups))
        # Use the last fold as validation
        last: Tuple[np.ndarray, np.ndarray] = splits[-1]
        return last


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights inversely proportional to class frequency.

    The positive class receives weight = n_negative / n_positive.
    This is the standard sklearn 'balanced' strategy.

    Args:
        y: Binary label array.

    Returns:
        Dict {0: 1.0, 1: n_neg/n_pos} or {0: 1.0, 1: 1.0} if balanced.
    """
    n = len(y)
    if n == 0:
        return {0: 1.0, 1: 1.0}
    n_pos = int(np.sum(y))
    n_neg = n - n_pos
    if n_pos == 0:
        logger.warning("No positive samples — setting equal weights.")
        return {0: 1.0, 1: 1.0}
    if n_neg == 0:
        logger.warning("No negative samples — setting equal weights.")
        return {0: 1.0, 1: 1.0}
    ratio = n_neg / n_pos
    logger.info(
        f"Class balance: {n_pos} positive / {n_neg} negative  "
        f"→ weight ratio 1:{ratio:.2f}"
    )
    return {0: 1.0, 1: round(ratio, 4)}


# ---------------------------------------------------------------------------
# Hyperparameter tuner
# ---------------------------------------------------------------------------


class HyperparameterTuner:
    """
    Simple grid-search tuner for CatBoost using repository-aware CV.

    Only tunes learning_rate and depth to keep runtime manageable.
    """

    GRID: Dict[str, List[Any]] = {
        "learning_rate": [0.05, 0.1, 0.2],
        "depth": [4, 6, 8],
    }

    def __init__(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_params = base_params or {}
        self.n_splits = n_splits
        self.random_state = random_state

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run grid search and return the best hyperparameters.

        Args:
            X: Feature matrix.
            y: Labels.
            groups: Repo group array.
            feature_names: Optional feature names for CatBoost Pool.

        Returns:
            Best parameter dict (merged with base_params).
        """
        if not _CATBOOST_AVAILABLE or CatBoostClassifier is None:
            raise ImportError("catboost is required for HyperparameterTuner.")

        splitter = RepositoryAwareSplitter(n_splits=self.n_splits)
        splits = splitter.split(X, y, groups)
        class_weights = compute_class_weights(y)

        best_auc = -1.0
        best_params: Dict[str, Any] = {}

        for lr in self.GRID["learning_rate"]:
            for depth in self.GRID["depth"]:
                params = {**self.base_params, "learning_rate": lr, "depth": depth,
                          "random_seed": self.random_state, "verbose": False,
                          "allow_writing_files": False}
                fold_aucs: List[float] = []

                for train_idx, val_idx in splits:
                    model = CatBoostClassifier(**params)
                    train_pool = Pool(
                        X[train_idx], y[train_idx],
                        feature_names=feature_names,
                        weight=[class_weights[int(yi)] for yi in y[train_idx]],
                    )
                    val_pool = Pool(
                        X[val_idx], y[val_idx],
                        feature_names=feature_names,
                    )
                    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
                    proba = model.predict_proba(X[val_idx])[:, 1]
                    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
                    auc = float(roc_auc_score(y[val_idx], proba))
                    fold_aucs.append(auc)

                mean_auc = float(np.mean(fold_aucs))
                logger.info(f"lr={lr}, depth={depth} → mean AUC={mean_auc:.4f}")
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {"learning_rate": lr, "depth": depth}

        logger.info(f"Best params: {best_params} (AUC={best_auc:.4f})")
        return {**self.base_params, **best_params}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_catboost(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    val_fraction: float = 0.2,
    output_dir: Optional[Path] = None,
    model_name: str = "risk_model",
) -> TrainingResult:
    """
    Train a CatBoost classifier with repository-aware train/validation split.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Binary label array (n_samples,).
        groups: Repo group array for GroupKFold (n_samples,).
        feature_names: Optional list of feature names (length n_features).
        params: CatBoost hyperparameters (defaults from dataset/config.py).
        val_fraction: Fraction of data for validation (repo-aware).
        output_dir: Directory to save model and metadata.
        model_name: Base name for saved files.

    Returns:
        TrainingResult with trained model and metrics.
    """
    if not _CATBOOST_AVAILABLE or CatBoostClassifier is None or Pool is None:
        raise ImportError(
            "catboost is required. Install with: pip install catboost"
        )

    # Default parameters
    default_params: Dict[str, Any] = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    if params:
        default_params.update(params)

    # Compute class weights from actual distribution
    class_weights = compute_class_weights(y)

    # Repository-aware split
    splitter = RepositoryAwareSplitter()
    train_idx, val_idx = splitter.get_single_split(X, y, groups, val_fraction)

    logger.info(
        f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples"
    )

    # Build CatBoost pools
    sample_weights_train = np.array(
        [class_weights[int(yi)] for yi in y[train_idx]]
    )
    train_pool = Pool(
        X[train_idx],
        y[train_idx],
        feature_names=feature_names,
        weight=sample_weights_train,
    )
    val_pool = Pool(
        X[val_idx],
        y[val_idx],
        feature_names=feature_names,
    )

    # Train
    model = CatBoostClassifier(**default_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
    )

    # AUC on train and val
    from sklearn.metrics import roc_auc_score
    train_proba = model.predict_proba(X[train_idx])[:, 1]
    val_proba = model.predict_proba(X[val_idx])[:, 1]
    train_auc = float(roc_auc_score(y[train_idx], train_proba))
    val_auc = float(roc_auc_score(y[val_idx], val_proba))

    logger.info(f"Train AUC: {train_auc:.4f}  Val AUC: {val_auc:.4f}")

    # Save model
    model_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}.cbm"
        model.save_model(str(model_path))

        metadata = {
            "feature_names": feature_names or [],
            "train_auc": train_auc,
            "val_auc": val_auc,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "class_weights": class_weights,
            "params": default_params,
        }
        metadata_path = output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")

    return TrainingResult(
        model=model,
        feature_names=feature_names or [],
        train_auc=train_auc,
        val_auc=val_auc,
        n_train=len(train_idx),
        n_val=len(val_idx),
        class_weights=class_weights,
        params=default_params,
        model_path=model_path,
        metadata_path=metadata_path,
    )
