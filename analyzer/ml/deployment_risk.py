"""Train and evaluate CatBoost on the Phase 7E/7F supervised 42-feature dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyzer.ml.evaluate import calibration_analysis, evaluate_model
from analyzer.ml.preprocessing import CANONICAL_FEATURES
from analyzer.ml.train import compute_class_weights
from dataset.phase7c_contract import FEATURE_SCHEMA_VERSION, LabelState
from dataset.phase7e_supervised import DATASET_VERSION

MODEL_VERSION = "phase7ef-catboost-v1"
MIN_POSITIVE_FOR_TRAINING = 10
MIN_NEGATIVE_FOR_TRAINING = 30
MIN_POSITIVE_FOR_RELIABLE_EVAL = 20


@dataclass(frozen=True)
class Sufficiency:
    trainable: bool
    reliable_evaluation: bool
    reason: str
    n_positive: int
    n_negative: int


def split_matrices(rows: Sequence[dict[str, Any]], *, positive_label: str = LabelState.HIGH.value) -> dict[str, Any]:
    """Build binary HIGH-vs-LOW matrices. MEDIUM is held out of the binary task."""
    usable = [row for row in rows if row.get("risk_label") in {LabelState.HIGH.value, LabelState.LOW_OBSERVED.value}]
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in usable:
        by_split.setdefault(str(row.get("split") or "train"), []).append(row)

    def _xy(items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([[float(item[name]) for name in CANONICAL_FEATURES] for item in items], dtype=np.float64) if items else np.zeros((0, 42), dtype=np.float64)
        y = np.array([int(item.get("risk_label") == positive_label) for item in items], dtype=np.int32)
        return x, y

    x_train, y_train = _xy(by_split.get("train", []))
    x_val, y_val = _xy(by_split.get("validation", []))
    x_test, y_test = _xy(by_split.get("test", []))
    groups = np.array([hash(item["repo_full_name"]) % (2**31) for item in by_split.get("train", [])], dtype=np.int64)
    return {
        "X_train": x_train, "y_train": y_train,
        "X_val": x_val, "y_val": y_val,
        "X_test": x_test, "y_test": y_test,
        "groups_train": groups,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "train_rows": by_split.get("train", []),
        "val_rows": by_split.get("validation", []),
        "test_rows": by_split.get("test", []),
    }


def assess_sufficiency(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Sufficiency:
    y_all = np.concatenate([y_train, y_val, y_test]) if len(y_train) or len(y_val) or len(y_test) else np.zeros(0, dtype=np.int32)
    n_pos = int(np.sum(y_all))
    n_neg = int(len(y_all) - n_pos)
    train_pos, train_neg = int(np.sum(y_train)), int(len(y_train) - np.sum(y_train))
    test_pos, test_neg = int(np.sum(y_test)), int(len(y_test) - np.sum(y_test))
    if n_pos == 0:
        return Sufficiency(False, False, "zero_high_labels", n_pos, n_neg)
    if n_pos < MIN_POSITIVE_FOR_TRAINING or n_neg < MIN_NEGATIVE_FOR_TRAINING:
        return Sufficiency(False, False, "minority_or_majority_below_training_floor", n_pos, n_neg)
    if train_pos == 0 or train_neg == 0:
        return Sufficiency(False, False, "train_split_missing_a_class", n_pos, n_neg)
    if test_pos == 0 or test_neg == 0:
        return Sufficiency(True, False, "test_split_missing_a_class", n_pos, n_neg)
    reliable = n_pos >= MIN_POSITIVE_FOR_RELIABLE_EVAL
    return Sufficiency(True, reliable, "ok" if reliable else "positive_class_below_reliable_eval_floor", n_pos, n_neg)


def _majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    majority = int(np.round(np.mean(y_train) >= 0.5)) if len(y_train) else 0
    pred = np.full_like(y_test, majority)
    correct = float(np.mean(pred == y_test)) if len(y_test) else float("nan")
    return {"strategy": "majority_class", "predicted_class": majority, "accuracy": correct}


def _logistic_baseline(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"strategy": "logistic_regression", "available": False}
    if len(np.unique(y_train)) < 2:
        return {"strategy": "logistic_regression", "available": False, "reason": "single_class_train"}
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    n_pos = int(np.sum(y_test))
    n_neg = len(y_test) - n_pos
    return {
        "strategy": "l2_logistic_regression",
        "available": True,
        "accuracy": float(np.mean(pred == y_test)) if len(y_test) else float("nan"),
        "roc_auc": float(roc_auc_score(y_test, proba)) if n_pos and n_neg else float("nan"),
        "pr_auc": float(average_precision_score(y_test, proba)) if n_pos else float("nan"),
    }


class DeploymentRiskModel:
    """Inference wrapper that enforces canonical 42-feature order."""

    def __init__(self, model: Any, *, model_version: str, feature_schema_version: str, threshold: float = 0.5) -> None:
        self.model = model
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version
        self.threshold = threshold

    def predict(self, features: dict[str, float] | np.ndarray) -> dict[str, Any]:
        if isinstance(features, dict):
            missing = [name for name in CANONICAL_FEATURES if name not in features]
            extra = [name for name in features if name not in CANONICAL_FEATURES and name in {"risk_label", "is_high"}]
            if missing:
                raise ValueError(f"feature schema mismatch: missing {missing}")
            vector = np.array([[float(features[name]) for name in CANONICAL_FEATURES]], dtype=np.float64)
        else:
            array = np.asarray(features, dtype=np.float64)
            vector = array.reshape(1, -1) if array.ndim == 1 else array
            if vector.shape[1] != 42:
                raise ValueError(f"feature schema mismatch: expected 42 columns, got {vector.shape[1]}")
        proba = self.model.predict_proba(vector)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("inference output does not have the expected probability shape")
        risk_probability = float(proba[0, 1])
        if not (0.0 <= risk_probability <= 1.0):
            raise ValueError("probabilities are not valid")
        predicted = LabelState.HIGH.value if risk_probability >= self.threshold else LabelState.LOW_OBSERVED.value
        return {
            "risk_probability": risk_probability,
            "risk_score": round(100.0 * risk_probability, 4),
            "predicted_risk_class": predicted,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
        }


def train_deployment_risk_model(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    random_seed: int = 42,
    iterations: int = 400,
) -> dict[str, Any]:
    from catboost import CatBoostClassifier, Pool

    matrices = split_matrices(rows)
    sufficiency = assess_sufficiency(matrices["y_train"], matrices["y_val"], matrices["y_test"])
    report: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_definition": "binary is_high: HIGH vs LOW_OBSERVED under phase7d-outcome-v1; MEDIUM excluded from the binary task",
        "training_rows": matrices["n_train"],
        "validation_rows": matrices["n_val"],
        "test_rows": matrices["n_test"],
        "feature_names": list(CANONICAL_FEATURES),
        "language_distribution": _count(rows, "primary_language"),
        "class_distribution": _count(rows, "risk_label"),
        "split_strategy": "repository_then_temporal",
        "random_seed": random_seed,
        "sufficiency": sufficiency.__dict__,
        "trained": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not sufficiency.trainable:
        report["limitations"] = [
            "STATISTICALLY INSUFFICIENT FOR RELIABLE RISK-MODEL EVALUATION" if sufficiency.n_positive < MIN_POSITIVE_FOR_RELIABLE_EVAL else sufficiency.reason,
            f"Minimum legitimate requirement: ≥{MIN_POSITIVE_FOR_TRAINING} HIGH and ≥{MIN_NEGATIVE_FOR_TRAINING} LOW_OBSERVED in the binary task, with both classes in train; ≥{MIN_POSITIVE_FOR_RELIABLE_EVAL} HIGH for reliable evaluation.",
        ]
        (output_dir / "deployment_risk_model_metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return report

    x_train, y_train = matrices["X_train"], matrices["y_train"]
    x_val, y_val = matrices["X_val"], matrices["y_val"]
    x_test, y_test = matrices["X_test"], matrices["y_test"]
    class_weights = compute_class_weights(y_train)
    params = {
        "iterations": iterations,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
        "class_weights": [class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
    }
    model = CatBoostClassifier(**params)
    train_pool = Pool(x_train, y_train, feature_names=list(CANONICAL_FEATURES))
    eval_set = Pool(x_val, y_val, feature_names=list(CANONICAL_FEATURES)) if len(y_val) else None
    if eval_set is not None and len(np.unique(y_val)) > 1:
        model.fit(train_pool, eval_set=eval_set, early_stopping_rounds=50)
    else:
        model.fit(train_pool)
    wrapper = DeploymentRiskModel(model, model_version=MODEL_VERSION, feature_schema_version=FEATURE_SCHEMA_VERSION)
    eval_result = evaluate_model(model, x_test if len(y_test) else x_val, y_test if len(y_test) else y_val, feature_names=list(CANONICAL_FEATURES))
    eval_x = x_test if len(y_test) else x_val
    eval_y = y_test if len(y_test) else y_val
    proba = model.predict_proba(eval_x)[:, 1]
    calibration = calibration_analysis(eval_y, proba) if len(np.unique(eval_y)) > 1 else {}
    model_path = output_dir / "deployment_risk_model.cbm"
    model.save_model(str(model_path))
    loaded = CatBoostClassifier()
    loaded.load_model(str(model_path))
    reload_proba = loaded.predict_proba(eval_x)[:, 1]
    identical = bool(np.allclose(proba, reload_proba))
    sample = {name: float(eval_x[0, index]) for index, name in enumerate(CANONICAL_FEATURES)} if len(eval_x) else {name: 0.0 for name in CANONICAL_FEATURES}
    inference = wrapper.predict(sample)
    import catboost
    report.update({
        "trained": True,
        "model_artifact": str(model_path),
        "imbalance_strategy": "catboost_class_weights_from_train_distribution",
        "catboost_configuration": {**params, "catboost_version": getattr(catboost, "__version__", "unknown")},
        "class_weights": class_weights,
        "baselines": {
            "majority": _majority_baseline(y_train, eval_y),
            "logistic_regression": _logistic_baseline(x_train, y_train, eval_x, eval_y),
        },
        "metrics": eval_result.to_dict(),
        "calibration": calibration,
        "serialization": {
            "saved": True,
            "loaded_predictions_identical": identical,
            "inference_sample": inference,
        },
        "limitations": [] if sufficiency.reliable_evaluation else [
            "STATISTICALLY INSUFFICIENT FOR RELIABLE RISK-MODEL EVALUATION",
            sufficiency.reason,
        ],
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    })
    (output_dir / "deployment_risk_model_metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


def _count(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    from collections import Counter
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))
