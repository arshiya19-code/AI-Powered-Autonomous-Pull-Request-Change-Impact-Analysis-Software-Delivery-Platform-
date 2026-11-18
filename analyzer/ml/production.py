"""Production-model preparation and inference contracts.

This module deliberately separates *validation* from training.  The Phase 7F
producer may still be writing the parquet file, so callers can validate a
candidate without opening or modifying any active cache/checkpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset.config import ALL_FEATURES, LEAKAGE_BLOCKED_FIELDS


LABELS = {"HIGH", "MEDIUM", "LOW_OBSERVED"}
EXCLUDED_LABELS = {"UNKNOWN", "CONFLICTED"}


@dataclass(frozen=True)
class DatasetValidation:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    rows: int
    usable_rows: int
    label_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors), "warnings": list(self.warnings),
                "rows": self.rows, "usable_rows": self.usable_rows,
                "label_distribution": self.label_distribution}


def validate_supervised_frame(frame: Any, metadata: Mapping[str, Any] | None = None) -> DatasetValidation:
    """Strictly validate the final supervised 42-feature frame.

    UNKNOWN/CONFLICTED rows are excluded by explicit policy; they are never
    coerced into a negative class.  No writes occur here.
    """
    errors: list[str] = []
    warnings: list[str] = []
    columns = list(getattr(frame, "columns", []))
    missing = [f for f in ALL_FEATURES if f not in columns]
    if missing: errors.append(f"Missing canonical features: {missing}")
    if len(set(columns)) != len(columns): errors.append("Duplicate dataframe columns detected")
    label_col = "label" if "label" in columns else "risk_label" if "risk_label" in columns else "outcome_label" if "outcome_label" in columns else None
    if not label_col: errors.append("No supervised label column found")
    identity_cols = [c for c in ("pr_id", "pr_identity", "pull_request_id") if c in columns]
    if identity_cols:
        ids = frame[identity_cols[0]]
        if bool(ids.duplicated().any()): errors.append(f"Duplicate PR identity in {identity_cols[0]}")
    if any(c in columns for c in LEAKAGE_BLOCKED_FIELDS): errors.append("Leakage fields present in model input")
    distribution: dict[str, int] = {}
    usable = 0
    if label_col:
        vals = frame[label_col].astype(str).str.upper()
        distribution = {str(k): int(v) for k, v in vals.value_counts(dropna=False).items()}
        invalid = sorted(set(vals) - LABELS - EXCLUDED_LABELS)
        if invalid: errors.append(f"Invalid labels: {invalid}")
        usable = int((~vals.isin(EXCLUDED_LABELS)).sum())
        if not (set(vals) & {"HIGH", "LOW_OBSERVED"}): warnings.append("Binary HIGH/LOW_OBSERVED task has no observed classes")
        if "HIGH" not in set(vals) or "LOW_OBSERVED" not in set(vals): warnings.append("Intended binary task is missing one class")
    if metadata:
        provenance = json.dumps(dict(metadata), sort_keys=True).lower()
        if "phase7d" not in provenance or "outcome" not in provenance:
            errors.append("Genuine Phase 7D outcome provenance is missing")
    else:
        errors.append("Dataset provenance metadata is required")
    return DatasetValidation(not errors, tuple(errors), tuple(warnings), len(frame), usable, distribution)


def leakage_safe_split(frame: Any, *, group_column: str = "repository", time_column: str | None = None, val_fraction: float = .2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return repository-disjoint indices, using time ordering when available."""
    n = len(frame)
    if n < 2: raise ValueError("At least two supervised rows are required")
    groups = frame[group_column].fillna("__missing__").astype(str) if group_column in frame.columns else None
    if groups is None: raise ValueError(f"Leakage-safe split requires {group_column!r}")
    unique = list(dict.fromkeys(groups.tolist()))
    if len(unique) < 2: raise ValueError("At least two repository groups are required")
    if time_column and time_column in frame.columns:
        order = np.argsort(frame[time_column].fillna("").astype(str).to_numpy())
        cut = max(1, min(n - 1, int(round(n * (1 - val_fraction)))))
        train, val = order[:cut], order[cut:]
        if set(groups.iloc[train]) & set(groups.iloc[val]):
            # Fall through to group holdout; temporal ordering is secondary to leakage prevention.
            pass
        else: return train, val, {"strategy": "temporal_holdout", "group_column": group_column, "time_column": time_column}
    val_groups = set(unique[-max(1, int(round(len(unique) * val_fraction))):])
    val = np.array([i for i, g in enumerate(groups) if g in val_groups], dtype=int)
    train = np.array([i for i, g in enumerate(groups) if g not in val_groups], dtype=int)
    if not len(train) or not len(val): raise ValueError("Unable to construct non-empty repository holdout")
    return train, val, {"strategy": "repository_group_holdout", "group_column": group_column, "validation_groups": sorted(val_groups), "random_state": random_state}


def production_artifact_dir(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / "artifacts" / "models" / "deployment_risk"


def training_not_ready_diagnostic(frame: Any, validation: DatasetValidation) -> dict[str, Any]:
    vals = validation.label_distribution
    binary = {"HIGH": vals.get("HIGH", 0), "LOW_OBSERVED": vals.get("LOW_OBSERVED", 0)}
    if min(binary.values()) == 0: reason = "binary_task_missing_class"
    else: reason = "ready_for_training"
    return {"status": "TRAINING_BLOCKED" if reason != "ready_for_training" else "READY", "reason": reason,
            "label_distribution": vals, "validation": validation.to_dict(), "generated_at": datetime.now(timezone.utc).isoformat()}


def load_final_dataset(path: str | Path = "data/processed/aidev_supervised_42.parquet") -> tuple[Any, dict[str, Any], DatasetValidation]:
    """Read the final parquet and its adjacent provenance metadata read-only."""
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Final dataset not found: {path}")
    try:
        import pandas as pd
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to read supervised parquet: {exc}") from exc
    candidates = [path.with_name(path.stem + "_metadata.json"), path.with_suffix(".metadata.json")]
    metadata_path = next((p for p in candidates if p.exists()), None)
    metadata = json.loads(metadata_path.read_text()) if metadata_path else {}
    validation = validate_supervised_frame(frame, metadata)
    return frame, metadata, validation


def train_final_catboost(*, dataset_path: str | Path = "data/processed/aidev_supervised_42.parquet", output_dir: str | Path | None = None) -> Any:
    """Train only when explicitly called after the producer is complete.

    This function is intentionally not called by the API or startup path.
    """
    frame, metadata, validation = load_final_dataset(dataset_path)
    if not validation.valid: raise ValueError(json.dumps(training_not_ready_diagnostic(frame, validation), indent=2))
    labels = frame["label" if "label" in frame.columns else "risk_label"].astype(str).str.upper()
    mask = labels.isin({"HIGH", "LOW_OBSERVED"})
    if labels[mask].nunique() < 2: raise ValueError(json.dumps(training_not_ready_diagnostic(frame, validation), indent=2))
    from .train import train_catboost
    X = frame.loc[mask, ALL_FEATURES].to_numpy(dtype=float)
    y = np.array([1 if x == "HIGH" else 0 for x in labels[mask]], dtype=int)
    group_column = "repository" if "repository" in frame.columns else "repo_id"
    groups = frame.loc[mask, group_column].fillna("__missing__").astype(str).factorize()[0]
    return train_catboost(X, y, groups, feature_names=list(ALL_FEATURES), output_dir=Path(output_dir) if output_dir else production_artifact_dir())
