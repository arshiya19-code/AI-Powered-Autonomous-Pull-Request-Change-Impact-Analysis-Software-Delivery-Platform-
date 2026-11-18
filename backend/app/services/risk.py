from __future__ import annotations
from pathlib import Path
from typing import Mapping
from backend.app.schemas.models import RiskPrediction
from dataset.config import CATBOOST_MODEL_FEATURES
import json
import os
import math

class RiskPredictionService:
    """CatBoost loading boundary; intentionally does not train or fabricate."""
    def __init__(self, artifact_path: str | None = None):
        configured = artifact_path or os.getenv("RISK_MODEL_PATH")
        self.artifact_path = Path(configured) if configured else Path("models/risk_model/catboost_apachejit.cbm")
        self.metadata_path = self.artifact_path.with_name(self.artifact_path.stem + "_metadata.json")
        self._model = None
        self._metadata = {}
        if self.artifact_path.exists():
            try:
                from catboost import CatBoostClassifier
                self._model = CatBoostClassifier(); self._model.load_model(str(self.artifact_path))
                if self.metadata_path.exists(): self._metadata = json.loads(self.metadata_path.read_text())
            except Exception:
                self._model = None

    @property
    def status(self) -> str:
        return "available" if self._model is not None else "unavailable"

    def predict(self, features: Mapping[str, float]) -> RiskPrediction:
        names = list(self._metadata.get("catboost_model_features") or self._metadata.get("feature_names") or CATBOOST_MODEL_FEATURES)
        model_name = self._metadata.get("model_name") or self.artifact_path.stem
        if self._model is None:
            return RiskPrediction(model_status="unavailable", readiness="MODEL_NOT_READY",
                unavailable_reason="CatBoost artifact or runtime could not be loaded.",
                model_name=model_name, model_version=self._metadata.get("model_version"), feature_names=names)
        try:
            missing = [name for name in names if name not in features]
            unavailable = [
                name for name in names
                if name in features and (
                    features[name] is None
                    or not isinstance(features[name], (int, float))
                    or not math.isfinite(float(features[name]))
                )
            ]
            if missing or unavailable:
                reason = []
                if missing:
                    reason.append(f"missing required model features: {missing}")
                if unavailable:
                    reason.append(f"unavailable required model features: {unavailable}")
                return RiskPrediction(
                    model_status="error", readiness="MODEL_ERROR",
                    unavailable_reason="; ".join(reason),
                    model_name=model_name,
                    model_version=self._metadata.get("model_version"),
                    feature_names=names,
                )
            row = [[float(features[name]) for name in names]]
            probabilities = self._model.predict_proba(row)[0]
            positive = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            threshold = float(self._metadata.get("selected_threshold", .5))
            return RiskPrediction(model_status="available", readiness="READY", risk_probability=positive,
                risk_class=int(positive >= threshold), model_name=model_name,
                model_version=self._metadata.get("model_version"), feature_names=names,
                threshold=threshold, class_probabilities={str(i): float(v) for i, v in enumerate(probabilities)})
        except Exception as exc:
            return RiskPrediction(model_status="error", readiness="MODEL_ERROR", unavailable_reason=type(exc).__name__,
                model_name=model_name, model_version=self._metadata.get("model_version"), feature_names=names)
