from __future__ import annotations
from typing import Mapping
from backend.app.schemas.models import ShapExplanation
from dataset.config import CATBOOST_MODEL_FEATURES
import math
import os
from pathlib import Path

class ShapService:
    def __init__(self, model=None):
        self.model = model
    def explain(self, features: Mapping[str, float], risk: object) -> ShapExplanation:
        model = self.model
        if model is None or getattr(risk, "model_status", "unavailable") != "available":
            return ShapExplanation(shap_status="unavailable", prediction={"readiness": "MODEL_NOT_READY"})
        try:
            import shap
            # CatBoost artifacts trained from arrays may expose positional names
            # ("0", "1", ...). The persisted risk contract is authoritative.
            names = list(getattr(risk, "feature_names", None) or CATBOOST_MODEL_FEATURES)
            missing = [name for name in names if name not in features or features[name] is None or not math.isfinite(float(features[name]))]
            if missing:
                return ShapExplanation(shap_status="error", prediction={"readiness": "MODEL_ERROR", "reason": f"required model features unavailable: {missing}"})
            explanation = shap.TreeExplainer(model)([[float(features[k]) for k in names]])
            values = explanation.values[0]
            if getattr(values, "ndim", 1) == 2: values = values[:, 1]
            pairs = [{"feature": n, "value": None if isinstance(features.get(n), float) and math.isnan(features.get(n)) else features.get(n), "contribution": float(v)} for n, v in zip(names, values)]
            risk_up = sorted((p for p in pairs if p["contribution"] > 0), key=lambda x: x["contribution"], reverse=True)[:5]
            protective = sorted((p for p in pairs if p["contribution"] < 0), key=lambda x: x["contribution"])[:5]
            return ShapExplanation(shap_status="available", values={p["feature"]: p["contribution"] for p in pairs},
                base_value=float(explanation.base_values[0][1] if getattr(explanation.base_values[0], "__len__", None) else explanation.base_values[0]),
                prediction=risk.model_dump(), top_risk_factors=risk_up, protective_factors=protective, feature_contributions=pairs)
        except Exception:
            return ShapExplanation(shap_status="error", prediction={"readiness": "MODEL_ERROR"})
