"""
ML Package for Phase 8B - AIDev Dataset Integration + CatBoost Risk Prediction

This package contains:
- dataset_builder: AIDev dataset ingestion and feature extraction
- labels: Label assignment and leakage prevention
- preprocessing: Feature matrix construction and quality checks
- train: CatBoost training with repository-aware CV
- evaluate: Model evaluation with threshold analysis and calibration
- explain: SHAP-based explanations
"""

from analyzer.ml.dataset_builder import AIDevDatasetBuilder, FeatureRow, FilterCounters
from analyzer.ml.labels import assign_high_risk_label, validate_no_label_leakage, LEAKAGE_FIELDS
from analyzer.ml.preprocessing import (
    build_feature_matrix,
    run_quality_checks,
    DatasetQualityReport,
)
from analyzer.ml.train import (
    RepositoryAwareSplitter,
    HyperparameterTuner,
    train_catboost,
)
from analyzer.ml.evaluate import (
    evaluate_model,
    threshold_analysis,
    cross_validate_by_repo,
    calibration_analysis,
    EvaluationResult,
)
from analyzer.ml.explain import (
    generate_shap_explanations,
    global_shap_summary,
    individual_pr_explanation,
)

__all__ = [
    # dataset_builder
    "AIDevDatasetBuilder",
    "FeatureRow",
    "FilterCounters",
    # labels
    "assign_high_risk_label",
    "validate_no_label_leakage",
    "LEAKAGE_FIELDS",
    # preprocessing
    "build_feature_matrix",
    "run_quality_checks",
    "DatasetQualityReport",
    # train
    "RepositoryAwareSplitter",
    "HyperparameterTuner",
    "train_catboost",
    # evaluate
    "evaluate_model",
    "threshold_analysis",
    "cross_validate_by_repo",
    "calibration_analysis",
    "EvaluationResult",
    # explain
    "generate_shap_explanations",
    "global_shap_summary",
    "individual_pr_explanation",
]