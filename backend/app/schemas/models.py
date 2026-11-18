from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

Decision = Literal["DEPLOY", "DELAY", "BLOCK", "ROLLBACK", "USE_CANDIDATE", "USE_ORIGINAL", "REVIEW_REQUIRED"]
CheckStatus = Literal["PASSED", "FAILED", "SKIPPED", "UNAVAILABLE", "ERROR"]
ValidationVariant = Literal["ORIGINAL", "AI_CANDIDATE"]


def normalize_validation_variant(value: Any) -> str:
    """Normalize legacy labels at the boundary; persist only workflow labels."""
    normalized = str(value or "ORIGINAL").strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"ORIGINAL", "ORIGINAL_SOURCE"}:
        return "ORIGINAL"
    if normalized in {"CANDIDATE", "AI_CANDIDATE", "AICANDIDATE"}:
        return "AI_CANDIDATE"
    return normalized

class PRMetadata(BaseModel):
    repository: str
    pull_request: Optional[int] = None
    base_sha: Optional[str] = None
    head_sha: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None

class ChangeSummary(BaseModel):
    files_changed: int = 0
    files_added: int = 0
    files_deleted: int = 0
    files_modified: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    symbols_changed: int = 0
    unsupported_files: int = 0

class DependencyChangeSummary(BaseModel):
    added: int = 0
    removed: int = 0
    modified: int = 0

class ImpactSummary(BaseModel):
    impact_score: Optional[float] = None
    impact_level: Optional[str] = None
    blast_radius: int = 0
    direct_nodes: int = 0
    impacted_nodes: int = 0
    maximum_depth: int = 0
    critical_nodes: int = 0

class RiskPrediction(BaseModel):
    model_status: Literal["available", "unavailable", "error"] = "unavailable"
    readiness: Literal["READY", "MODEL_NOT_READY", "MODEL_ERROR"] = "MODEL_NOT_READY"
    risk_probability: Optional[float] = None
    risk_class: Optional[int] = None
    model_version: Optional[str] = None
    model_name: Optional[str] = None
    feature_names: List[str] = Field(default_factory=list)
    threshold: Optional[float] = None
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    unavailable_reason: Optional[str] = None

class ShapExplanation(BaseModel):
    shap_status: Literal["available", "unavailable", "error"] = "unavailable"
    values: Optional[Dict[str, float]] = None
    base_value: Optional[float] = None
    prediction: Dict[str, Any] = Field(default_factory=dict)
    top_risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    protective_factors: List[Dict[str, Any]] = Field(default_factory=list)
    feature_contributions: List[Dict[str, Any]] = Field(default_factory=list)

class DecisionResult(BaseModel):
    decision: Optional[Decision] = None
    status: Literal["pending", "decided"] = "pending"
    reason_codes: List[str] = Field(default_factory=list)
    human_readable_reasons: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    blocking_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    input_signals: Dict[str, Any] = Field(default_factory=dict)
    decided_at: datetime = Field(default_factory=datetime.utcnow)

class ValidationSummary(BaseModel):
    ci: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["available", "unavailable", "passed", "failed", "not_run"] = "not_run"
    checks: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    overall_status: Optional[str] = None
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    unavailable: int = 0
    error_count: int = 0
    blocking_failures: List[str] = Field(default_factory=list)

class ValidationCheck(BaseModel):
    evidence: Dict[str, Any] = Field(default_factory=dict)
    check_type: str
    status: CheckStatus
    command: Optional[List[str]] = None
    tool: Optional[str] = None
    exit_code: Optional[int] = None
    duration: Optional[float] = None
    duration_seconds: Optional[float] = None
    workspace: Optional[str] = None
    variant: ValidationVariant = "ORIGINAL"

    @field_validator("variant", mode="before")
    @classmethod
    def canonical_variant(cls, value: Any) -> str:
        return normalize_validation_variant(value)
    verifies: str = ""
    command_display: Optional[str] = None
    severity_counts: Dict[str, int] = Field(default_factory=dict)
    summary: str = ""
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    warnings: List[str] = Field(default_factory=list)

class RepairResult(BaseModel):
    status: Literal["available", "unavailable", "error"] = "unavailable"
    provider: Optional[str] = None
    model: Optional[str] = None
    issue_summary: Optional[str] = None
    proposed_changes: List[Any] = Field(default_factory=list)
    patch: Optional[str] = None
    explanation: Optional[str] = None
    affected_files: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    provider_error: Dict[str, Any] = Field(default_factory=dict)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    regression_test: Dict[str, Any] = Field(default_factory=dict)
    integration_test: Dict[str, Any] = Field(default_factory=dict)
    generated_tests: List[str] = Field(default_factory=list)
    candidate: Dict[str, Any] = Field(default_factory=dict)

class TestGenerationResult(BaseModel):
    status: Literal["available", "unavailable", "error"] = "unavailable"
    target_symbols: List[str] = Field(default_factory=list)
    generated_test_files: Dict[str, str] = Field(default_factory=dict)
    test_rationale: str = ""
    framework_detected: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    generated_tests: List[Dict[str, Any]] = Field(default_factory=list)
    generated_integration_tests: List[Dict[str, Any]] = Field(default_factory=list)
    generation_attempts: List[Dict[str, Any]] = Field(default_factory=list)

class DeploymentSimulationResult(BaseModel):
    simulation_status: Literal["SIMULATED", "BLOCKED", "ERROR"] = "SIMULATED"
    recommended_action: Decision
    readiness: bool = False
    blocking_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
    risk_summary: Dict[str, Any] = Field(default_factory=dict)
    recommended_actions: List[str] = Field(default_factory=list)
    lifecycle_state: str = "READY_FOR_APPROVAL"
    lifecycle_history: List[str] = Field(default_factory=lambda: ["READY_FOR_APPROVAL"])
    deployment_mode: Literal["simulation", "production"] = "simulation"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    simulation_details: Dict[str, Any] = Field(default_factory=dict)
    decision_context: Dict[str, Any] = Field(default_factory=dict)

class RepairSummary(BaseModel):
    status: Literal["available", "unavailable", "error"] = "unavailable"
    patch: Optional[str] = None
    explanation: Optional[str] = None

class DeploymentSummary(BaseModel):
    action: Optional[Decision] = None
    status: Literal["not_started", "simulated", "completed", "failed"] = "not_started"
    message: Optional[str] = None

class MonitoringSummary(BaseModel):
    status: Literal["unavailable", "available", "regression"] = "unavailable"
    metrics: Optional[Dict[str, float]] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    health_state: Literal["HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"] = "UNKNOWN"
    request_metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    latency_indicators: Dict[str, Optional[float]] = Field(default_factory=dict)
    resource_indicators: Dict[str, Optional[float]] = Field(default_factory=dict)
    deployment_id: Optional[str] = None
    regression_indicators: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provider: str = "internal"
    source: str = "internal"

class FeedbackRecord(BaseModel):
    workflow_id: str
    repository: str = "local"
    pull_request: Optional[int] = None
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    features: Optional[Dict[str, Optional[float]]] = None
    features_fingerprint: Optional[str] = None
    predicted_risk: Optional[float] = None
    model_version: Optional[str] = None
    shap_summary: Optional[Dict[str, float]] = None
    decision: Optional[str] = None
    validation_outcome: Optional[str] = None
    repair_outcome: Optional[str] = None
    deployment_result: Optional[str] = None
    actual_deployment_outcome: Optional[str] = None
    monitoring_health: Optional[str] = None
    rollback_occurred: Optional[bool] = None
    final_outcome_label: Optional[str] = None
    outcome_known: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    structural_impact: Optional[Dict[str, Any]] = None
    selected_variant: Optional[str] = None
    original_validation: Optional[Dict[str, Any]] = None
    candidate_validation: Optional[Dict[str, Any]] = None
    approval: Optional[str] = None
    deployment_outcome: Optional[str] = None
    prediction_usefulness: Optional[str] = None
    repair_usefulness: Optional[str] = None
    user_note: Optional[str] = None

class DeploymentHealthEvidence(BaseModel):
    error_rate: Optional[float] = None
    baseline_error_rate: Optional[float] = None
    latency: Optional[float] = None
    baseline_latency: Optional[float] = None
    health_checks_failed: int = 0
    service_available: Optional[bool] = None
    severe_regression: bool = False
    deployment_id: Optional[str] = None

class HistoryItem(BaseModel):
    workflow_id: str
    repository: str
    pull_request: Optional[int] = None
    timestamp: datetime
    risk_status: Optional[str] = None
    decision: Optional[str] = None
    validation_status: Optional[str] = None
    deployment_result: Optional[str] = None
    monitoring_health: Optional[str] = None
    rollback_status: Optional[bool] = None

class AnalysisResponse(BaseModel):
    graph: Dict[str, Any] = Field(default_factory=lambda: {'nodes': [], 'edges': []})
    pr_metadata: PRMetadata
    detected_languages: List[str] = Field(default_factory=list)
    change_summary: ChangeSummary
    dependency_change_summary: DependencyChangeSummary
    impact_summary: ImpactSummary
    change_details: Dict[str, Any] = Field(default_factory=dict)
    impact_details: Dict[str, Any] = Field(default_factory=dict)
    features: Dict[str, Optional[float]]
    model_status: str = "unavailable"
    risk_prediction: RiskPrediction
    shap: Optional[ShapExplanation] = None
    decision: DecisionResult
    validation: ValidationSummary
    repair: RepairSummary
    deployment: DeploymentSummary
    monitoring: MonitoringSummary
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    analysis_complete: bool = False

class WorkflowResponse(BaseModel):
    finalized_demo: bool = False
    purpose: Optional[str] = None
    source_analysis_id: Optional[str] = None
    artifact_origin: Optional[str] = None
    provider_request_count: Optional[int] = None
    replayed: bool = False
    workflow_id: Optional[str] = None
    analysis: Optional[AnalysisResponse] = None
    validation: ValidationSummary = Field(default_factory=ValidationSummary)
    repair: RepairResult = Field(default_factory=RepairResult)
    test_generation: TestGenerationResult = Field(default_factory=TestGenerationResult)
    deployment: Optional[DeploymentSimulationResult] = None
    decision: Optional[DecisionResult] = None
    reason_codes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    monitoring: MonitoringSummary = Field(default_factory=MonitoringSummary)
    stages: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    approval_state: str = "READY_FOR_APPROVAL"
    approval_metadata: Dict[str, Any] = Field(default_factory=dict)
    provider_readiness: Dict[str, Any] = Field(default_factory=dict)
    candidate_comparison: Dict[str, Any] = Field(default_factory=dict)
    selection: Dict[str, Any] = Field(default_factory=dict)
    pre_repair_decision: Optional[DecisionResult] = None
    post_candidate_decision: Optional[DecisionResult] = None
    feedback: Dict[str, Any] = Field(default_factory=dict)
    repair_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    ci_pipeline: List[Dict[str, Any]] = Field(default_factory=list)

class SystemStatus(BaseModel):
    tool_readiness: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "degraded"]
    model_status: str
    shap_status: str
    services: Dict[str, str]
    provider_readiness: Dict[str, Any] = Field(default_factory=dict)
