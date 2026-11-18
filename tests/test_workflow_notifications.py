from backend.app.schemas.models import (
    AnalysisResponse,
    ChangeSummary,
    DecisionResult,
    DependencyChangeSummary,
    ImpactSummary,
    PRMetadata,
    RiskPrediction,
    ValidationSummary,
)
from backend.app.services.repair import RepairService
from backend.app.services.workflow import DeliveryWorkflowService


class StaticDecision:
    def decide(self, **kwargs):
        return DecisionResult(decision="BLOCK", status="decided")


class StaticAnalysis:
    decision = StaticDecision()

    def analyze(self, request, notify=None):
        if notify:
            notify("ANALYSIS_STARTED", {"source": "test"})
        return AnalysisResponse(
            pr_metadata=PRMetadata(repository=request.repository, pull_request=request.pull_request),
            change_summary=ChangeSummary(),
            dependency_change_summary=DependencyChangeSummary(),
            impact_summary=ImpactSummary(),
            features={},
            risk_prediction=RiskPrediction(),
            decision={}, validation={}, repair={}, deployment={}, monitoring={},
        )


class StaticValidation:
    def validate(self, root, **kwargs):
        return ValidationSummary(status="passed")

    def not_run(self):
        return ValidationSummary(status="not_run")


def make_service():
    return DeliveryWorkflowService(
        analysis=StaticAnalysis(),
        validation=StaticValidation(),
        repair=RepairService(),
    )


def make_request(tmp_path):
    from backend.app.services.analysis import PRAnalysisRequest
    return PRAnalysisRequest(repository="local/test", pull_request=1, diff="", repository_path=str(tmp_path))


def test_default_notify_accepts_event_name_and_payload(tmp_path):
    result = make_service().run(make_request(tmp_path), validate=True, repository_path=str(tmp_path))
    assert result.validation.status == "passed"


def test_injected_notify_receives_both_values_and_preserves_payload(tmp_path):
    events = []
    result = make_service().run(
        make_request(tmp_path), validate=True, repository_path=str(tmp_path),
        notify=lambda event_name, payload: events.append((event_name, payload)),
    )
    assert ("ANALYSIS_STARTED", {"source": "test"}) in events
    assert ("VALIDATION_COMPLETE", {
        "status": "passed",
        "message": "Original validation: passed",
        "blocking_failures": [],
    }) in events
    assert events[-1][0] == "WORKFLOW_BLOCKED"
    assert isinstance(events[-1][1], dict)
    assert result.validation.status == "passed"
