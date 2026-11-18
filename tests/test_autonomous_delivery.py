from pathlib import Path
from backend.app.schemas.models import RiskPrediction, ValidationCheck, ValidationSummary, DecisionResult
from backend.app.services.decision import DecisionEngine
from backend.app.services.deployment import DeploymentService
from backend.app.services.repair import RepairService
from backend.app.services.test_generation import TestGenerationService
from backend.app.services.validation import detect_project, ValidationService

def test_repair_and_test_generation_are_explicitly_unavailable_without_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert RepairService().propose().status == "unavailable"
    result = TestGenerationService().generate(root=tmp_path, changed_symbols=["f"])
    assert result.status == "unavailable" and result.target_symbols == ["f"]

def test_project_detection(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]")
    profile = detect_project(tmp_path)
    assert profile.language == "Python" and profile.framework == "pytest"

def test_validation_aggregation_blocks_security_and_preserves_states():
    summary = ValidationService().aggregate({
        "build": ValidationCheck(check_type="build", status="PASSED"),
        "security": ValidationCheck(check_type="security", status="FAILED", summary="finding"),
        "dependency": ValidationCheck(check_type="dependency", status="UNAVAILABLE"),
    })
    assert summary.status == "failed"
    assert summary.blocking_failures == ["security"]

def test_decision_precedence_and_simulation():
    validation = ValidationSummary(status="failed", blocking_failures=["build"])
    decision = DecisionEngine().decide(risk=RiskPrediction(), impact_score=0, blast_radius=0, critical_nodes=0, analysis_complete=True, validation=validation)
    assert decision.decision == "BLOCK" and "BUILD_FAILED" in decision.reason_codes
    simulated = DeploymentService().simulate(decision, validation=validation)
    assert simulated.recommended_action == "BLOCK" and simulated.readiness is False

def test_no_fake_rollback_for_predeployment_risk():
    result = DeploymentService().simulate(DecisionResult(decision="DELAY", status="decided"), risk=RiskPrediction(model_status="available", risk_probability=0.99), analysis_complete=True)
    assert result.recommended_action == "DELAY"
