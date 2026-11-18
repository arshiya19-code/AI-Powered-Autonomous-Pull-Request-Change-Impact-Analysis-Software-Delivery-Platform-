from pathlib import Path
import pytest
from backend.app.presentation import build_presentation
from backend.app.schemas.models import (
    AnalysisResponse, ChangeSummary, DependencyChangeSummary, ImpactSummary,
    PRMetadata, RepairResult, RiskPrediction, TestGenerationResult,
    DeploymentSimulationResult, DecisionResult, ValidationCheck, ValidationSummary, WorkflowResponse,
)
from backend.app.services.repair import LLMProvider, RepairService
from backend.app.services.test_generation import TestGenerationService
from backend.app.services.workflow import DeliveryWorkflowService


class StructuredTestProvider(LLMProvider):
    name = "test-provider"
    model = "test-model"
    def __init__(self, tests): self.tests = tests; self.context = None
    def propose_repair(self, context):
        self.context = context
        return RepairResult(status="available", provider=self.name, model=self.model,
                            explanation="Targeted evidence", proposed_changes=self.tests)


def test_provider_readiness_is_resolved_from_current_environment(monkeypatch):
    service = RepairService()
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen/qwen-2.5-coder-32b-instruct")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    ready = service.readiness()
    assert ready["status"] == "READY" and ready["provider_display"] == "OpenRouter"
    assert ready["model_display"] == "Qwen2.5-Coder 32B"
    assert "secret" not in str(ready)


def test_stale_repair_does_not_hide_live_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter"); monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"); monkeypatch.setenv("LLM_API_KEY", "secret")
    payload={"workflow_id":"same-id","analysis":{},"repair":{"status":"unavailable"},"provider_readiness":RepairService().readiness()}
    assert build_presentation("same-id","COMPLETE",payload)["repair"]["label"] == "Ready for your request"


def test_full_repair_context_uses_same_analysis():
    analysis=AnalysisResponse(pr_metadata=PRMetadata(repository="owner/repo",pull_request=1),change_summary=ChangeSummary(),dependency_change_summary=DependencyChangeSummary(),impact_summary=ImpactSummary(blast_radius=4),features={},risk_prediction=RiskPrediction(),decision={},validation={},repair={},deployment={},monitoring={},change_details={"changed_files":["app.py"],"changed_symbols":[{"name":"calculate","file":"app.py"}]},impact_details={"impacted_nodes":["Calculator.state"],"propagation_paths":[["calculate","state"]]})
    workflow=WorkflowResponse(workflow_id="same-id",analysis=analysis,validation=ValidationSummary(status="passed"))
    context=DeliveryWorkflowService.repair_context(workflow,diff="diff evidence")
    assert context["analysis_id"] == "same-id" and context["diff"] == "diff evidence"
    assert context["changed_symbols"][0]["file"] == "app.py" and context["propagation_paths"]


def test_test_generation_rejects_production_file(tmp_path):
    provider=StructuredTestProvider([{
        "path":"app.py", "type":"regression", "purpose":"invalid production target",
        "content":"def test_example():\n    assert True\n",
    }])
    result=TestGenerationService(provider).generate(root=tmp_path,changed_symbols=[{"name":"calculate","file":"app.py"}])
    assert result.status == "error"
    assert any("generated tests cannot target production files" in warning for warning in result.warnings)


def test_generated_test_preflight_rejects_without_rewriting(tmp_path):
    details = [{
        "path": "tests/ai_generated/test_generated.py",
        "content": "import json\n\ndef test_generated():\n    assert True\n",
        "generated_code": "import json\n\ndef test_generated():\n    assert True\n",
        "type": "regression",
        "test_kind": "regression",
        "purpose": "exercise generated behavior",
    }]

    original = details[0]["content"]
    with pytest.raises(ValueError, match="Ruff preflight"):
        TestGenerationService._ruff_preflight(tmp_path, details)
    assert details[0]["content"] == original
    assert details[0]["generated_code"] == original


def test_generated_test_rejects_unavailable_application_fixture(tmp_path):
    details = [{
        "path": "tests/ai_generated/test_generated.py",
        "content": "def test_generated(client):\n    assert client\n",
    }]

    try:
        TestGenerationService._fixture_preflight(tmp_path, details)
    except ValueError as exc:
        assert "unavailable fixture(s): client" in str(exc)
        assert "source/AST regression test" in str(exc)
    else:
        raise AssertionError("unavailable fixture was accepted")


def test_generated_regression_must_exercise_repaired_target():
    details = [{
        "path": "tests/ai_generated/test_unrelated.py",
        "purpose": "exercise unrelated helper",
        "content": "def test_unrelated():\n    assert True\n",
        "test_kind": "regression",
    }]

    with pytest.raises(ValueError, match="does not exercise the repaired target.*primes"):
        TestGenerationService._repair_relevance_preflight(details, ["primes"])


def test_candidate_requires_generated_regression_to_execute():
    summary = ValidationSummary(
        status="passed",
        checks={"regression": ValidationCheck(check_type="regression", status="UNAVAILABLE").model_dump()},
    )
    generated = TestGenerationResult(
        status="available",
        generated_tests=[{"test_file": "tests/ai_generated/test_fix.py"}],
    )

    enforced = DeliveryWorkflowService._enforce_ai_test_execution(summary, generated)

    assert enforced.status == "failed"
    assert enforced.blocking_failures == ["regression"]


def test_generated_test_schema_and_visibility(tmp_path):
    provider=StructuredTestProvider([{
        "path":"tests/ai_generated/test_app.py", "type":"regression",
        "purpose":"Protect calculate behavior",
        "content":"def test_percentage():\n    assert True\n",
    }])
    result=TestGenerationService(provider).generate(root=tmp_path,changed_symbols=[{"name":"calculate","file":"app.py"}])
    assert result.status == "available"
    assert result.targets[0]["file"] == "app.py"
    generated=result.generated_tests[0]
    assert generated["path"] == "tests/ai_generated/test_app.py"
    assert generated["test_file"] == "tests/ai_generated/test_app.py"
    assert generated["generated_code"] == "def test_percentage():\n    assert True\n"
    assert generated["type"] == "regression" and generated["test_kind"] == "regression"
    assert generated["purpose"] == "Protect calculate behavior"
    assert result.generated_test_files == {"tests/ai_generated/test_app.py": generated["generated_code"]}


def test_validation_explanation_and_security_advisories():
    validation=ValidationSummary(status="passed",passed=1,checks={"security":ValidationCheck(check_type="security",status="PASSED",stdout_excerpt="Low: 2\nMedium: 1\nHigh: 0").model_dump()})
    presented=build_presentation("same-id","COMPLETE",{"workflow_id":"same-id","analysis":{},"validation":validation.model_dump()})["validation"]
    security=presented["checks"][0]
    assert security["security_gate"] == "PASS" and security["severity_counts"] == {"Low":2,"Medium":1,"High":0}


def test_validation_variants_are_canonical_and_api_presentation_keeps_scope():
    original = ValidationCheck(check_type="unit", status="PASSED", variant="ORIGINAL")
    candidate = ValidationCheck(check_type="unit", status="PASSED", variant="AI_CANDIDATE")
    legacy = ValidationCheck(check_type="unit", status="PASSED", variant="Candidate")
    assert original.variant == "ORIGINAL"
    assert candidate.variant == "AI_CANDIDATE"
    assert legacy.variant == "AI_CANDIDATE"
    presented = build_presentation("same-id", "COMPLETE", {"workflow_id":"same-id", "analysis":{}, "validation":{"checks":{"unit":candidate.model_dump()}}})
    assert presented["validation"]["checks"][0]["variant"] == "AI_CANDIDATE"
    frontend = Path(__file__).parents[1] / "frontend" / "src" / "main.tsx"
    assert "['Variant',check.variant]" in frontend.read_text(encoding="utf-8")


def test_approval_uses_selected_candidate_validation_and_decision():
    result = WorkflowResponse(
        workflow_id="candidate-approval",
        validation=ValidationSummary(status="failed", blocking_failures=["unit"]),
        repair=RepairResult(candidate={
            "selectable": True,
            "validation": ValidationSummary(status="passed").model_dump(),
            "reanalysis":{"status":"AVAILABLE", "decision":DecisionResult(decision="DEPLOY", status="decided").model_dump()},
        }),
        decision=DecisionResult(decision="BLOCK", status="decided"),
        deployment=DeploymentSimulationResult(recommended_action="BLOCK", simulation_status="BLOCKED"),
        approval_state="READY_FOR_APPROVAL",
        selection={"variant":"AI_CANDIDATE"},
    )
    service = DeliveryWorkflowService()
    service.workflows[result.workflow_id] = result
    approved = service.approve(result.workflow_id)
    assert approved.approval_state == "DEPLOYMENT_COMPLETED"
    assert approved.deployment.simulation_details["selected_variant"] == "AI_CANDIDATE"
