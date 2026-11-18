from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.presentation import _validation_presentation
from backend.app.schemas.models import RepairResult, ValidationCheck
from backend.app.services.candidate import CandidateWorkspaceService
from backend.app.services.repair import LLMProvider, RepairService
from backend.app.services.validation import SafeCommandRunner, ValidationService, detect_project
from backend.app.services.workflow import DeliveryWorkflowService
from backend.app.services.test_generation import TestGenerationService
from tests.test_ai_repair_validation_loop import workflow, make_repo, LoopAnalysis


class NoPatchProvider(LLMProvider):
    name = "qwen-test"
    model = "qwen-test"

    def propose_repair(self, context):
        return RepairResult(status="available", issue_summary="No safe repair was identified.")


def test_pytest_no_tests_is_tooling_evidence_not_a_code_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SimpleNamespace(
        returncode=5, stdout="\nno tests ran in 0.01s\n", stderr="",
    ))

    check = SafeCommandRunner().run(["python", "-m", "pytest", "-q"], tmp_path)

    assert check.status == "UNAVAILABLE"
    assert check.evidence == {"failure_category": "ENVIRONMENT_TOOLING", "reason_code": "NO_TESTS_COLLECTED", "state": "NOT_DEFINED"}
    assert "not failed" in check.summary


def test_duration_and_python_compilation_semantics_are_honest(tmp_path):
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    profile = detect_project(tmp_path)
    assert "compileall" in profile.commands["build"]
    check = ValidationCheck(check_type="build", status="PASSED", command=profile.commands["build"],
                            command_display="python -m compileall -q .", workspace=str(tmp_path), duration=.224,
                            duration_seconds=.224, verifies=ValidationService.VERIFY["build"])
    rendered = _validation_presentation(ValidationService().aggregate({"build": check}).model_dump())["checks"][0]
    assert rendered["label"] == "Python Compilation Check"
    assert rendered["duration_label"] == "Duration: 0.22 s"
    assert "not a package build" in rendered["why_it_matters"]


def test_manifest_validation_is_separate_from_vulnerability_scanning(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='1'\n", encoding="utf-8")
    result = ValidationService().validate(tmp_path, requested=["dependency", "dependency_vulnerability"])
    assert result.checks["dependency"]["status"] == "PASSED"
    assert "manifest" in result.checks["dependency"]["verifies"].lower()
    vulnerability = result.checks["dependency_vulnerability"]
    if vulnerability["status"] == "UNAVAILABLE":
        assert vulnerability["summary"] == "Dependency vulnerability scan: Not configured"


def test_three_no_patch_generations_never_claim_candidate_validation(tmp_path):
    make_repo(tmp_path)
    repair = RepairService(NoPatchProvider())
    service = DeliveryWorkflowService(analysis=LoopAnalysis(), validation=ValidationService(), repair=repair, tests=TestGenerationService(repair))
    result = service.generate_candidate(workflow(), root=str(tmp_path), diff="real diff")
    assert [item["status"] for item in result.repair_attempts] == ["NO_PATCH_RETURNED"] * 3
    assert result.repair.candidate["status"] == "not_created"
    assert result.repair.candidate["selectable"] is False
    assert result.repair.candidate["reason"] == "AI candidate not created. No safe repair was identified."


def test_generated_test_patch_rejects_production_source(tmp_path):
    make_repo(tmp_path)
    candidate = CandidateWorkspaceService().create(tmp_path)
    patch = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a * b\n+    return a + b\n"
    try:
        with pytest.raises(ValueError, match="production source"):
            CandidateWorkspaceService().apply_patch(candidate, patch, test_only=True)
    finally:
        candidate.cleanup()


def test_blocking_failure_marks_downstream_stages_skipped(tmp_path):
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    result = ValidationService().validate(tmp_path, variant="AI_CANDIDATE")
    assert result.checks["build"]["status"] == "FAILED"
    assert all(result.checks[key]["status"] == "SKIPPED" for key in list(result.checks)[1:])
    assert all(result.checks[key]["workspace"] == str(tmp_path.resolve()) for key in result.checks)
    assert all(result.checks[key]["variant"] == "AI_CANDIDATE" for key in result.checks)


def test_targeted_integration_metadata_is_persistable():
    from backend.app.schemas.models import TestGenerationResult
    item = {"test_id":"integration-1","test_name":"test_cli_percent","test_file":"tests/test_ai_integration.py",
            "target_symbol":"CalculatorCLI.run -> Calculator.percent","purpose":"Protect the CLI-to-model path",
            "scenario":"CLI receives 50%","input_setup":"args=['50%']","expected_behavior":"prints 0.5",
            "generated_code":"def test_cli_percent(): pass","execution_status":"PASS","duration_seconds":.04,
            "stdout_stderr_summary":"1 passed"}
    result = TestGenerationResult(status="available", generated_integration_tests=[item])
    assert result.model_dump()["generated_integration_tests"][0] == item


def test_failed_security_gate_is_never_presented_as_passed():
    check = ValidationCheck(check_type="security", status="FAILED", severity_counts={"High": 1, "Medium": 1, "Low": 2})
    rendered = _validation_presentation(ValidationService().aggregate({"security": check}).model_dump())["checks"][0]
    assert rendered["security_gate"] == "FAIL"
    assert rendered["human_result"].startswith("FAIL · Blocking High findings: 1")


def test_no_candidate_pipeline_skips_every_candidate_execution_stage(tmp_path):
    make_repo(tmp_path)
    repair = RepairService(NoPatchProvider())
    service = DeliveryWorkflowService(analysis=LoopAnalysis(), validation=ValidationService(), repair=repair, tests=TestGenerationService(repair))
    result = service.generate_candidate(workflow(), root=str(tmp_path), diff="real diff")
    by_name = {item["stage"]: item for item in result.ci_pipeline}
    assert by_name["Candidate workspace / patch validation"]["status"] == "SKIPPED"
    assert all(by_name[name]["status"] == "SKIPPED" for name in (
        "Python Compilation", "Existing repository PyTest", "AI regression tests",
        "AI targeted integration tests", "Bandit", "Dependency manifest validation",
        "Snyk", "Candidate deterministic re-analysis",
    ))
