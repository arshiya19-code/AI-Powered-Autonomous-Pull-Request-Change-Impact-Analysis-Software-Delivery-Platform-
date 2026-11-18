from pathlib import Path

from backend.app.schemas.models import (
    AnalysisResponse, ChangeSummary, DependencyChangeSummary, ImpactSummary,
    PRMetadata, RepairResult, RiskPrediction, ValidationCheck,
    ValidationSummary, WorkflowResponse,
)
from backend.app.services.repair import LLMProvider, RepairService
from backend.app.services.test_generation import TestGenerationService
from backend.app.services.validation import ValidationService
from backend.app.services.workflow import DeliveryWorkflowService


GENERATED_TEST = "from app import add\n\n\ndef test_add_regression():\n    assert add(1, 2) == 3\n"


class LoopProvider(LLMProvider):
    name = "qwen-test"
    model = "qwen-test"

    def __init__(self, always_fail=False):
        self.always_fail = always_fail
        self.repair_calls = []

    def propose_repair(self, context):
        if context.get("purpose") == "regression_test_generation":
            return RepairResult(
                status="available", explanation="Executable targeted coverage",
                proposed_changes=[{"path":"tests/ai_generated/test_add_regression.py","type":"regression","content":GENERATED_TEST,"test_name":"test_add_regression","target_symbol":"app.add","scenario":"1 + 2","purpose":"Protect addition","expected_behavior":"3"}],
            )
        self.repair_calls.append(context)
        attempt = int(context.get("attempt", 1))
        operator = "-" if self.always_fail else "+"
        return RepairResult(
            status="available", issue_summary="Correct add", explanation=f"Attempt {attempt}",
            patch=f"diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a * b\n+    return a {operator} b\n",
            affected_files=["app.py"], confidence=.9,
        )


def workflow() -> WorkflowResponse:
    analysis = AnalysisResponse(
        pr_metadata=PRMetadata(repository="owner/repo", pull_request=1),
        change_summary=ChangeSummary(files_changed=1, symbols_changed=1),
        dependency_change_summary=DependencyChangeSummary(), impact_summary=ImpactSummary(),
        features={}, risk_prediction=RiskPrediction(),
        change_details={"changed_files":["app.py"],"changed_symbols":[{"name":"add","file":"app.py"}]},
        impact_details={"impacted_nodes":[],"propagation_paths":[]},
        decision={}, validation={}, repair={}, deployment={}, monitoring={},
    )
    return WorkflowResponse(workflow_id="loop", analysis=analysis, validation=ValidationSummary(status="passed"))


class LoopValidation:
    def __init__(self, always_fail=False): self.calls=0; self.always_fail=always_fail
    def validate(self, root, generated_test_files=None):
        self.calls += 1; failed=self.always_fail or self.calls == 1
        checks={name:ValidationCheck(check_type=name,status="UNAVAILABLE").model_dump() for name in ("integration","dependency")}
        checks.update({name:ValidationCheck(check_type=name,status="PASSED").model_dump() for name in ("build","unit","security")})
        checks["regression"]=ValidationCheck(check_type="regression",status="FAILED" if failed else "PASSED",stderr_excerpt="assert 1 == 3" if failed else "1 passed").model_dump()
        return ValidationSummary(status="failed" if failed else "passed",checks=checks,blocking_failures=["regression"] if failed else [])


class LoopAnalysis:
    def analyze(self, request):
        measured=workflow().analysis
        measured.features={"measured":1.0}
        return measured


def service(provider: LoopProvider, always_fail=False) -> DeliveryWorkflowService:
    repair = RepairService(provider)
    return DeliveryWorkflowService(analysis=LoopAnalysis(),repair=repair,tests=TestGenerationService(repair),validation=LoopValidation(always_fail))


def make_repo(root: Path) -> None:
    (root / "app.py").write_text("def add(a, b):\n    return a * b\n", encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")


def test_failed_tests_retry_with_frozen_patch_and_preserve_original(tmp_path):
    make_repo(tmp_path); original=(tmp_path / "app.py").read_text()
    provider=LoopProvider(); result=service(provider).generate_candidate(workflow(), root=str(tmp_path), diff="real diff")
    assert [a["status"] for a in result.repair_attempts] == ["CANDIDATE_VALIDATION_FAILED", "CANDIDATE_VALIDATED"], result.repair_attempts
    assert len(provider.repair_calls) == 1
    assert result.repair_attempts[1]["production_patch_frozen"]
    assert result.repair_attempts[0]["patch"] == result.repair_attempts[1]["patch"]
    assert result.repair.candidate["status"] == "validated" and result.repair.candidate["selectable"] is True
    assert result.test_generation.generated_tests[0]["execution_status"] == "PASS"
    assert result.test_generation.generated_tests[0]["generated_code"]
    assert all(check["variant"] == "AI_CANDIDATE" for check in result.repair.candidate["validation"]["checks"].values())
    assert result.repair.candidate["files"][0]["original_code"] != result.repair.candidate["files"][0]["candidate_code"]
    assert result.repair.candidate["reanalysis"]["status"] == "AVAILABLE"
    assert (tmp_path / "app.py").read_text() == original
    assert len(result.ci_pipeline) == 10
    assert [stage["stage"] for stage in result.ci_pipeline][-3:] == [
        "Dependency manifest validation", "Snyk", "Candidate deterministic re-analysis"
    ]
    by_name = {stage["stage"]: stage for stage in result.ci_pipeline}
    assert "Dependency vulnerability scan" not in by_name
    assert "SonarQube Analysis" not in by_name
    assert by_name["Snyk"]["status"] == "NOT CONFIGURED"


def test_maximum_attempts_rejects_and_recommends_original(tmp_path):
    make_repo(tmp_path); provider=LoopProvider(always_fail=True)
    result=service(provider,always_fail=True).generate_candidate(workflow(), root=str(tmp_path), diff="real diff")
    assert len(result.repair_attempts) == DeliveryWorkflowService.MAX_REPAIR_ATTEMPTS == 3
    assert result.repair.candidate["status"] == "rejected" and result.repair.candidate["selectable"] is False
    assert result.candidate_comparison["recommendation"] == "USE_ORIGINAL"


def test_empty_retry_patch_is_recorded_and_attempt_three_uses_fresh_candidate(tmp_path):
    make_repo(tmp_path)
    provider = LoopProvider()
    original_propose = provider.propose_repair

    def sequenced(context):
        if context.get("purpose") == "regression_test_generation":
            return original_propose(context)
        if int(context.get("attempt", 1)) == 2:
            provider.repair_calls.append(context)
            return RepairResult(status="available", issue_summary="No retry patch", patch="   \n")
        return original_propose(context)

    provider.propose_repair = sequenced
    result = service(provider).generate_candidate(workflow(), root=str(tmp_path), diff="real diff")

    assert [item["status"] for item in result.repair_attempts] == [
        "CANDIDATE_VALIDATION_FAILED", "CANDIDATE_VALIDATED"
    ]
    assert len(provider.repair_calls) == 1
    assert result.repair_attempts[1]["production_patch_frozen"]
    assert result.repair.candidate["status"] == "validated"
    assert result.repair.candidate["attempt"] == 2
    assert result.repair.candidate["files"][0]["candidate_code"] != (tmp_path / "app.py").read_text()
    assert (tmp_path / "app.py").read_text() == "def add(a, b):\n    return a * b\n"


def test_three_empty_repairs_never_become_candidate_or_ready_for_approval(tmp_path):
    make_repo(tmp_path)
    provider = LoopProvider()

    def empty(context):
        if context.get("purpose") == "regression_test_generation":
            return RepairResult(status="available", proposed_changes=[])
        provider.repair_calls.append(context)
        return RepairResult(status="available", issue_summary="empty", patch="\t \n")

    provider.propose_repair = empty
    result = service(provider).generate_candidate(workflow(), root=str(tmp_path))

    assert len(result.repair_attempts) == 3
    assert all(item["candidate_created"] is False for item in result.repair_attempts)
    assert result.repair.candidate["status"] == "not_created"
    assert result.repair.patch is None
    assert result.approval_state == "BLOCKED"


def test_security_and_dependency_failures_are_blocking():
    validation=ValidationService().aggregate({
        "security":ValidationCheck(check_type="security",status="FAILED"),
        "dependency":ValidationCheck(check_type="dependency",status="FAILED"),
    })
    assert validation.status == "failed"
    assert validation.blocking_failures == ["security", "dependency"]
