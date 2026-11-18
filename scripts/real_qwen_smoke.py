"""Run the real Qwen repair through the same production workflow as the dashboard."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.analysis import PRAnalysisRequest
from backend.app.services.workflow import DeliveryWorkflowService

WORK = ROOT / ".tmp" / "real-qwen-smoke"
EVIDENCE = WORK / "evidence.json"


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    (WORK / "calculator.py").write_text("def percentage(value, total):\n    return value / total * 10\n", encoding="utf-8")
    (WORK / "test_calculator.py").write_text("from calculator import percentage\n\ndef test_baseline():\n    assert percentage(25, 100) == 2.5\n", encoding="utf-8")
    original = (WORK / "calculator.py").read_text(encoding="utf-8")
    diff = """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def percentage(value, total):
-    return value / total * 10
+    return value / total * 100
"""
    request = PRAnalysisRequest(repository="local/real-qwen-smoke", pull_request=1,
                                title="Fix percentage calculation",
                                body="Percentage must be value / total * 100.",
                                repository_path=str(WORK), diff=diff,
                                sources={"calculator.py": original})
    service = DeliveryWorkflowService()
    result = service.run(request, validate=True, repository_path=str(WORK))
    result = service.generate_candidate(result, root=str(WORK), diff=diff)
    candidate = result.repair.candidate or {}
    approval_error = None
    if candidate.get("selectable"):
        result.selection = {"variant": "AI_CANDIDATE", "selected_by": "real_qwen_smoke"}
        service.workflows[result.workflow_id] = result
        if result.approval_state == "READY_FOR_APPROVAL":
            try:
                result = service.approve(result.workflow_id)
            except Exception as exc:
                approval_error = f"{type(exc).__name__}: {exc}"
        else:
            approval_error = f"Workflow was not ready for approval: {result.approval_state}"
        candidate = result.repair.candidate or {}
    attempts = result.repair_attempts or []
    evidence = {
        "provider_invoked": any(a.get("repair_status") == "available" for a in attempts),
        "provider": result.repair.provider, "model": result.repair.model,
        "nonempty_patch": bool(result.repair.patch),
        "candidate_created": candidate.get("status") == "validated",
        "candidate_selectable": candidate.get("selectable") is True,
        "attempt_count": len(attempts),
        "repair_attempts": attempts,
        "final_candidate_id": candidate.get("candidate_id"),
        "final_changed_files": candidate.get("patched_files", []),
        "original_unchanged": (WORK / "calculator.py").read_text(encoding="utf-8") == original,
        "attempts": attempts, "candidate_validation": candidate.get("validation"),
        "generated_test_status": result.test_generation.status,
        "generated_tests": result.test_generation.generated_tests,
        "ci_pipeline": result.ci_pipeline,
        "candidate_reanalysis": candidate.get("reanalysis"),
        "candidate_comparison": result.candidate_comparison,
        "selected_variant": (result.selection or {}).get("variant"),
        "approval_error": approval_error,
        "approval_state": result.approval_state,
        "deployment_simulated": bool(result.deployment and result.deployment.simulation_status == "SIMULATED"),
        "deployment": result.deployment.model_dump() if result.deployment else None,
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items() if k not in {"attempts", "generated_tests", "ci_pipeline", "candidate_comparison", "deployment"}}, indent=2))


if __name__ == "__main__":
    main()
