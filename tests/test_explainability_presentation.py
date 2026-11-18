from backend.app.presentation import build_presentation
from backend.app.schemas.models import (
    DecisionResult,
    DeploymentSimulationResult,
    ValidationSummary,
    WorkflowResponse,
)
from backend.app.services.workflow import DeliveryWorkflowService


ANALYSIS_ID = "ed92a09d-b58f-423f-942f-f72eed820085"


def persisted_payload():
    checks = {
        name: {"status": status, "summary": "persisted evidence", "duration": 0.1}
        for name, status in {
            "build": "PASSED",
            "unit": "PASSED",
            "security": "PASSED",
            "dependency": "PASSED",
            "regression": "UNAVAILABLE",
            "integration": "UNAVAILABLE",
        }.items()
    }
    return {
        "workflow_id": ANALYSIS_ID,
        "analysis": {
            "detected_languages": ["py"],
            "change_summary": {"files_changed": 3, "lines_added": 23, "lines_deleted": 7, "symbols_changed": 15},
            "impact_summary": {"impact_score": 55.51, "impact_level": "HIGH", "impacted_nodes": 33, "maximum_depth": 4, "critical_nodes": 2},
            "risk_prediction": {"model_status": "available", "risk_probability": 0.4379597525, "threshold": 0.565},
            "shap": {
                "top_risk_factors": [
                    {"feature": "lines_deleted", "value": 7, "contribution": 0.1116819543},
                    {"feature": "files_changed", "value": 3, "contribution": 0.0699357723},
                ],
                "protective_factors": [
                    {"feature": "lines_added", "value": 23, "contribution": -0.2078894108},
                    {"feature": "total_lines_changed", "value": 30, "contribution": -0.1801323666},
                ],
            },
            "graph": {
                "nodes": [
                    {"id": "compute", "label": "compute_result", "qualified_name": "__init__.Calculator.compute_result", "file": "calculator/__init__.py", "module": "calculator.__init__", "changed": True, "critical": True, "depth": 0},
                    {"id": "run", "label": "run", "qualified_name": "cli.CalculatorCLI.run", "file": "calculator/cli.py", "module": "calculator.cli", "changed": False, "critical": True, "depth": 1},
                ],
                "edges": [{"source": "compute", "target": "run", "type": "CALLS"}],
            },
        },
        "validation": {"status": "passed", "passed": 4, "unavailable": 2, "blocking_failures": [], "checks": checks},
        "decision": {"decision": "DELAY", "human_readable_reasons": ["The change affects a large dependency blast radius."]},
        # This reflects the previously persisted display bug. Presentation must
        # still use the authoritative Decision Engine recommendation.
        "deployment": {"recommended_action": "DEPLOY", "lifecycle_state": "DEPLOYMENT_COMPLETED"},
        "approval_state": "DEPLOYMENT_COMPLETED",
        "approval_metadata": {"decision": "APPROVED", "actor": "Arshiya", "timestamp": 1789286400},
        "repair": {"status": "unavailable", "provider": None},
        "monitoring": {"status": "unavailable"},
    }


def test_shap_and_probability_explanations_are_directional_not_causal():
    result = build_presentation(ANALYSIS_ID, "COMPLETE", persisted_payload())
    risk = result["risk"]
    assert risk["classification"] == "LOW"
    assert "43.8%" in risk["narrative"] and "56.5%" in risk["narrative"]
    assert risk["increasing_factors"][0]["contribution"] == 0.1116819543
    assert "upward" in risk["increasing_factors"][0]["explanation"]
    assert risk["reducing_factors"][0]["contribution"] == -0.2078894108
    assert "downward" in risk["reducing_factors"][0]["explanation"]
    assert "baseline" in risk["shap_definition"]
    assert "caus" not in risk["shap_definition"].lower()


def test_structural_ml_decision_and_same_analysis_correlation():
    result = build_presentation(ANALYSIS_ID, "COMPLETE", persisted_payload())
    assert result["analysis_id"] == ANALYSIS_ID
    assert "different things" in " ".join(result["structural_comparison"]["points"])
    assert "33 impacted nodes" in result["structural_comparison"]["analysis_specific"]
    assert result["decision"]["recommendation"] == "DELAY"
    assert "Blast radius affects 33 nodes" in result["decision"]["reasons"]
    assert "2 critical components are affected" in result["decision"]["reasons"]
    assert result["summary"]["change"] == "3 files / 30 lines"
    assert result["impact"]["narrative"] == (
        "This PR modifies 15 symbols across 3 Python files. Changes propagate to 33 dependent nodes over 4 levels "
        "and affect 2 critical components, giving a HIGH structural impact score of 55.51."
    )
    assert result["test_generation"]["intentions"] == []


def test_approval_validation_repair_and_monitoring_labels():
    result = build_presentation(ANALYSIS_ID, "COMPLETE", persisted_payload())
    assert result["deployment"] == {
        "system_recommendation": "DELAY",
        "human_decision": "APPROVED",
        "final_action": "DEPLOYMENT SIMULATION COMPLETED",
        "actor": "Arshiya",
        "timestamp": 1789286400,
    }
    assert result["validation"]["label"] == "PASSED WITH LIMITATIONS"
    assert [item["status"] for item in result["validation"]["checks"]][-2:] == ["PASSED", "PASSED"]
    assert result["repair"]["label"] == "AI Repair not configured"
    assert result["monitoring"] == {"workflow_status": "ACTIVE", "external_telemetry": "NOT CONFIGURED"}


def test_explicit_approval_preserves_delay_recommendation():
    workflow = WorkflowResponse(
        workflow_id="approval-test",
        approval_state="READY_FOR_APPROVAL",
        validation=ValidationSummary(status="passed"),
        decision=DecisionResult(decision="DELAY", status="decided"),
        deployment=DeploymentSimulationResult(recommended_action="DELAY", lifecycle_state="READY_FOR_APPROVAL"),
        stages={"USER_APPROVAL": {"status": "ready_for_approval"}, "DEPLOYMENT": {"status": "not_started"}, "MONITORING": {"status": "not_started"}},
    )
    service = DeliveryWorkflowService.__new__(DeliveryWorkflowService)
    service.workflows = {"approval-test": workflow}
    service._record_action = lambda *_: None
    workflow.selection = {"variant": "ORIGINAL", "actor": "test"}
    approved = service.approve("approval-test")
    assert approved.decision.decision == "DELAY"
    assert approved.deployment.recommended_action == "DELAY"
    assert approved.deployment.lifecycle_state == "DEPLOYMENT_COMPLETED"
