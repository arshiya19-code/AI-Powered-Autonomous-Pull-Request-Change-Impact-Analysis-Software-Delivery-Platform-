"""Deterministic, user-facing explanations derived from persisted workflow evidence."""
from __future__ import annotations

from typing import Any
import re
from backend.app.schemas.models import normalize_validation_variant


FEATURE_LABELS = {
    "files_changed": "Files changed",
    "lines_deleted": "Lines deleted",
    "lines_added": "Lines added",
    "total_lines_changed": "Total lines changed",
}

FEATURE_SENTENCES = {
    "files_changed": "The PR modifies {value:g} files. In this prediction, that {direction} estimated risk.",
    "lines_deleted": "The PR deletes {value:g} lines. This contributed {direction_word} to the model score.",
    "lines_added": "The PR adds {value:g} lines. For this model prediction, this contributed {direction_word}.",
    "total_lines_changed": "The PR changes {value:g} lines in total. In this prediction, that contributed {direction_word}.",
}


def format_duration(value: Any, status: str | None = None) -> str:
    """Format only measured execution time; non-executed states have no duration."""
    if status in {'SKIPPED', 'NOT_CONFIGURED', 'NOT_DEFINED', 'NOT_REQUESTED', 'NO_CANDIDATE', 'UNAVAILABLE'} or value is None:
        return 'Not executed'
    seconds = max(0.0, float(value))
    if seconds >= 1:
        return f'Duration: {seconds:.2f} s'
    if seconds >= 0.01:
        return f'Duration: {seconds:.2f} s'
    return f'Duration: {seconds * 1000:.0f} ms' if seconds > 0 else 'Duration: <0.01 s'


def _feature_item(item: dict[str, Any]) -> dict[str, Any]:
    feature = str(item.get("feature", ""))
    contribution = float(item.get("contribution") or 0)
    value = item.get("value")
    numeric_value = float(value or 0)
    upward = contribution > 0
    template = FEATURE_SENTENCES.get(feature)
    explanation = (
        template.format(
            value=numeric_value,
            direction="increased" if upward else "reduced",
            direction_word="upward" if upward else "downward",
        )
        if template
        else f"For this model prediction, {FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())} contributed {'upward' if upward else 'downward'}."
    )
    return {
        "feature": feature,
        "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
        "value": value,
        "contribution": contribution,
        "explanation": explanation,
    }


def _validation_presentation(validation: dict[str, Any]) -> dict[str, Any]:
    passed = int(validation.get("passed") or 0)
    unavailable = int(validation.get("unavailable") or 0)
    status = validation.get("status")
    if status == "passed" and unavailable:
        label = "PASSED WITH LIMITATIONS"
    elif status == "passed":
        label = "PASSED"
    elif status == "failed":
        label = "FAILED"
    elif status in ("unavailable", "not_run"):
        label = "NOT AVAILABLE"
    else:
        label = str(status or "UNKNOWN").upper()
    labels = {
        "build": "Syntax / Build Validation",
        "unit": "Repository tests",
        "security": "Bandit · Python Static Security",
        "snyk": "Snyk · Dependency/Security Scan",
        "dependency": "Dependency Manifest Validation",
        "dependency_vulnerability": "Dependency vulnerability scan",
        "regression": "AI-generated regression tests",
        "targeted_integration": "AI-generated targeted integration tests",
        "lint": "Ruff",
    }
    # This is the final product-facing validation contract.  Legacy CI
    # integrations may still be present internally for migration compatibility,
    # but they must never leak into the dashboard or presentation API.
    order = ("build", "unit", "regression", "targeted_integration", "lint", "security", "dependency", "snyk")
    checks = validation.get("checks") or {}
    rendered = []
    for key in order:
        if key not in checks:
            continue
        check = checks[key] or {}
        raw_status = check.get("status")
        output=(check.get("stdout_excerpt") or "")+"\n"+(check.get("stderr_excerpt") or "")
        counts=check.get("severity_counts") or {level:int(match.group(1)) if (match:=re.search(rf"{level}:\s*(\d+)",output,re.I)) else 0 for level in ("Low","Medium","High")}
        unavailable_summary = {
            "regression":"This repository does not define a separate regression command. Existing repository tests are executed, and AI-generated targeted regression tests are handled separately.",
            "integration":"This repository does not define a separate integration command; this is a repository capability distinction, not a platform failure.",
            "targeted_integration":"No AI-generated targeted integration test was supplied for this candidate; this stage is not applicable.",
            "dependency_vulnerability":"Dependency vulnerability scan: Not configured",
        }.get(key)
        test_counts = re.search(r"(\d+)\s+passed(?:,\s*(\d+)\s+skipped)?", output, re.I) if key == "unit" else None
        if key == "unit" and raw_status in {"FAILED", "ERROR"}:
            failures = re.search(r"(\d+)\s+failed", output, re.I)
            human_result = (f"FAIL · {failures.group(1)} failed" if failures else str(raw_status))
            if test_counts:
                human_result += f", {test_counts.group(1)} passed"
        elif key == "unit" and test_counts:
            human_result = f"{test_counts.group(1)} passed" + (f", {test_counts.group(2)} GUI test skipped under repository CI convention" if test_counts.group(2) else "")
        elif key == "security" and raw_status in {"PASSED", "FAILED"}:
            gate = "PASS" if raw_status == "PASSED" and counts["High"] == 0 else "FAIL" if raw_status in {"FAILED", "ERROR"} else str(raw_status or "Not available").replace("_", " ").title()
            human_result = f"{gate} · Blocking High findings: {counts['High']} · Advisories: {counts['Medium']} Medium, {counts['Low']} Low"
        elif key in {"snyk"} and raw_status == "UNAVAILABLE":
            human_result = "Not configured"
        elif key in {"regression", "targeted_integration", "integration"} and raw_status == "UNAVAILABLE":
            human_result = "Not applicable" if key == "targeted_integration" else "Not separately defined"
        else:
            human_result = "PASS" if raw_status == "PASSED" else str(raw_status or "Not available").replace("_", " ").title()
        command=check.get("command") or []
        if key=="build" and "compileall" in command: labels[key]="Python Compilation Check"
        why=check.get("verifies") or {"unit":"Exercises the repository's existing automated behavior checks.","security":"The configured deployment gate blocks High-severity findings. Medium and Low findings remain visible for review.","dependency":"Parses manifests only; it does not inspect known vulnerabilities.","dependency_vulnerability":"Reports known dependency vulnerabilities only when a scanner is configured.","regression":"Executes AI-generated defect-specific tests.","targeted_integration":"Executes AI-generated cross-component paths.","integration":"Shows whether the repository defines a dedicated cross-component suite."}.get(key)
        command_display=check.get("command_display") or (" ".join(str(item) for item in command) if command else "No external command")
        if raw_status == "UNAVAILABLE":
            command_display=f"Not run — {unavailable_summary or check.get('summary') or 'no command is configured'}"
        rendered.append({
            "key": key,
            "label": labels[key],
            "status": raw_status,
            "summary": unavailable_summary if raw_status=="UNAVAILABLE" and unavailable_summary else check.get("summary"),
            "human_result": human_result,
            "why_it_matters":why,
            "security_gate":("PASS" if raw_status=="PASSED" and counts["High"]==0 else "FAIL" if raw_status in {"FAILED","ERROR"} else None) if key=="security" else None,
            "severity_counts":counts,
            "evidence":check.get("evidence",{}),
            "duration_seconds": check.get("duration_seconds",check.get("duration")),
            "duration_label": format_duration(check.get('duration_seconds',check.get('duration')), check.get('status')),
            "command":command_display,
            "workspace":check.get("workspace"),
            "variant":normalize_validation_variant(check.get("variant","ORIGINAL")),
            "stdout_excerpt": check.get("stdout_excerpt"),
            "stderr_excerpt": check.get("stderr_excerpt"),
        })
    return {
        "label": label,
        "passed": passed,
        "unavailable": unavailable,
        "summary": f"{passed} applicable checks passed" + (f"; {unavailable} dedicated checks not defined by this repository" if unavailable else ""),
        "checks": rendered,
    }


def _decision_presentation(decision: dict[str, Any], analysis: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    impact = analysis.get("impact_summary") or {}
    risk = analysis.get("risk_prediction") or {}
    recommendation = decision.get("decision")
    reasons: list[str] = []
    if str(impact.get("impact_level", "")).upper() == "HIGH":
        reasons.append("High structural impact")
    if impact.get("impacted_nodes") is not None:
        reasons.append(f"Blast radius affects {int(impact.get('impacted_nodes') or 0)} nodes")
    if impact.get("maximum_depth") is not None:
        reasons.append(f"Maximum dependency depth is {int(impact.get('maximum_depth') or 0)}")
    critical = int(impact.get("critical_nodes") or 0)
    if critical:
        reasons.append(f"{critical} critical component{'s are' if critical != 1 else ' is'} affected")
    if validation.get("status") == "passed":
        reasons.append("Validation passed")
    probability, threshold = risk.get("risk_probability"), risk.get("threshold")
    if probability is not None and threshold is not None and probability < threshold:
        reasons.append("ML risk is below the high-risk threshold")
    if not validation.get("blocking_failures"):
        reasons.append("No blocking validation failures")
    structural = str(impact.get("impact_level", "")).upper() == "HIGH" or int(impact.get("impacted_nodes") or 0) >= 20 or critical > 0
    summary = (
        "Validation passed and the ML risk is low, but the dependency blast radius is large and affects critical components. Review is recommended before deployment."
        if recommendation == "DELAY" and validation.get("status") == "passed" and probability is not None and threshold is not None and probability < threshold and structural
        else " ".join(decision.get("human_readable_reasons") or [])
    )
    return {"recommendation": recommendation, "reasons": reasons, "summary": summary}


def build_presentation(analysis_id: str, row_status: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build copy-ready UI facts without changing or duplicating persisted evidence."""
    workflow = payload if payload.get("workflow_id") else {}
    analysis = workflow.get("analysis") or payload.get("analysis") or payload
    risk = analysis.get("risk_prediction") or {}
    shap = analysis.get("shap") or {}
    impact = analysis.get("impact_summary") or {}
    changes = analysis.get("change_summary") or {}
    validation = workflow.get("validation") or analysis.get("validation") or {}
    decision = workflow.get("decision") or analysis.get("decision") or {}
    deployment = workflow.get("deployment") or {}
    repair = workflow.get("repair") or analysis.get("repair") or {}
    readiness = payload.get("provider_readiness") or workflow.get("provider_readiness") or {}
    approval = workflow.get("approval_metadata") or {}

    probability, threshold = risk.get("risk_probability"), risk.get("threshold")
    classification = None if probability is None else ("HIGH" if threshold is not None and probability >= threshold else "LOW")
    increasing = [_feature_item(item) for item in (shap.get("top_risk_factors") or []) if float(item.get("contribution") or 0) > 0]
    reducing = [_feature_item(item) for item in (shap.get("protective_factors") or []) if float(item.get("contribution") or 0) < 0]
    risk_narrative = None
    if probability is not None and threshold is not None:
        risk_narrative = (
            f"The model estimates a {probability * 100:.1f}% deployment-risk probability, "
            f"{'above' if probability >= threshold else 'below'} the {threshold * 100:.1f}% high-risk threshold, "
            f"so this change is classified as {classification} ML risk."
        )

    graph = analysis.get("graph") or {}
    nodes = graph.get("nodes") or []
    node_by_id = {node.get("id"): node for node in nodes}
    edges = graph.get("edges") or []
    node_details = []
    for node in nodes:
        if not node.get("changed") and not node.get("critical"):
            continue
        related_ids = []
        for edge in edges:
            if edge.get("source") == node.get("id"):
                related_ids.append(edge.get("target"))
            elif edge.get("target") == node.get("id"):
                related_ids.append(edge.get("source"))
        states = []
        if node.get("changed"):
            states.append("Changed by this pull request")
        if node.get("critical"):
            states.append("Classified as critical by the structural impact analysis")
        node_details.append({
            "symbol": node.get("qualified_name") or node.get("label"),
            "file": node.get("file"),
            "module": node.get("module"),
            "changed": bool(node.get("changed")),
            "critical": bool(node.get("critical")),
            "why": "; ".join(states),
            "related": [node_by_id[item].get("qualified_name") or node_by_id[item].get("label") for item in related_ids if item in node_by_id],
        })

    files = int(changes.get("files_changed") or 0)
    symbols = int(changes.get("symbols_changed") or 0)
    impacted = int(impact.get("impacted_nodes") or 0)
    depth = int(impact.get("maximum_depth") or 0)
    critical = int(impact.get("critical_nodes") or 0)
    language = "Python" if "py" in (analysis.get("detected_languages") or []) else "source"
    impact_score = float(impact.get("impact_score") or 0)
    impact_level = str(impact.get("impact_level") or "Not available").upper()
    impact_narrative = (
        f"This PR modifies {symbols} symbols across {files} {language} file{'s' if files != 1 else ''}. "
        f"Changes propagate to {impacted} dependent nodes over {depth} levels and affect {critical} critical "
        f"component{'s' if critical != 1 else ''}, giving a {impact_level} structural impact score of {impact_score:.2f}."
    )

    validation_ui = _validation_presentation(validation)
    decision_ui = _decision_presentation(decision, analysis, validation)
    human_decision = approval.get("decision")
    lifecycle = deployment.get("lifecycle_state")
    if not human_decision and lifecycle == "USER_REJECTED":
        human_decision = "REJECTED"
    final_action = "DEPLOYMENT SIMULATION COMPLETED" if lifecycle == "DEPLOYMENT_COMPLETED" else ("AWAITING YOUR APPROVAL" if lifecycle == "READY_FOR_APPROVAL" else str(lifecycle or "NOT STARTED").replace("_", " "))
    candidate = repair.get("candidate") or {}
    repair_attempted = repair.get("status") in {"error", "available"} or bool(repair.get("proposed_changes") or repair.get("warnings"))
    no_candidate = not repair.get("patch") and candidate.get("status") != "validated"
    if candidate.get("status") == "validated":
        repair_label = "Validated"
        repair_message = repair.get("explanation") or repair.get("issue_summary") or "A validated AI candidate is persisted for this analysis."
    elif readiness.get("status") == "READY" and repair_attempted and no_candidate:
        repair_label = "No validated AI candidate was produced."
        repair_message = (
            "The model analyzed the pull request, but the provider output was malformed or incomplete for an applicable patch. "
            "The original code was preserved, and no unsupported candidate claim was made."
        )
    elif readiness.get("status") == "READY" and repair.get("status") == "unavailable":
        repair_label = "Ready for your request"
        repair_message = "Generate a proposal when you want the configured provider to inspect this analysis. Passing validation does not disable repair."
    elif repair.get("status") == "unavailable" and not repair.get("provider"):
        repair_label = "AI Repair not configured"
        repair_message = "No live code-generation provider is configured."
    else:
        repair_label = str(repair.get("status") or "Not available").replace("_", " ").title()
        repair_message = repair.get("issue_summary") or repair.get("explanation")
    total_lines = int(changes.get("lines_added") or 0) + int(changes.get("lines_deleted") or 0)
    selection = (workflow.get("selection") or {}).get("variant") or (deployment.get("simulation_details") or {}).get("selected_variant")
    test_generation = workflow.get("test_generation") or {}
    test_intentions = list(test_generation.get("generated_tests") or [])
    feedback_persisted = not any("Feedback persistence unavailable" in str(item) for item in (workflow.get("warnings") or []))
    finalized = lifecycle == "DEPLOYMENT_COMPLETED"
    trace_specs = (
        ("PR acquired", "PR_ACQUIRED", bool(analysis)),
        ("Source parsed", "CODE_PARSED", bool(analysis)),
        ("Graph built", "GRAPH_BUILT", bool(graph)),
        ("Impact analyzed", "IMPACT", bool(impact)),
        ("Risk predicted", "RISK", probability is not None),
        ("SHAP explained", "EXPLANATION", bool(shap)),
        ("Decision generated", "DECISION", bool(decision)),
        ("Repair attempted", "REPAIR", repair_attempted),
        ("Regression generation attempted", "TEST_GENERATION", test_generation.get("status") in {"available", "unavailable", "error"} and repair_attempted),
        ("Validation complete", "VALIDATION", bool(validation)),
        ("Variant selected", None, bool(selection)),
        ("Human approval", "USER_APPROVAL", human_decision == "APPROVED"),
        ("Deployment simulation", "DEPLOYMENT", finalized),
        ("Workflow monitoring", "MONITORING", finalized or bool(workflow.get("monitoring"))),
        ("Feedback record", "FEEDBACK", feedback_persisted and bool(workflow)),
    )
    stages = workflow.get("stages") or payload.get("stages") or {}
    trace = []
    for label, stage_key, inferred_complete in trace_specs:
        stage = stages.get(stage_key, {}) if stage_key else {}
        status = stage.get("status")
        if status in {"PASS", "completed"}: status = "PASS"
        elif status not in {"PENDING", "RUNNING", "FAILED", "SKIPPED", "NOT_CONFIGURED", "NOT_DEFINED"}:
            status = "PASS" if inferred_complete else "SKIPPED" if row_status == "FAILED" else "PENDING"
        trace.append({**stage, "stage": label, "status": status, "summary": stage.get("message") or stage.get("summary")})

    return {
        "analysis_id": analysis_id,
        "risk": {
            "probability": probability,
            "threshold": threshold,
            "classification": classification,
            "narrative": risk_narrative,
            "shap_definition": "SHAP explains how each model feature moved this prediction relative to the model's baseline. Positive values push predicted risk upward; negative values push it downward.",
            "increasing_factors": increasing,
            "reducing_factors": reducing,
        },
        "structural_comparison": {
            "title": f"Why is structural impact {str(impact.get('impact_level') or 'unavailable').upper()} while ML risk is {classification or 'unavailable'}?",
            "points": [
                "CatBoost estimates deployment-risk probability using its four runtime statistical features.",
                "Structural Impact measures dependency propagation and blast radius using the custom graph.",
                "They measure different things and are intentionally not forced into the same score.",
                "The Decision Engine combines these signals with validation, critical nodes, and repair availability.",
            ],
            "analysis_specific": f"The ML model classifies the PR as {classification or 'unavailable'} risk. Dependency analysis shows {impacted} impacted nodes, a total blast radius of {impact.get('blast_radius', impacted)}, depth {depth}, and {critical} critical component{'s' if critical != 1 else ''}. The Decision Engine recommends {decision.get('decision') or 'further review'}.",
        },
        "decision": decision_ui,
        "validation": validation_ui,
        "repair": {"label": repair_label, "message": repair_message, "provider":readiness, "attempted": repair_attempted, "candidate_status": (repair.get("candidate") or {}).get("status") or "not_created"},
        "monitoring": {"workflow_status": "ACTIVE", "external_telemetry": "NOT CONFIGURED" if (workflow.get("monitoring") or {}).get("status") != "available" else "AVAILABLE"},
        "impact": {"narrative": impact_narrative, "nodes": node_details},
        "deployment": {**{"system_recommendation": decision.get("decision"), "human_decision": human_decision or "NOT DECIDED", "final_action": final_action, "actor": approval.get("actor"), "timestamp": approval.get("timestamp")},**({"selected_variant":selected_variant} if (selected_variant:=((workflow.get("selection") or {}).get("variant") or (deployment.get("simulation_details") or {}).get("selected_variant"))) else {})},
        "candidate_comparison":workflow.get("candidate_comparison") or {},
        "test_generation": {"intentions": test_intentions, "patch_accepted": test_generation.get("status") == "available" and bool(test_generation.get("generated_test_files")), "message": "Executable test patch was not accepted because the provider did not return a valid test-only unified diff."},
        "feedback":workflow.get("feedback") or {},
        "trace": trace,
        "summary": {
            "change": f"{files} files / {total_lines} lines",
            "structural": str(impact.get("impact_level") or "Not available").upper(),
            "impacted_nodes": impacted,
            "critical_nodes": critical,
            "ml": classification or "Not available",
            "probability": probability,
            "validation": validation_ui["label"].title(),
            "recommendation": decision.get("decision") or "Not available",
            "human_approval": "Approved" if human_decision == "APPROVED" else ("Rejected" if human_decision == "REJECTED" else "Not decided"),
            "deployment": final_action.title(),
            "repair": "Not configured" if repair_label == "AI Repair not configured" else repair_label,
            "selected_variant": selection or "Not selected",
            "workflow_monitoring": "ACTIVE",
            "external_telemetry": "Not configured" if (workflow.get("monitoring") or {}).get("status") != "available" else "Available",
            "analysis_status": row_status,
        },
    }
