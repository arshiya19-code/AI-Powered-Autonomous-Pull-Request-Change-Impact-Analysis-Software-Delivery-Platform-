from __future__ import annotations
from typing import Any, Callable
from pathlib import Path
from .github import SOURCE_SUFFIXES
import inspect
import re
from time import perf_counter
from uuid import uuid4
from backend.app.schemas.models import WorkflowResponse, DecisionResult, RepairResult, TestGenerationResult, ValidationSummary, ValidationCheck
from .analysis import AnalysisService, PRAnalysisRequest
from .deployment import DeploymentService
from .test_generation import TestGenerationService
from .validation import ValidationService
from .monitoring import MonitoringService
from .feedback import FeedbackService
from . import metrics
from .candidate import CandidateWorkspaceService
from .repair import ProviderResponseError, ProviderTransportError
from datetime import datetime, timezone

class DeliveryWorkflowService:
    MAX_REPAIR_ATTEMPTS = 3
    def __init__(self, analysis=None, validation=None, repair=None, tests=None, deployment=None, monitoring=None, feedback=None):
        self.analysis=analysis or AnalysisService(); self.validation=validation or self.analysis.validation
        self.repair=repair or self.analysis.repair; self.tests=tests or TestGenerationService(self.repair); self.deployment=deployment or DeploymentService(); self.monitoring=monitoring or MonitoringService(); self.feedback=feedback or FeedbackService()
        self.candidates = CandidateWorkspaceService()
        self.workflows = {}

    @staticmethod
    def _canonical_candidate_validation(summary: ValidationSummary) -> ValidationSummary:
        """Normalize every check produced by a candidate executor at the boundary."""
        checks = {}
        for name, raw in (summary.checks or {}).items():
            value = raw.model_dump() if isinstance(raw, ValidationCheck) else dict(raw)
            value["check_type"] = value.get("check_type") or name
            value["variant"] = "AI_CANDIDATE"
            checks[name] = ValidationCheck.model_validate(value).model_dump()
        return summary.model_copy(update={"checks": checks})

    @staticmethod
    def _enforce_ai_test_execution(summary: ValidationSummary, generated: TestGenerationResult) -> ValidationSummary:
        """A candidate is not validated unless every generated test class ran and passed."""
        required = []
        if generated.generated_tests:
            required.append("regression")
        if generated.generated_integration_tests:
            required.append("targeted_integration")
        missing = [name for name in required if (summary.checks.get(name) or {}).get("status") != "PASSED"]
        if not missing:
            return summary
        blocking = list(dict.fromkeys([*summary.blocking_failures, *missing]))
        errors = [*summary.errors, *[f"Generated {name.replace('_', ' ')} tests did not complete successfully." for name in missing]]
        return summary.model_copy(update={"status": "failed", "blocking_failures": blocking, "errors": errors})
    def run(self, request: PRAnalysisRequest, *, mode="proposal", validate=False, repository_path=None, notify: Callable[[str, dict], None] | None = None, workflow_id=None) -> WorkflowResponse:
        started=perf_counter(); workflow_id=workflow_id or str(uuid4()); metrics.analyses.labels("started").inc()
        injected_notify = notify
        notify = notify or (lambda event_name, payload: None)
        try: analysis=self.analysis.analyze(request, notify=notify) if injected_notify else self.analysis.analyze(request)
        except Exception:
            metrics.analyses.labels("failed").inc(); metrics.workflow_runs.labels("failed").inc(); raise
        metrics.analyses.labels("completed").inc(); metrics.analysis_duration.observe(perf_counter()-started)
        root=repository_path or request.repository_path
        notify('VALIDATION_STARTED' if validate and root else 'VALIDATION_UNAVAILABLE', {})
        validation=self.validation.validate(root, requested=None, variant="ORIGINAL") if validate and root else self.validation.not_run()
        if validate and root: notify('VALIDATION_COMPLETE', {'status': validation.status, 'message': f'Original validation: {validation.status}', 'blocking_failures': validation.blocking_failures})
        readiness=self.repair.readiness()
        repair=RepairResult(status="unavailable", provider=readiness.get("provider"), model=readiness.get("model"), issue_summary="Ready to generate a user-requested repair proposal." if readiness["status"]=="READY" else "No configured repair provider is currently available.")
        generated=TestGenerationResult(status="unavailable", warnings=["Targeted regression tests are generated with a user-requested repair proposal."])
        repair_signal="not_requested"
        decision=self.analysis.decision.decide(risk=analysis.risk_prediction, impact_score=analysis.impact_summary.impact_score, blast_radius=analysis.impact_summary.blast_radius, critical_nodes=analysis.impact_summary.critical_nodes, analysis_complete=analysis.analysis_complete, validation=validation, repair_state=repair_signal)
        deployment=self.deployment.simulate(decision, validation=validation, analysis_complete=analysis.analysis_complete, risk=analysis.risk_prediction, impact=analysis.impact_summary)
        approvable = validation.status == "passed" and not validation.blocking_failures and decision.decision in ("DEPLOY", "DELAY") and not decision.blocking_factors
        deployment = deployment.model_copy(update={"simulation_status":"BLOCKED", "lifecycle_state": "READY_FOR_APPROVAL" if approvable else "BLOCKED", "simulation_details":{"mode":"simulation","production_changed":False,"message":"No deployment runs before explicit user approval."}, "decision_context":decision.input_signals})
        monitoring=self.monitoring.current(); metrics.recommendations.labels(decision.decision or "unknown").inc(); metrics.deployments.labels(deployment.recommended_action).inc(); metrics.workflow_runs.labels("completed").inc(); metrics.workflow_duration.observe(perf_counter()-started); metrics.validations.labels(validation.status).inc(); metrics.repairs.labels(repair.status).inc()
        warnings=analysis.warnings+repair.warnings+generated.warnings
        try:
            from backend.app.schemas.models import FeedbackRecord
            self.feedback.record(FeedbackRecord(workflow_id=workflow_id, repository=request.repository, pull_request=request.pull_request, features=analysis.features, predicted_risk=analysis.risk_prediction.risk_probability, model_version=analysis.risk_prediction.model_version, shap_summary=analysis.shap.values if analysis.shap and analysis.shap.values else None, decision=decision.decision, validation_outcome=validation.status, original_validation=validation.model_dump(), structural_impact=analysis.impact_summary.model_dump(), repair_outcome="not_requested", deployment_result="AWAITING_APPROVAL" if deployment.lifecycle_state == "READY_FOR_APPROVAL" else "BLOCKED", monitoring_health=monitoring.health_state, rollback_occurred=False))
        except Exception as exc: warnings.append(f"Feedback persistence unavailable: {type(exc).__name__}")
        stages = {name: {"status": "completed" if name not in {"REPAIR", "TEST_GENERATION", "USER_APPROVAL", "DEPLOYMENT", "MONITORING", "FEEDBACK"} else ("ready_for_approval" if name == "USER_APPROVAL" and approvable else "not_started"), "started_at": None, "completed_at": None, "duration": None, "summary": None, "reason_code": None} for name in ("PR_ACQUIRED", "CODE_PARSED", "GRAPH_BUILT", "IMPACT", "RISK", "EXPLANATION", "DECISION", "REPAIR", "TEST_GENERATION", "VALIDATION", "CANDIDATE_COMPARISON", "USER_APPROVAL", "DEPLOYMENT", "MONITORING", "FEEDBACK")}
        stages["VALIDATION"]["summary"] = validation.status
        stages["REPAIR"].update({"status":"awaiting_user","summary":"Provider ready; proposal is user controlled" if readiness["status"]=="READY" else "Provider not configured"})
        stages["TEST_GENERATION"].update({"status":"awaiting_repair","summary":"Generated separately from repository-defined suites"})
        stages["DECISION"]["summary"] = deployment.recommended_action
        stages["CANDIDATE_COMPARISON"].update({"status":"not_started","summary":"Awaiting a user-requested candidate"})
        stages["USER_APPROVAL"]["reason_code"] = deployment.lifecycle_state
        result = WorkflowResponse(workflow_id=workflow_id, analysis=analysis, validation=validation, repair=repair, test_generation=generated, deployment=deployment, decision=decision, pre_repair_decision=decision, reason_codes=decision.reason_codes, warnings=warnings, monitoring=monitoring, stages=stages, approval_state=deployment.lifecycle_state, provider_readiness=readiness)
        self.workflows[workflow_id] = result
        notify('AWAITING_APPROVAL' if result.approval_state == 'READY_FOR_APPROVAL' else 'WORKFLOW_BLOCKED', {})
        return result

    @staticmethod
    def repair_context(result: WorkflowResponse, *, diff: str | None = None) -> dict[str, Any]:
        analysis=result.analysis
        if analysis is None: raise ValueError("Persisted analysis evidence is missing")
        return {
            "analysis_id":result.workflow_id, "repository":analysis.pr_metadata.repository,
            "pull_request":analysis.pr_metadata.pull_request, "diff":diff,
            "pr_title":analysis.pr_metadata.title, "pr_body":analysis.pr_metadata.body,
            "base_sha":analysis.pr_metadata.base_sha, "head_sha":analysis.pr_metadata.head_sha,
            "changed_files":analysis.change_details.get("changed_files", []),
            "changed_symbols":analysis.change_details.get("changed_symbols", []),
            "impacted_symbols":analysis.impact_details.get("impacted_nodes", []),
            "dependency_changes":analysis.dependency_change_summary.model_dump(),
            "propagation_paths":analysis.impact_details.get("propagation_paths", []),
            "critical_nodes":[n for n in (analysis.graph.get("nodes") or []) if n.get("critical")],
            "blast_radius":analysis.impact_summary.blast_radius,
            "impact":analysis.impact_summary.model_dump(), "risk":analysis.risk_prediction.model_dump(),
            "shap":analysis.shap.model_dump() if analysis.shap else None,
            "decision":(result.pre_repair_decision or result.decision).model_dump() if (result.pre_repair_decision or result.decision) else None,
            "validation":result.validation.model_dump(), "detected_languages":analysis.detected_languages,
        }

    @staticmethod
    def source_context(root, limit=24000, relevant_files=None):
        sources = {}; size = 0
        root_path = Path(root)
        requested = []
        relevant_symbols = []
        modified_files = set()
        for item in relevant_files or []:
            name = item.get("file") if isinstance(item, dict) else item
            if isinstance(name, str): requested.append(name.replace('\\', '/'))
            if isinstance(item, dict):
                if item.get("change_type") == "modified" and isinstance(name, str):
                    modified_files.add(name.replace("\\", "/"))
                symbol = item.get("qualified_name") or item.get("name") or item.get("symbol")
                if isinstance(symbol, str) and symbol:
                    relevant_symbols.append(symbol.rsplit(".", 1)[-1])
        paths = []
        for name in sorted(dict.fromkeys(requested), key=lambda name: (CandidateWorkspaceService._is_test_path(name), name not in modified_files, requested.index(name))):
            path = root_path / name
            if path.is_file(): paths.append(path)
        # Add only source files that mention a changed/impacted symbol.  This
        # captures callers needed for a repair while keeping unrelated files out
        # of the provider context.
        related = []
        if relevant_symbols:
            symbol_pattern = re.compile(r"\b(?:" + "|".join(re.escape(value) for value in dict.fromkeys(relevant_symbols)) + r")\b")
            for path in sorted(root_path.rglob('*')):
                if path in paths or not path.is_file() or path.is_symlink() or path.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                if any(part.startswith('.') for part in path.relative_to(root_path).parts) or path.stat().st_size > 100000:
                    continue
                source = path.read_text(encoding='utf-8', errors='replace')
                if symbol_pattern.search(source): related.append(path)
        candidates = [*paths, *related] if paths or related else sorted(root_path.rglob('*'))
        for path in candidates:
            if not path.is_file() or path.is_symlink() or any(part.startswith('.') for part in path.relative_to(root).parts): continue
            if path.suffix.lower() not in SOURCE_SUFFIXES or path.stat().st_size > 100000: continue
            source = path.read_text(encoding='utf-8', errors='replace')
            if size + len(source) > limit: continue
            sources[path.relative_to(root).as_posix()] = source; size += len(source)
        return sources

    @staticmethod
    def candidate_diff(root, candidate, base_sha, changed_files, fallback):
        """Measure the same PR base against candidate source, not the repair delta."""
        if not base_sha or not (Path(root) / '.git').exists(): return fallback
        import subprocess
        import difflib
        chunks = []
        for name in sorted(set(changed_files)):
            previous = subprocess.run(['git', '-c', f'safe.directory={Path(root).resolve()}', 'show', f'{base_sha}:{name}'], cwd=root,
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
            before = previous.stdout if previous.returncode == 0 else ''
            path = Path(candidate) / name
            after = path.read_text(encoding='utf-8') if path.is_file() else ''
            if before == after: continue
            chunks.append(f'diff --git a/{name} b/{name}\n' + ''.join(difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile='a/'+name, tofile='b/'+name)))
        return ''.join(chunks)

    def generate_candidate(self, result: WorkflowResponse, *, root: str, diff: str | None = None) -> WorkflowResponse:
        if result.pre_repair_decision is None: result.pre_repair_decision=result.decision
        context=self.repair_context(result,diff=diff); original_validation=result.validation
        context["source_context"]=self.source_context(root, limit=6500, relevant_files=[*context["changed_files"], *context["changed_symbols"]])
        import subprocess
        context["base_source_context"] = {}
        if context.get("base_sha") and (Path(root) / ".git").exists():
            for name in context["changed_files"]:
                snapshot = subprocess.run(["git", "show", f"{context['base_sha']}:{name}"], cwd=root, capture_output=True, timeout=30)
                if snapshot.returncode == 0:
                    context["base_source_context"][name] = snapshot.stdout.decode("utf-8", errors="replace")
        context["repository_layout"] = sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file() and not any(part.startswith(".") or part in {"node_modules", "__pycache__"} for part in p.relative_to(root).parts))[:200]
        context["request_reason"]="User explicitly requested a repair review because structural impact is high or critical nodes are affected; passing validation must not suppress analysis."
        changed=context["changed_symbols"]; impacted=context["impacted_symbols"]
        test_evidence={
            **context,
            "source_context": self.source_context(root, limit=10000, relevant_files=context["changed_files"]),
            "impacted_symbols": impacted[:10],
            "propagation_paths": context["propagation_paths"][:5],
        }
        attempts=[]; candidate_validation=None; changed_files=[]; written=[]; candidate_files=[]; candidate_reanalysis=None
        repair=RepairResult(); generated=TestGenerationResult()
        candidate_info={"status":"not_created","original_preserved":True,"selectable":False}
        # A user-requested retry of a persisted failed repair should retain the
        # prior rejection evidence instead of repeating the initial prompt.
        failure_evidence = dict((result.repair_attempts[-1].get("failure_feedback") or {}) if result.repair_attempts else {})
        original_targeted=None; preserved_candidate=None
        prior_candidate = result.repair.candidate or {}
        if not prior_candidate.get("persisted_source"):
            prior_attempt = next((item for item in reversed(result.repair_attempts)
                                  if item.get("patch_applicable") and item.get("candidate_created") and item.get("files")), None)
            if prior_attempt:
                prior_candidate = {"persisted_source": True, "files": prior_attempt["files"], "patch": prior_attempt["patch"]}
        frozen_repair = (
            result.repair.model_copy(deep=True, update={"patch": prior_candidate["patch"], "status": "available"})
            if prior_candidate.get("persisted_source") and prior_candidate.get("files") and prior_candidate.get("patch")
            else None
        )
        semantic_attempts = 0
        test_rounds = 0
        max_test_rounds = getattr(self.tests, "MAX_ATTEMPTS", 3)
        for attempt_number in range(1, self.MAX_REPAIR_ATTEMPTS + max_test_rounds + 1):
            attempt_context={**context,"attempt":attempt_number,"max_attempts":self.MAX_REPAIR_ATTEMPTS}
            if failure_evidence:
                # Keep the previous attempt's evidence intact while asking the
                # provider for a genuinely new proposal.
                attempt_context["failure_feedback"]={**failure_evidence,
                    "prior_candidate_patch":failure_evidence.get("prior_candidate_patch"),
                    "prior_repair_explanation":failure_evidence.get("prior_repair_explanation")}
            if frozen_repair is None:
                if semantic_attempts >= self.MAX_REPAIR_ATTEMPTS:
                    break
                semantic_attempts += 1
                repair=self.repair.propose(attempt_context)
            else:
                if test_rounds >= max_test_rounds:
                    break
                repair=frozen_repair.model_copy(deep=True)
            raw_patch=repair.patch if isinstance(repair.patch, str) else None
            patch_nonempty=bool(raw_patch and raw_patch.strip())
            attempt={"attempt":attempt_number,"status":"GENERATION_ATTEMPT","state_history":["GENERATION_ATTEMPT"],
                     "provider":repair.provider,"model":repair.model,"response_metadata":repair.response_metadata,
                     "production_patch_frozen":frozen_repair is not None,"semantic_attempt":semantic_attempts,
                     "provider_status":repair.status,"repair_status":repair.status,
                     "raw_response_available":repair.status in {"available","unavailable","error"},
                     "parsed_patch":patch_nonempty,"patch_nonempty":patch_nonempty,
                     "patch_applicable":False,"candidate_created":False,"changed_files":[],
                     "validation_status":None,"failure_feedback":None,"failure_reason":None,"patch":raw_patch}
            if repair.status == "error" and repair.issue_summary == "Provider returned a rejected patch." and patch_nonempty:
                reason = "; ".join(repair.warnings) or repair.issue_summary or "Provider patch failed the unified-diff contract."
                failure_evidence = {
                    "patch_rejection_reason": reason,
                    "prior_candidate_patch": raw_patch,
                    "prior_repair_explanation": repair.explanation,
                    "repair_instruction": "Generate the entire patch again from scratch. Do not reuse the invalid hunk header. Use a COMPLETE git-compatible unified diff with numeric hunk ranges and unchanged context.",
                }
                attempt.update({
                    "status": "PATCH_REJECTED",
                    "state_history": ["GENERATION_ATTEMPT", "PATCH_REJECTED"],
                    "failure_reason": reason,
                    "failure_feedback": failure_evidence,
                    "patch_validation": "FAIL",
                })
                attempts.append(attempt)
                continue
            if repair.status == "error" and repair.provider_error:
                failure_evidence={"provider_error":repair.provider_error,"warnings":repair.warnings}
                reason=repair.issue_summary or "Provider request failed."
                attempt.update({"status":"PROVIDER_ERROR",
                                "state_history":["GENERATION_ATTEMPT", "PROVIDER_ERROR"],
                                "failure_reason":reason,"failure_feedback":failure_evidence})
                attempts.append(attempt)
                if repair.provider_error.get("type") == "STRUCTURED_OUTPUT_ERROR":
                    failure_evidence["repair_instruction"] = "; ".join(repair.warnings)
                    continue
                semantic_attempts -= 1
                attempt["semantic_attempt"] = None
                # ConfiguredProvider has already exhausted bounded transport
                # retries. Do not burn semantic repair attempts on the same
                # network/rate-limit incident.
                break
            if repair.status != "available" or not patch_nonempty:
                failure_evidence={"provider_failure":repair.issue_summary,"warnings":repair.warnings}
                reason=repair.issue_summary or "No valid repair patch was returned."
                attempt.update({"status":"NO_PATCH_RETURNED",
                                "state_history":["GENERATION_ATTEMPT", "NO_PATCH_RETURNED"],
                                "failure_reason":reason,"failure_feedback":failure_evidence})
                attempts.append(attempt)
                continue
            candidate=None; patch_stage="FAIL"; generated_failed=False
            candidate_validation=None; changed_files=[]; written=[]; candidate_files=[]; generated=TestGenerationResult()
            try:
                candidate=self.candidates.create(root)
                changed_files=self.candidates.apply_patch(candidate,repair.patch); patch_stage="PASS"
                candidate_files=self.candidates.comparison(candidate,changed_files)
                actual_changes=[item for item in candidate_files if item.get("original_code") != item.get("candidate_code")]
                production_changes=[item for item in actual_changes if not self.candidates._is_test_path(item["file"])]
                if not production_changes:
                    raise ValueError("NO_PRODUCTION_CHANGE")
                frozen_repair = repair.model_copy(deep=True)
                test_rounds += 1
                attempt.update({"patch_applicable":True,"changed_files":[item["file"] for item in production_changes],"candidate_created":True})
                attempt["state_history"].append("CANDIDATE_CREATED")
                attempt["files"]=candidate_files
                # Keep source evidence independent from downstream test generation
                # and validation. A valid production patch remains reviewable even
                # when generated-test creation fails afterward.
                preserved_candidate={"candidate_id":str(uuid4()),"workspace":"isolated_disposable_candidate",
                    "workspace_scope":"AI_CANDIDATE","persisted_source":True,"original_preserved":True,
                    "patch":repair.patch,"patched_files":[item["file"] for item in production_changes],
                    "files":[dict(item) for item in candidate_files],"affected_symbols":changed,
                    "attempt":attempt_number}
                attempt["state_history"].append("TEST_GENERATION_STARTED")
                patch_symbols = set(re.findall(r"^[ +\-]*\s*(?:async\s+)?def\s+(\w+)\(", repair.patch, re.MULTILINE))
                relevant_test_sources = [item for item in changed if isinstance(item, dict)
                    and item.get("file") in repair.affected_files
                    and (not patch_symbols or str(item.get("qualified_name") or item.get("name") or "").rsplit(".", 1)[-1] in patch_symbols)]
                relevant_test_sources = relevant_test_sources or repair.affected_files
                if repair.regression_test:
                    generated_failed = True
                    generated=self.tests.from_repair(
                        repair, root=candidate.path, changed_symbols=changed,
                        impacted_symbols=impacted[:20],
                    )
                    generated_failed = False
                else:
                    generated=self.tests.generate(root=candidate.path,changed_symbols=changed,impacted_symbols=impacted[:20],language=context["detected_languages"],repair=repair.model_dump(),analysis_context={**test_evidence,"candidate_source":self.source_context(candidate.path, limit=10000, relevant_files=relevant_test_sources),"failure_feedback":failure_evidence})
                attempt["test_generation_attempts"] = generated.generation_attempts
                if generated.status != "available" or not generated.generated_test_files:
                    generated_failed = True
                    raise ValueError("TEST_GENERATION_FAILED: " + "; ".join(generated.warnings))
                written=self.candidates.add_generated_tests(candidate, generated)
                # Persist the exact validated complete source written only to the candidate.
                for category in (generated.generated_tests, generated.generated_integration_tests):
                    for item in category:
                        name = item.get("test_file") or item.get("path")
                        if not name or name.replace('\\', '/') not in written:
                            raise ValueError('Generated test metadata must identify an applied test file')
                        item["generated_code"]=(candidate.path / name).read_text(encoding='utf-8')
                        generated.generated_test_files[name]=item["generated_code"]
                        item.update({"provider":item.get("provider") or repair.provider,"model":item.get("model") or repair.model,"workflow_attempt":attempt_number,
                                     "path_validation":"PASSED","syntax_validation":"PASSED",
                                     "candidate_only_write":True,"workspace":"isolated_disposable_candidate"})
                written=list(dict.fromkeys(written))
                attempt["generated_test_artifacts"] = generated.model_dump()
                candidate_files=self.candidates.comparison(candidate,changed_files)
                for item in candidate_files:
                    item.update({"affected_symbol":next((s.get("qualified_name") or s.get("name") for s in changed if isinstance(s,dict) and s.get("file")==item["file"]),None),"issue_being_fixed":repair.issue_summary,"ai_explanation":repair.explanation,"confidence":repair.confidence})
                integration_files=[item.get("test_file") for item in generated.generated_integration_tests if item.get("test_file")]
                regression_files=[item.get("test_file") for item in generated.generated_tests if item.get("test_file")]
                if isinstance(self.validation, ValidationService):
                    baseline = self.candidates.create(root)
                    try:
                        self.candidates.add_generated_tests(baseline, generated)
                        original_regression = self.validation.validate(baseline.path,
                            requested=['regression'], generated_test_files=regression_files or written,
                            generated_integration_files=integration_files, variant='ORIGINAL')
                        original_integration = self.validation.validate(baseline.path,
                            requested=['targeted_integration'], generated_test_files=regression_files or written,
                            generated_integration_files=integration_files, variant='ORIGINAL')
                        original_targeted = self.validation.aggregate({
                            key: ValidationCheck.model_validate(value)
                            for key, value in {**original_regression.checks, **original_integration.checks}.items()
                        })
                    finally:
                        baseline.cleanup()
                attempt["state_history"].append("CANDIDATE_VALIDATION_STARTED")
                parameters=inspect.signature(self.validation.validate).parameters
                kwargs={"generated_test_files":regression_files or written}
                if "generated_integration_files" in parameters: kwargs["generated_integration_files"]=integration_files
                if "variant" in parameters: kwargs["variant"]="AI_CANDIDATE"
                if "ci_context" in parameters:
                    kwargs["ci_context"]={"analysis_id":result.workflow_id,"candidate_id":str(uuid4()),"attempt":attempt_number,"base_sha":context["base_sha"],"head_sha":context["head_sha"]}
                    kwargs["baseline_root"]=root
                production_before = self.candidates.production_snapshot(candidate.path)
                candidate_validation=self._canonical_candidate_validation(self.validation.validate(candidate.path,**kwargs))
                production_after = self.candidates.production_snapshot(candidate.path)
                if production_before != production_after:
                    mutated = sorted(name for name in production_before.keys() | production_after.keys()
                                     if production_before.get(name) != production_after.get(name))
                    raise ValueError("Generated tests or validation modified production source: " + ", ".join(mutated))
                candidate_validation=self._enforce_ai_test_execution(candidate_validation, generated)
                regression=(candidate_validation.checks.get("regression") or {})
                execution_status="PASS" if regression.get("status")=="PASSED" else "FAIL" if regression.get("status") in {"FAILED","ERROR"} else "UNAVAILABLE"
                execution_output=(regression.get("stdout_excerpt") or "")+(regression.get("stderr_excerpt") or "")
                generated.generated_tests=[{**item,"command":regression.get("command_display") or regression.get("command"),"execution_status":execution_status,"execution_output":execution_output or regression.get("summary",""),"duration_seconds":regression.get("duration_seconds",regression.get("duration",0)),"stdout_stderr_summary":execution_output or regression.get("summary","")} for item in generated.generated_tests]
                integration=(candidate_validation.checks.get("targeted_integration") or {})
                integration_status="PASS" if integration.get("status")=="PASSED" else "FAIL" if integration.get("status") in {"FAILED","ERROR"} else "NOT_APPLICABLE"
                integration_output=(integration.get("stdout_excerpt") or "")+(integration.get("stderr_excerpt") or "")
                generated.generated_integration_tests=[{**item,"command":integration.get("command_display") or integration.get("command"),"execution_status":integration_status,"execution_output":integration_output or integration.get("summary",""),"duration_seconds":integration.get("duration_seconds",integration.get("duration",0)),"stdout_stderr_summary":integration_output or integration.get("summary","")} for item in generated.generated_integration_tests]
                passed=candidate_validation.status=="passed" and not candidate_validation.blocking_failures
                terminal="CANDIDATE_VALIDATED" if passed else "CANDIDATE_VALIDATION_FAILED"
                attempt.update({"status":terminal,"state_history":[*attempt["state_history"],terminal],"validation":candidate_validation.model_dump(),
                                "validation_status":candidate_validation.status,"failed_checks":candidate_validation.blocking_failures})
                attempts.append(attempt)
                if passed:
                    try:
                        sources=self.source_context(candidate.path, limit=5000000)
                        measured_diff=self.candidate_diff(root, candidate.path, context['base_sha'], [*context['changed_files'], *changed_files], repair.patch)
                        measured=self.analysis.analyze(PRAnalysisRequest(repository=context["repository"],pull_request=context["pull_request"],repository_path=str(candidate.path),diff=measured_diff,sources=sources,base_sources=context.get("base_source_context") or {},head_sources=sources))
                        candidate_reanalysis={"status":"AVAILABLE","structural_impact":measured.impact_summary.model_dump(),"ml_risk":measured.risk_prediction.model_dump(),"feature_fingerprint":self.feedback.fingerprint(measured.features),"feature_count":len(measured.features),"features":measured.features,"graph":measured.graph,"impact_details":measured.impact_details,"shap":measured.shap.model_dump() if measured.shap else None,"analysis_complete":measured.analysis_complete,"decision":measured.decision.model_dump()}
                    except Exception as exc:
                        candidate_reanalysis={"status":"UNAVAILABLE","reason":f"Deterministic candidate re-analysis could not be produced: {type(exc).__name__}"}
                    if candidate_reanalysis['status'] != 'AVAILABLE':
                        attempt.update(status='CANDIDATE_VALIDATION_FAILED', failure_reason=candidate_reanalysis['reason'])
                        attempt['state_history'][-1] = 'CANDIDATE_VALIDATION_FAILED'
                        failure_evidence={'candidate_reanalysis': candidate_reanalysis}
                        attempt['failure_feedback'] = failure_evidence
                        continue
                    candidate_info={"status":"validated","label":"VALIDATED CANDIDATE","selectable":candidate_reanalysis.get("status")=="AVAILABLE","candidate_id":str(uuid4()),"workspace":"isolated_disposable_candidate","workspace_scope":"AI_CANDIDATE","persisted_source":True,"original_preserved":True,"patch":repair.patch,"patched_files":changed_files,"files":candidate_files,"affected_symbols":changed,"generated_test_files":written,"generated_test_source":generated.generated_test_files,"validation":candidate_validation.model_dump(),"reanalysis":candidate_reanalysis,"attempt":attempt_number}
                    break
                failed_checks={name:value for name,value in candidate_validation.checks.items() if name in candidate_validation.blocking_failures}
                failure_text="\n".join((value.get("stdout_excerpt") or "")+"\n"+(value.get("stderr_excerpt") or "") for value in failed_checks.values())
                expected_actual=re.search(r"(?:Expected|assert)\s*[:=]?\s*([^\n]+).*?(?:Actual|==)\s*[:=]?\s*([^\n]+)",failure_text,re.I|re.S)
                failed_test_names=[t.get("test_name") for t in [*generated.generated_tests,*generated.generated_integration_tests] if t.get("execution_status")=="FAIL"]
                # Preserve concrete repository-test evidence as well as AI-test
                # evidence. This is the retry prompt's source of truth.
                all_failed_checks={name:value for name,value in candidate_validation.checks.items() if name in candidate_validation.blocking_failures}
                all_failure_text="\n".join((value.get("stdout_excerpt") or "")+"\n"+(value.get("stderr_excerpt") or "") for value in all_failed_checks.values())
                test_names=re.findall(r"(?:[\w./\\-]+::test_[\w]+|test_[\w]+)", all_failure_text)
                expected_actual=re.search(r"(?:Expected|assert)\s*[:=]?\s*([^\n]+?)(?:\s+==\s+|\s+but\s+was\s+|\n.*?Actual\s*[:=]?\s*)([^\n]+)", all_failure_text,re.I|re.S)
                failure_evidence={"failed_test_names":list(dict.fromkeys([*failed_test_names,*test_names])),
                    "failed_commands":[v.get("command_display") or v.get("command") for v in all_failed_checks.values()],
                    "validation_failures":candidate_validation.blocking_failures,
                    "expected":expected_actual.group(1).strip() if expected_actual else None,
                    "actual":expected_actual.group(2).strip() if expected_actual else None,
                    "assertion_or_traceback_excerpt":all_failure_text[-4000:],"checks":all_failed_checks,
                    "prior_candidate_patch":repair.patch,"prior_repair_explanation":repair.explanation}
                attempt["failure_feedback"] = failure_evidence
                if not set(candidate_validation.blocking_failures) & {"regression", "targeted_integration"}:
                    # New tests cannot repair a repository lint/security/build
                    # failure. Keep the accepted source and its actual checks.
                    break
            except ProviderTransportError as exc:
                failure_evidence={"provider_error": {
                    "type": exc.error_type, "status_code": exc.status_code,
                    "retryable": exc.retryable, "retry_after_seconds": exc.retry_after_seconds,
                    "transport_attempts": exc.transport_attempts,
                }}
                attempt.update({"status":"PROVIDER_ERROR",
                                "state_history":[*attempt["state_history"],"PROVIDER_ERROR"],
                                "failure_reason":str(exc),"failure_feedback":failure_evidence,
                                "patch_validation":patch_stage})
                attempt["test_provider_attempts"] = getattr(exc, "provider_attempts", [])
                attempt["test_generation_attempts"] = getattr(exc, "test_generation_attempts", [])
                attempts.append(attempt)
                break
            except (ValueError, OSError, ProviderResponseError) as exc:
                failure_evidence={"candidate_error":str(exc),
                                  "prior_candidate_patch":repair.patch,
                                  "prior_repair_explanation":repair.explanation}
                if getattr(exc, "diagnostic", None):
                    failure_evidence["patch_diagnostic"] = exc.diagnostic
                    failure_evidence["repair_instruction"] = "Regenerate a COMPLETE unified diff using the rejection reason above; do not return a snippet or placeholder."
                attempt["failure_feedback"] = failure_evidence
                state="TEST_GENERATION_FAILED" if generated_failed or ("TEST_GENERATION_STARTED" in attempt["state_history"] and "CANDIDATE_VALIDATION_STARTED" not in attempt["state_history"]) else ("CANDIDATE_VALIDATION_FAILED" if "CANDIDATE_VALIDATION_STARTED" in attempt["state_history"] else "PATCH_REJECTED")
                reason=str(exc)
                if reason == "NO_PRODUCTION_CHANGE": state="PATCH_REJECTED"
                elif state == "PATCH_REJECTED" and "failed validation" in reason.lower():
                    reason="PATCH_PARSE_FAILED: " + reason
                elif state == "PATCH_REJECTED" and "could not be applied" in reason.lower():
                    reason="PATCH_NOT_APPLICABLE: " + reason
                attempt.update({"status":state,"state_history":[*attempt["state_history"],state],"failure_reason":reason,"patch_validation":patch_stage,
                                "failure_feedback":failure_evidence})
                attempts.append(attempt)
                if state == "TEST_GENERATION_FAILED":
                    # TestGenerationService already used its own three retries.
                    break
            finally:
                if candidate: candidate.cleanup()
        if candidate_info["status"] != "validated" and (preserved_candidate or any(a["status"] not in {"NO_PATCH_RETURNED","EMPTY_PATCH","GENERATION_ATTEMPT","PROVIDER_ERROR"} for a in attempts)):
            latest=(attempts[-1].get("failure_reason") or ", ".join(attempts[-1].get("failed_checks") or []) or "candidate validation still fails") if attempts else "no candidate was generated"
            candidate_info={**(preserved_candidate or {}),"status":"rejected","label":"REJECTED","selectable":False,
                "original_preserved":True,"generated_test_files":[],
                "generated_test_status":generated.status,
                "generated_test_evidence":generated.model_dump(),
                "validation":candidate_validation.model_dump() if candidate_validation else None,
                "reanalysis":candidate_reanalysis,
                "attempt":len(attempts),"reason":f"{len(attempts)} AI generation attempts were made. No selectable candidate is available. Latest evidence: {latest}."}
        elif candidate_info["status"] != "validated":
            latest=(attempts[-1].get("failure_reason") if attempts else None) or "No usable production patch was returned."
            candidate_info={"status":"not_created","label":"NO CANDIDATE CREATED","selectable":False,"original_preserved":True,"patch":None,"files":[],"attempt":len(attempts),"reason":f"AI candidate not created. {latest}"}
        repair.candidate=candidate_info
        if candidate_info["status"] != "validated":
            # A failed AI candidate must not inherit the pre-repair approval
            # state. Original-vs-candidate selection remains explicit elsewhere.
            result.approval_state="BLOCKED"
            result.stages.setdefault("USER_APPROVAL", {})["status"]="blocked"
            result.stages["USER_APPROVAL"]["reason_code"]="CANDIDATE_VALIDATION_FAILED"
            if result.deployment:
                result.deployment=result.deployment.model_copy(update={"lifecycle_state":"BLOCKED","simulation_status":"BLOCKED"})
            repair.patch=None
        candidate_passed=candidate_info.get('selectable', False)
        if candidate_passed and original_validation.status!="passed": recommendation="USE_CANDIDATE"; narrative="The candidate passes every applicable blocking gate while the original has blocking validation evidence."
        elif candidate_passed and (candidate_reanalysis or {}).get("status")=="AVAILABLE":
            original_impact=context["impact"].get("impact_score"); candidate_impact=(candidate_reanalysis.get("structural_impact") or {}).get("impact_score")
            original_risk=context["risk"].get("risk_probability"); candidate_risk=(candidate_reanalysis.get("ml_risk") or {}).get("risk_probability")
            comparable=all(value is not None for value in (original_impact,candidate_impact,original_risk,candidate_risk))
            if comparable and candidate_impact < original_impact and candidate_risk < original_risk: recommendation="USE_CANDIDATE"; narrative="Both variants pass, and deterministic candidate re-analysis measures lower structural impact and lower ML risk."
            elif comparable and (candidate_impact > original_impact or candidate_risk > original_risk): recommendation="USE_ORIGINAL"; narrative="Both variants pass, but deterministic candidate re-analysis measures a higher structural-impact or ML-risk signal."
            else: recommendation="REVIEW_REQUIRED"; narrative="Both variants pass, but the measured structural-impact and ML-risk evidence does not establish that either variant is better."
        elif candidate_passed: recommendation="REVIEW_REQUIRED"; narrative="Both variants pass applicable validation. Candidate structural and ML risk could not be measured, so human review is required."
        else: recommendation="USE_ORIGINAL"; narrative=candidate_info.get("reason","No validated candidate is available; keep original.")
        measured_candidate=candidate_reanalysis or {"status":"UNAVAILABLE","reason":"No validated candidate was available for deterministic re-analysis."}
        improvements=[]; regressions=[]; unchanged=["Original evidence remains immutable and distinct from candidate evidence."]
        if original_targeted and candidate_validation:
            for key in ('regression', 'targeted_integration'):
                before=original_targeted.checks.get(key, {}).get('status')
                after=candidate_validation.checks.get(key, {}).get('status')
                if before == 'FAILED' and after == 'PASSED': improvements.append(f'{key}: identical generated tests failed on original and passed on candidate.')
                elif before == 'PASSED' and after == 'FAILED': regressions.append(f'{key}: generated tests passed on original and failed on candidate.')
                elif before == after and before in ('PASSED', 'FAILED'): unchanged.append(f'{key}: both variants measured {before}.')
        if candidate_passed and original_validation.status != "passed": improvements.append("Candidate clears blocking validation that the original did not pass.")
        if candidate_passed and (candidate_reanalysis or {}).get("status") == "AVAILABLE":
            original_impact=context["impact"].get("impact_score"); candidate_impact=(candidate_reanalysis.get("structural_impact") or {}).get("impact_score")
            original_risk=context["risk"].get("risk_probability"); candidate_risk=(candidate_reanalysis.get("ml_risk") or {}).get("risk_probability")
            for label, original_value, candidate_value in (("Structural impact",original_impact,candidate_impact),("ML risk",original_risk,candidate_risk)):
                if original_value is None or candidate_value is None: continue
                if candidate_value < original_value: improvements.append(f"{label} decreased from {original_value:.4f} to {candidate_value:.4f}.")
                elif candidate_value > original_value: regressions.append(f"{label} increased from {original_value:.4f} to {candidate_value:.4f}.")
                else: unchanged.append(f"{label} remained {original_value:.4f}.")
        if candidate_validation and original_validation.checks.get("unit",{}).get("status") == candidate_validation.checks.get("unit",{}).get("status"):
            unchanged.append("Existing repository tests have the same pass/fail outcome for both variants.")
        comparison={"original":{"diff":diff,"shap":context["shap"],"decision":context["decision"],"source":[{"file":f.get("file"),"code":f.get("original_code")} for f in candidate_files],"structural_impact":context["impact"],"ml_risk":context["risk"],"build":original_validation.checks.get("build"),"repository_tests":original_validation.checks.get("unit"),"regression_tests":original_targeted.checks.get("regression") if original_targeted else None,"targeted_integration_tests":original_targeted.checks.get("targeted_integration") if original_targeted else None,"repository_integration_tests":original_validation.checks.get("integration"),"security":original_validation.checks.get("security"),"dependency_manifest":original_validation.checks.get("dependency"),"dependency_vulnerability":original_validation.checks.get("dependency_vulnerability"),"validation":original_validation.model_dump()},"candidate":{"patch":candidate_info.get("patch"),"shap":measured_candidate.get("shap"),"decision":measured_candidate.get("decision"),"source":[{"file":f.get("file"),"code":f.get("candidate_code")} for f in candidate_files],"structural_impact":measured_candidate.get("structural_impact") or measured_candidate,"ml_risk":measured_candidate.get("ml_risk") or measured_candidate,"feature_fingerprint":measured_candidate.get("feature_fingerprint"),"build":(candidate_validation.checks.get("build") if candidate_validation else None),"repository_tests":(candidate_validation.checks.get("unit") if candidate_validation else None),"regression_tests":(candidate_validation.checks.get("regression") if candidate_validation else None),"targeted_integration_tests":(candidate_validation.checks.get("targeted_integration") if candidate_validation else None),"repository_integration_tests":(candidate_validation.checks.get("integration") if candidate_validation else None),"security":(candidate_validation.checks.get("security") if candidate_validation else None),"dependency_manifest":(candidate_validation.checks.get("dependency") if candidate_validation else None),"dependency_vulnerability":(candidate_validation.checks.get("dependency_vulnerability") if candidate_validation else None),"patch_summary":{"files":changed_files,"additions":sum(1 for x in (repair.patch or '').splitlines() if x.startswith('+') and not x.startswith('+++')),"deletions":sum(1 for x in (repair.patch or '').splitlines() if x.startswith('-') and not x.startswith('---'))},"validation":candidate_validation.model_dump() if candidate_validation else None,"generated_tests":generated.model_dump()},"measured_improvements":improvements,"measured_regressions":regressions,"unchanged_evidence":unchanged,"recommendation":recommendation,"narrative":narrative}
        post=DecisionResult(decision=recommendation,status="decided",reason_codes=["ALL_CANDIDATE_GATES_PASSED"] if candidate_passed else ["CANDIDATE_REJECTED"],human_readable_reasons=[narrative],blocking_factors=[] if candidate_passed else ["CANDIDATE_VALIDATION_FAILED"],input_signals={"candidate_status":candidate_info["status"],"attempts":len(attempts)})
        checks=candidate_validation.checks if candidate_validation else {}
        specs=[("Candidate workspace / patch validation",None,"Creates an isolated workspace and rejects malformed or out-of-scope production patches."),("Python Compilation","build","Detects Python syntax and bytecode-compilation errors; it is not a package build."),("Existing repository PyTest","unit","Protects behavior already covered by the repository."),("AI regression tests","regression","Executes generated code that reproduces the repaired defect."),("AI targeted integration tests","targeted_integration","Exercises generated cross-component paths when applicable."),("Ruff","lint","Runs Ruff against the complete isolated candidate workspace."),("Bandit","security","Blocks High findings while retaining Medium and Low advisories."),("Dependency manifest validation","dependency","Parses dependency manifests; it is not a vulnerability scan."),("Snyk","snyk","Introduced Critical/High block; Medium/Low advisory."),("Candidate deterministic re-analysis",None,"Recomputes structural and ML evidence for the candidate." )]
        if "sonarqube" in checks:
            specs.insert(-1, ("SonarQube Analysis", "sonarqube", "Executes the configured quality gate."))
        ci=[]; blocked=False
        no_candidate_reason="No valid AI candidate was created."
        for index,(name,key,why) in enumerate(specs):
            evidence=(checks.get(key) or {}) if key else {}
            if index==0: status="PASS" if changed_files else "SKIPPED"
            elif name=="Candidate deterministic re-analysis":
                status = "PASS" if (candidate_reanalysis or {}).get("status")=="AVAILABLE" else ("SKIPPED" if blocked or candidate_info.get("status") != "validated" else "FAIL")
            elif not evidence and candidate_info.get("status") != "validated": status="SKIPPED"
            elif blocked: status="SKIPPED"
            else:
                status=str(evidence.get("status","UNAVAILABLE")).replace("PASSED","PASS").replace("FAILED","FAIL")
                if status == "UNAVAILABLE":
                    status = "NOT DEFINED" if key == "targeted_integration" or (evidence.get("evidence") or {}).get("state") == "NOT_DEFINED" else "NOT CONFIGURED" if key == "snyk" else "UNAVAILABLE"
            if status in {"FAIL","ERROR"}: blocked=True
            if index == 0 and status == "PASS": what_ran=f"Validated and applied the provider patch to an isolated candidate workspace ({len(changed_files)} production file(s))."
            elif index == 0 and status == "SKIPPED": what_ran="Skipped because no candidate patch was available."
            elif candidate_info.get("status") != "validated" and status == "SKIPPED": what_ran=no_candidate_reason
            elif status in {"SKIPPED","NOT DEFINED","NOT CONFIGURED","UNAVAILABLE"}: what_ran=evidence.get("summary") or "Not reached because an earlier blocking stage failed."
            else: what_ran=evidence.get("command_display") or evidence.get("summary") or "Persisted workflow operation"
            ci.append({"stage":name,"status":status,"what_ran":what_ran,"duration_seconds":evidence.get("duration_seconds",evidence.get("duration")),"why_it_matters":why,"workspace":evidence.get("workspace"),"variant":evidence.get("variant","AI_CANDIDATE")})
        result.repair=repair; result.test_generation=generated; result.repair_attempts=attempts; result.ci_pipeline=ci; result.candidate_comparison=comparison; result.post_candidate_decision=post; result.provider_readiness=self.repair.readiness()
        result.stages.setdefault("REPAIR",{}).update({"status":"PASS" if preserved_candidate else "FAILED","summary":attempts[-1]["status"] if attempts else repair.status})
        result.stages.setdefault("TEST_GENERATION",{}).update({"status":"PASS" if generated.status=="available" else "FAILED" if preserved_candidate else "SKIPPED","summary":generated.status})
        result.stages.setdefault("VALIDATION",{}).update({"status":"PASS" if candidate_validation and candidate_validation.status=="passed" else "FAILED" if candidate_validation else "SKIPPED","summary":candidate_validation.status if candidate_validation else "No candidate validation result"})
        result.stages.setdefault("CANDIDATE_COMPARISON",{}).update({"status":"completed","summary":recommendation})
        try:
            from backend.app.schemas.models import FeedbackRecord
            previous=self.feedback.get(result.workflow_id); values=previous.model_dump() if previous else {"workflow_id":result.workflow_id,"repository":result.analysis.pr_metadata.repository,"pull_request":result.analysis.pr_metadata.pull_request}
            values.update({"repair_outcome":repair.status,"original_validation":original_validation.model_dump(),"candidate_validation":candidate_validation.model_dump() if candidate_validation else None,"structural_impact":result.analysis.impact_summary.model_dump(),"metadata":{**(previous.metadata if previous else {}),"test_generation_status":generated.status,"candidate_status":candidate_info["status"],"post_candidate_decision":post.decision}})
            self.feedback.record(FeedbackRecord.model_validate(values))
        except Exception as exc:
            result.warnings.append(f"Feedback persistence unavailable: {type(exc).__name__}")
        return result

    def approve(self, workflow_id: str) -> WorkflowResponse:
        result = self.workflows.get(workflow_id)
        if result is None: raise KeyError("Workflow not found")
        if result.approval_state != "READY_FOR_APPROVAL": raise ValueError("Workflow is not awaiting approval")
        stages = result.stages or {}
        for name in ("USER_APPROVAL", "DEPLOYMENT", "MONITORING"):
            stages.setdefault(name, {"status": "not_started", "reason_code": None})
        result.stages = stages
        selected=(result.selection or {}).get("variant")
        if selected not in {"ORIGINAL", "AI_CANDIDATE"}: raise ValueError("A human must select a variant before approval")
        # Approval gates are evaluated against the selected variant. Original
        # blockers remain visible in comparison, but cannot veto a validated
        # candidate selected by the user.
        selected_decision = result.decision
        if selected == "AI_CANDIDATE":
            reanalysis = (result.repair.candidate or {}).get("reanalysis") or {}
            selected_decision = DecisionResult.model_validate(reanalysis.get("decision") or {})
        if not result.deployment or selected_decision.decision not in ("DEPLOY", "DELAY") or selected_decision.blocking_factors: raise ValueError("Selected variant recommendation does not permit deployment")
        selected_validation = result.validation if selected == "ORIGINAL" else ValidationSummary.model_validate((result.repair.candidate or {}).get("validation") or {})
        if selected == "AI_CANDIDATE" and not result.repair.candidate.get("selectable"): raise ValueError("The AI candidate has not passed every blocking validation gate")
        if selected_validation.status != "passed" or selected_validation.blocking_failures: raise ValueError("Selected variant validation is not in an acceptable passed state")
        result.approval_state = "USER_APPROVED"
        now = datetime.now(timezone.utc)
        history = ["READY_FOR_APPROVAL", "USER_APPROVED", "DEPLOYMENT_STARTED", "DEPLOYMENT_COMPLETED"]
        # Human approval authorizes the simulation; it does not rewrite the
        # Decision Engine's persisted recommendation (for example, DELAY).
        result.deployment = result.deployment.model_copy(update={"simulation_status":"SIMULATED", "readiness":True, "blocking_reasons":[], "recommended_actions":["monitor simulated deployment"], "lifecycle_state":"DEPLOYMENT_COMPLETED", "lifecycle_history":history, "started_at":now, "finished_at":now, "simulation_details":{"mode":"simulation","production_changed":False,"selected_variant":selected,"message":f"Approved deployment simulation completed safely using {selected.replace('_',' ').title()}."}})
        result.stages["USER_APPROVAL"]["status"] = "completed"
        result.stages["USER_APPROVAL"]["reason_code"] = "USER_APPROVED"
        result.stages["DEPLOYMENT"]["status"] = "completed"
        result.stages["MONITORING"]["status"] = "completed"
        result.monitoring = result.monitoring.model_copy(update={"deployment_id":result.workflow_id})
        result.approval_state = "DEPLOYMENT_COMPLETED"
        self._record_action(result, "USER_APPROVED")
        return result

    def reject(self, workflow_id: str) -> WorkflowResponse:
        result = self.workflows.get(workflow_id)
        if result is None: raise KeyError("Workflow not found")
        if result.approval_state != "READY_FOR_APPROVAL": raise ValueError("Workflow is not awaiting approval")
        stages = result.stages or {}
        stages.setdefault("USER_APPROVAL", {"status": "not_started", "reason_code": None})
        result.stages = stages
        result.approval_state = "USER_REJECTED"
        result.deployment = result.deployment.model_copy(update={"lifecycle_state":"USER_REJECTED", "lifecycle_history":["READY_FOR_APPROVAL", "USER_REJECTED"]}) if result.deployment else None
        result.stages["USER_APPROVAL"]["status"] = "rejected"
        result.stages["USER_APPROVAL"]["reason_code"] = "USER_REJECTED"
        self._record_action(result, "USER_REJECTED")
        return result

    def _record_action(self, result, action: str) -> None:
        from backend.app.schemas.models import FeedbackRecord
        previous = self.feedback.get(result.workflow_id)
        values = previous.model_dump() if previous else {"workflow_id":result.workflow_id, "repository":result.analysis.pr_metadata.repository if result.analysis else "local"}
        values.update({"decision":result.decision.decision if result.decision else None,
                       "validation_outcome":result.validation.status,
                       "selected_variant":(result.selection or {}).get("variant"),
                       "approval":action,
                       "deployment_result":"DEPLOYMENT_COMPLETED" if action == "USER_APPROVED" else action,
                       "actual_deployment_outcome":"SIMULATED_DEPLOYED" if action == "USER_APPROVED" else "NOT_DEPLOYED",
                       "metadata":{**(previous.metadata if previous else {}), "approval_state":action,
                                   "deployment_mode":"simulation", "timestamp":datetime.now(timezone.utc).isoformat()}})
        self.feedback.record(FeedbackRecord.model_validate(values))
