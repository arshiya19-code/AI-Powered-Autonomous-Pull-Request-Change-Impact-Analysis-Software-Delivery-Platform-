from backend.app.schemas.models import DecisionResult, DeploymentSummary, DeploymentSimulationResult, ValidationSummary
class DeploymentService:
    def simulate(self, decision: DecisionResult | None = None, *, validation: ValidationSummary | None = None, analysis_complete=True, post_deployment_regression=False, risk=None, impact=None):
        if post_deployment_regression: action="ROLLBACK"; reasons=["POST_DEPLOYMENT_REGRESSION"]
        elif validation and validation.blocking_failures: action="BLOCK"; reasons=[f"{x.upper()}_FAILED" for x in validation.blocking_failures]
        elif not analysis_complete: action="DELAY"; reasons=["ANALYSIS_INCOMPLETE"]
        elif decision and decision.decision in ("BLOCK","DELAY"): action=decision.decision; reasons=list(decision.reason_codes)
        else: action="DEPLOY"; reasons=[]
        return DeploymentSimulationResult(simulation_status="BLOCKED" if action in ("BLOCK","ROLLBACK") else "SIMULATED", recommended_action=action, readiness=action=="DEPLOY", blocking_reasons=reasons, warnings=["Simulation only; no production infrastructure was changed."], evidence_summary={"analysis_complete":analysis_complete,"model_status":getattr(risk,"model_status", "unavailable")}, risk_summary={"model_status":getattr(risk,"model_status", "unavailable"),"risk_probability":getattr(risk,"risk_probability", None),"blast_radius":getattr(impact,"blast_radius", None) if impact else None}, recommended_actions=["resolve blocking evidence and rerun"] if action in ("BLOCK","DELAY") else ["monitor deployment"])
