from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from backend.app.schemas.models import DecisionResult, RiskPrediction, ValidationSummary, MonitoringSummary

@dataclass
class DecisionThresholds:
    high_risk: float = 0.8
    large_blast_radius: int = 20

class DecisionEngine:
    def __init__(self, thresholds: DecisionThresholds | None = None):
        self.thresholds = thresholds or DecisionThresholds()

    def decide(self, *, risk: RiskPrediction, impact_score: Optional[float], blast_radius: int,
               critical_nodes: int, analysis_complete: bool,
               validation: ValidationSummary | None = None,
               repair_state: str = "not_considered",
               security_failed: bool = False, post_deployment_regression: bool = False,
               monitoring: MonitoringSummary | None = None) -> DecisionResult:
        codes, reasons = [], []
        signals = {
            "analysis_complete": analysis_complete,
            "model_status": risk.model_status,
            "risk_probability": risk.risk_probability,
            "risk_threshold": risk.threshold,
            "impact_score": impact_score,
            "blast_radius": blast_radius,
            "critical_nodes": critical_nodes,
            "validation_status": validation.status if validation else "not_run",
            "validation_blocking_failures": list(validation.blocking_failures) if validation else [],
            "repair_state": repair_state,
            "monitoring_health": monitoring.health_state if monitoring else "UNKNOWN",
        }
        health_codes = list(monitoring.regression_indicators) if monitoring and monitoring.health_state == "UNHEALTHY" else []
        if post_deployment_regression: health_codes.append("POST_DEPLOYMENT_REGRESSION")
        if health_codes:
            ordered = [x for x in ("SERVICE_UNAVAILABLE", "HEALTH_CHECK_FAILED", "ERROR_RATE_SPIKE", "LATENCY_REGRESSION", "POST_DEPLOYMENT_REGRESSION") if x in health_codes]
            return DecisionResult(decision="ROLLBACK", status="decided", reason_codes=ordered or ["POST_DEPLOYMENT_REGRESSION"], human_readable_reasons=["Post-deployment health evidence indicates rollback eligibility."], confidence=0.95, blocking_factors=ordered or ["POST_DEPLOYMENT_REGRESSION"], recommended_actions=["halt rollout", "restore the last known healthy version"], input_signals=signals)
        if security_failed:
            codes.append("SECURITY_CHECK_FAILED"); reasons.append("Security validation failed.")
        if validation and (validation.status == "failed" or validation.blocking_failures):
            codes.append("VALIDATION_FAILED"); reasons.append("One or more validation checks failed.")
            for check in validation.blocking_failures:
                code = f"{check.upper()}_FAILED"
                if code not in codes: codes.append(code)
        if not analysis_complete:
            codes.append("ANALYSIS_INCOMPLETE"); reasons.append("Analysis did not complete.")
        if risk.model_status != "available":
            codes.append("MODEL_UNAVAILABLE"); reasons.append("No trained risk model is available.")
        if risk.risk_probability is not None and risk.risk_probability >= self.thresholds.high_risk:
            codes.append("HIGH_MODEL_RISK"); reasons.append("The model risk exceeds the configured threshold.")
        if blast_radius >= self.thresholds.large_blast_radius:
            codes.append("LARGE_BLAST_RADIUS"); reasons.append("The change affects a large dependency blast radius.")
        if critical_nodes:
            codes.append("CRITICAL_COMPONENT_AFFECTED"); reasons.append("Critical components are affected.")
        if codes:
            blocking = any(c in codes for c in ("SECURITY_CHECK_FAILED", "BUILD_FAILED", "UNIT_FAILED", "REGRESSION_FAILED", "INTEGRATION_FAILED", "VALIDATION_FAILED"))
            action = "BLOCK" if blocking else "DELAY"
            return DecisionResult(decision=action, status="decided", reason_codes=codes, human_readable_reasons=reasons, confidence=0.9 if blocking else 0.75, blocking_factors=codes if blocking else [], recommended_actions=["resolve blocking checks"] if blocking else ["obtain the missing evidence and re-evaluate"], input_signals=signals)
        return DecisionResult(decision="DEPLOY", status="decided", human_readable_reasons=["Risk and validation gates passed."], confidence=0.8, recommended_actions=["proceed with monitored deployment"], input_signals=signals)
