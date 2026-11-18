from __future__ import annotations

import os
from datetime import datetime
from typing import Mapping, Optional, Protocol
from backend.app.schemas.models import DeploymentHealthEvidence, MonitoringSummary
from . import metrics

class MonitoringProvider(Protocol):
    def snapshot(self, deployment_id: Optional[str] = None) -> MonitoringSummary: ...

class MonitoringService:
    """Provider-independent monitoring boundary; no source means UNKNOWN, not zero."""
    def __init__(self, provider: MonitoringProvider | None = None, *, error_rate_threshold: float | None = None, latency_threshold: float | None = None):
        self.provider = provider
        self.error_rate_threshold = error_rate_threshold if error_rate_threshold is not None else float(os.getenv("MONITORING_ERROR_RATE_THRESHOLD", "0.05"))
        self.latency_threshold = latency_threshold if latency_threshold is not None else float(os.getenv("MONITORING_LATENCY_THRESHOLD", "1.5"))

    def current(self) -> MonitoringSummary:
        if self.provider:
            return self.provider.snapshot()
        return MonitoringSummary(status="unavailable", health_state="UNKNOWN", provider="none", source="external", message="External runtime monitoring is unavailable; no provider is configured.", warnings=["No Prometheus/runtime monitoring source is configured."])

    def evaluate(self, evidence: DeploymentHealthEvidence | Mapping[str, object], *, deployment_id: str | None = None) -> MonitoringSummary:
        e = evidence if isinstance(evidence, DeploymentHealthEvidence) else DeploymentHealthEvidence.model_validate(evidence)
        reasons: list[str] = []
        if e.service_available is False: reasons.append("SERVICE_UNAVAILABLE")
        if e.health_checks_failed > 0: reasons.append("HEALTH_CHECK_FAILED")
        if e.severe_regression: reasons.append("POST_DEPLOYMENT_REGRESSION")
        if e.error_rate is not None and e.baseline_error_rate is not None and e.error_rate - e.baseline_error_rate >= self.error_rate_threshold: reasons.append("ERROR_RATE_SPIKE")
        if e.latency is not None and e.baseline_latency is not None and e.latency >= e.baseline_latency * self.latency_threshold: reasons.append("LATENCY_REGRESSION")
        severe = any(x in reasons for x in ("SERVICE_UNAVAILABLE", "POST_DEPLOYMENT_REGRESSION", "ERROR_RATE_SPIKE"))
        state = "UNHEALTHY" if severe else ("DEGRADED" if reasons else "HEALTHY")
        metrics.monitoring_health.labels(state).set(1)
        return MonitoringSummary(status="regression" if reasons else "available", health_state=state, timestamp=datetime.utcnow(), deployment_id=deployment_id or e.deployment_id, regression_indicators=reasons, source="evidence", provider="deployment-health", request_metrics={"error_rate": e.error_rate}, latency_indicators={"latency": e.latency}, warnings=[])

    def metrics_summary(self) -> dict:
        return {"external_runtime_monitoring": "configured" if self.provider or os.getenv("PROMETHEUS_BASE_URL") else "unavailable", "health_state": "UNKNOWN" if not self.provider else "available", "source": "internal prometheus registry"}
