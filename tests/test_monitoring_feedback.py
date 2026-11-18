from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.models import DeploymentHealthEvidence, MonitoringSummary, RiskPrediction
from backend.app.services.decision import DecisionEngine
from backend.app.services.feedback import FeedbackService
from backend.app.services.monitoring import MonitoringService



def test_metrics_endpoint_and_bounded_metric_labels(client):
    body = client.get("/metrics")
    assert body.status_code == 200
    assert "change_impact_analyses_total" in body.text
    assert "repository" not in body.text

def test_monitoring_health_states_are_evidence_based():
    service = MonitoringService(error_rate_threshold=0.05, latency_threshold=1.5)
    assert service.current().health_state == "UNKNOWN"
    assert service.evaluate({}).health_state == "HEALTHY"
    assert service.evaluate({"latency": 2, "baseline_latency": 1}).health_state == "DEGRADED"
    result = service.evaluate(DeploymentHealthEvidence(service_available=False, deployment_id="sim-1"))
    assert result.health_state == "UNHEALTHY"
    assert result.regression_indicators == ["SERVICE_UNAVAILABLE"]

def test_rollback_requires_post_deployment_evidence():
    engine = DecisionEngine()
    risk = RiskPrediction(model_status="available", risk_probability=0.99)
    pre = engine.decide(risk=risk, impact_score=1, blast_radius=0, critical_nodes=0, analysis_complete=True)
    assert pre.decision != "ROLLBACK"
    monitor = MonitoringSummary(status="regression", health_state="UNHEALTHY", regression_indicators=["ERROR_RATE_SPIKE"])
    post = engine.decide(risk=risk, impact_score=1, blast_radius=0, critical_nodes=0, analysis_complete=True, monitoring=monitor)
    assert post.decision == "ROLLBACK"
    assert post.reason_codes == ["ERROR_RATE_SPIKE"]

def test_feedback_persistence_and_bounded_history(tmp_path):
    store = FeedbackService(tmp_path / "feedback.sqlite3")
    store.record(workflow_id="w-1", repository="demo", decision="DEPLOY", validation_outcome="passed")
    store.record(workflow_id="w-2", repository="demo", decision="BLOCK", validation_outcome="failed")
    assert store.get("w-1").decision == "DEPLOY"
    assert len(store.list(limit=1)) == 1

def test_history_and_monitoring_api_shapes(client):
    assert client.get("/api/v1/monitoring/status").json()["health_state"] == "UNKNOWN"
    assert client.get("/api/v1/history?limit=1").status_code == 200
