from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.models import RiskPrediction, ValidationSummary
from backend.app.services.decision import DecisionEngine
from backend.app.services.repair import RepairService
from backend.app.services.monitoring import MonitoringService



def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}

def test_system_status_is_explicitly_degraded_without_model(client):
    body = client.get('/api/v1/system/status').json()
    assert body['model_status'] == 'unavailable'
    assert body['shap_status'] == 'unavailable'

def test_feature_schema_and_pr_orchestration(client):
    diff = 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n def f():\n+    return 1\n'
    response = client.post('/api/v1/analyze/features', json={'repository': 'demo', 'diff': diff, 'sources': {'a.py': 'def f():\n    return 1\n'}})
    assert response.status_code == 200
    assert response.json()['feature_count'] == 42
    assert response.json()['features']['files_changed'] == 1

def test_absent_model_and_shap_are_not_fabricated(client):
    response = client.post('/api/v1/analyze/pr', json={'repository': 'demo', 'diff': ''})
    assert response.status_code == 200
    body = response.json()
    assert body['risk_prediction']['model_status'] == 'unavailable'
    assert body['risk_prediction']['risk_probability'] is None
    assert body['shap']['shap_status'] == 'unavailable'

def test_decision_engine_blocks_incomplete_analysis_without_claiming_deploy():
    result = DecisionEngine().decide(risk=RiskPrediction(), impact_score=0, blast_radius=0, critical_nodes=0, analysis_complete=False)
    assert result.decision != 'DEPLOY'
    assert 'ANALYSIS_INCOMPLETE' in result.reason_codes

def test_validation_and_unavailable_service_boundaries(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert ValidationSummary().status == 'not_run'
    assert RepairService().propose().status == 'unavailable'
    assert MonitoringService().current().status == 'unavailable'

