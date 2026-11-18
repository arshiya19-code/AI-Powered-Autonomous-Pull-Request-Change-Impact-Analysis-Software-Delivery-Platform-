"""Authenticated test client for legacy administrator API contracts."""
import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch):
    """Unit tests opt into mocked providers, never the workstation's live keys."""
    for name in list(os.environ):
        if name.startswith('LLM_'):
            monkeypatch.delenv(name, raising=False)

@pytest.fixture(autouse=True)
def configured_test_origin(monkeypatch):
    """Allow the explicit legacy test origin without changing production defaults."""
    from backend.app import auth
    auth.origins.add('http://localhost:5173')

@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend.app.main import app, service, workflow
    from backend.app.services.risk import RiskPredictionService
    from backend.app.services.shap import ShapService
    unavailable = RiskPredictionService(str(tmp_path / "missing-model.cbm"))
    monkeypatch.setattr(service, "risk", unavailable)
    monkeypatch.setattr(service, "shap", ShapService())
    monkeypatch.setattr(workflow.analysis, "risk", unavailable)
    monkeypatch.setattr(workflow.analysis, "shap", ShapService())
    from backend.app import auth, database
    engine=create_engine('sqlite:///'+str(tmp_path/'api-accounts.db'),connect_args={'check_same_thread':False})
    database.Base.metadata.create_all(engine)
    factory=sessionmaker(engine,expire_on_commit=False)
    monkeypatch.setattr(auth,'DB',factory)
    auth.attempts.clear()
    with TestClient(app,headers={'origin':'http://localhost:5173'}) as client:
        result=client.post('/api/auth/register',json={'email':'api-contract@example.test','display_name':'API test','password':'isolated test passphrase'})
        assert result.status_code==201
        with factory() as db:
            db.get(database.User,result.json()['id']).role='ADMIN';db.commit()
        yield client
    engine.dispose()
