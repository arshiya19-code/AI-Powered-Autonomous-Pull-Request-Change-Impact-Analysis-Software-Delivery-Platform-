"""One user-owned Track B workflow covering the real internal analysis architecture."""
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import admin, auth, database, github, product
from backend.app.main import app
from backend.app.services.feedback import FeedbackService


def test_real_pr_to_approved_simulation_monitoring_history_and_access(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "platform.db"), connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    for module in (auth, product):
        monkeypatch.setattr(module, "DB", factory)
    monkeypatch.setattr(admin.auth, "DB", factory)
    monkeypatch.setenv("FEEDBACK_STORE_PATH", str(tmp_path / "feedback.db"))
    auth.attempts.clear()
    headers = {"origin": "http://localhost:5173"}
    owner, other, administrator = (TestClient(app, headers=headers) for _ in range(3))
    for client, name in ((owner, "owner"), (other, "other"), (administrator, "admin")):
        response = client.post("/api/auth/register", json={"email": f"{name}@example.test", "display_name": name, "password": "a long private password"})
        assert response.status_code == 201
    owner_id = owner.get("/api/auth/me").json()["id"]
    admin_id = administrator.get("/api/auth/me").json()["id"]
    with factory() as db:
        db.get(database.User, admin_id).role = "ADMIN"
        connection = database.RepositoryConnection(user_id=owner_id, repository="team/demo", provider="github", installation_id="9", repository_id="90", full_name="team/demo", connection_status="CONNECTED")
        db.add(connection); db.flush()
        pr = database.PullRequest(repository_connection_id=connection.id, provider_pr_id="420", pr_number=42, title="Safe focused change", state="open", author="dev", base_sha="a" * 40, head_sha="b" * 40)
        db.add(pr); db.commit(); pr_id = pr.id

    diff = "diff --git a/sample.py b/sample.py\n--- a/sample.py\n+++ b/sample.py\n@@ -1,2 +1,2 @@\n def changed():\n-    return 1\n+    return 2\n"
    source = "def changed():\n    return 2\n\ndef caller():\n    return changed()\n"
    def materialize(_installation, _repo, _base, _head, workspace):
        root = Path(workspace)
        (root / "sample.py").write_text(source)
        (root / "test_sample.py").write_text("from sample import changed\n\ndef test_changed(): assert changed() == 2\n")
        (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths=['.']\n")
        return diff, {"sample.py": source, "test_sample.py": (root / "test_sample.py").read_text()}, []
    monkeypatch.setattr(github.service, "materialize_pull_request", materialize)
    monkeypatch.setattr(github.service, "get_pull_request", lambda *_args: {
        "id": 420, "title": "Safe focused change", "body": "", "number": 42,
        "base": {"sha": "a" * 40}, "head": {"sha": "b" * 40},
        "state": "open", "user": {"login": "dev"},
    })
    class Immediate:
        def submit(self, function, *args):
            function(*args)
    monkeypatch.setattr(product, "executor", Immediate())

    created = owner.post(f"/api/github/pull-requests/{pr_id}/analyze")
    assert created.status_code == 202
    analysis_id = created.json()["id"]
    detail = owner.get(f"/api/product/analyses/{analysis_id}").json()
    assert detail["status"] == "COMPLETE"
    workflow = detail["workflow"]
    analysis = workflow["analysis"]
    assert analysis["graph"]["nodes"] and analysis["change_details"]["changed_files"] == ["sample.py"]
    assert analysis["impact_details"]["directly_changed_nodes"]
    assert analysis["risk_prediction"]["model_status"] == "available"
    assert analysis["risk_prediction"]["feature_names"] == ["files_changed", "lines_added", "lines_deleted", "total_lines_changed"]
    assert analysis["shap"]["shap_status"] == "available"
    assert workflow["decision"]["decision"] in ("DEPLOY", "DELAY")
    assert workflow["validation"]["status"] == "passed"
    assert workflow["repair"]["status"] == "unavailable"
    assert workflow["approval_state"] == "READY_FOR_APPROVAL"
    assert workflow["deployment"]["simulation_status"] == "BLOCKED"

    selected = owner.post(f"/api/product/analyses/{analysis_id}/selection", json={"variant": "ORIGINAL", "reason": "Reviewed original evidence"})
    assert selected.status_code == 200
    approved = owner.post(f"/api/product/analyses/{analysis_id}/approve", json={"note": "Reviewed for the demo"})
    assert approved.status_code == 200
    final = owner.get(f"/api/product/analyses/{analysis_id}").json()
    assert final["workflow"]["approval_metadata"]["actor_id"] == owner_id
    assert final["workflow"]["deployment"]["simulation_status"] == "SIMULATED"
    assert final["workflow"]["deployment"]["lifecycle_history"][-2:] == ["DEPLOYMENT_STARTED", "DEPLOYMENT_COMPLETED"]
    assert final["workflow"]["monitoring"]["status"] == "unavailable"
    assert {event["type"] for event in final["events"]} >= {"USER_APPROVED", "DEPLOYMENT_STARTED", "DEPLOYMENT_COMPLETED"}
    history = owner.get("/api/product/analyses").json()["items"][0]
    assert history["decision"] in ("DEPLOY", "DELAY") and history["approval"] == "DEPLOYMENT_COMPLETED"
    assert other.get(f"/api/product/analyses/{analysis_id}").status_code == 404
    admin_detail = administrator.get(f"/api/admin/analyses/{analysis_id}")
    assert admin_detail.status_code == 200 and admin_detail.json()["result"]["workflow_id"] == workflow["workflow_id"]
    feedback = FeedbackService(tmp_path / "feedback.db").get(workflow["workflow_id"])
    assert feedback and feedback.predicted_risk is not None and feedback.actual_deployment_outcome == "SIMULATED_DEPLOYED"
    engine.dispose()
