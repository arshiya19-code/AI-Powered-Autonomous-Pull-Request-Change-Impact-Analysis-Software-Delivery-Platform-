"""Read-only, secret-safe demo readiness report.

Run from the project root with: python -m scripts.demo_readiness
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil, sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configured(*names: str) -> bool:
    return all(bool(os.getenv(name)) for name in names)


def main() -> int:
    os.chdir(ROOT)
    report: dict[str, object] = {}
    try:
        from sqlalchemy import inspect, text
        from backend.app.database import engine
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        tables = set(inspect(engine).get_table_names())
        report["database"] = {"ready": {"users", "analyses", "repository_connections", "pull_requests"}.issubset(tables), "engine": engine.dialect.name}
        if "alembic_version" in tables:
            with engine.connect() as connection:
                current = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
        else:
            current = "schema-initialized-with-create_all"
        heads = sorted(path.stem for path in (ROOT / "backend" / "migrations" / "versions").glob("*.py") if path.stem[:1].isdigit())
        latest = heads[-1] if heads else None
        report["migration"] = {"ready": bool(latest and (current == latest or current == latest.split("_", 1)[0])), "current": current, "latest_file": latest}
    except Exception as exc:
        report["database"] = {"ready": False, "reason": type(exc).__name__}
        report["migration"] = {"current": "unavailable", "latest_file": None}

    from backend.app.services.risk import RiskPredictionService
    risk = RiskPredictionService()
    report["github"] = {"ready": configured("GITHUB_APP_ID", "GITHUB_APP_SLUG") and bool(os.getenv("GITHUB_APP_PRIVATE_KEY") or os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")), "webhook_secret": "configured" if os.getenv("GITHUB_WEBHOOK_SECRET") else "not_configured"}
    report["runtime_model"] = {"ready": risk.status == "available", "artifact": str(risk.artifact_path), "version": risk._metadata.get("model_version")}
    report["shap"] = {"ready": risk.status == "available" and importlib.util.find_spec("shap") is not None}
    report["repair_provider"] = {"ready": configured("LLM_PROVIDER", "LLM_BASE_URL"), "provider": os.getenv("LLM_PROVIDER") or "not_configured", "model": os.getenv("LLM_MODEL") or "qwen2.5-coder"}
    report["validation_runner"] = {"ready": Path(sys.executable).is_file(), "python": str(Path(sys.executable)), "git": "available" if shutil.which("git") else "unavailable"}
    report["monitoring_provider"] = {"ready": bool(os.getenv("PROMETHEUS_BASE_URL")), "status": "configured" if os.getenv("PROMETHEUS_BASE_URL") else "not_configured"}
    build_present = (ROOT / "frontend" / "dist" / "index.html").is_file()
    report["frontend"] = {"ready": build_present, "expected_origin": os.getenv("FRONTEND_URL") or os.getenv("APP_URL") or "http://localhost:5173", "build_present": build_present}
    try:
        from backend.app.main import health
        report["backend_health"] = {"ready": health().get("status") == "ok", "mode": "in_process"}
    except Exception as exc:
        report["backend_health"] = {"ready": False, "reason": type(exc).__name__}

    for name, value in report.items():
        state = "READY" if isinstance(value, dict) and value.get("ready") else "NOT READY / OPTIONAL"
        print(f"{name:22} {state}")
    print("\n" + json.dumps(report, indent=2, sort_keys=True))
    required = (report.get("database", {}).get("ready") and report.get("runtime_model", {}).get("ready")
                and report.get("shap", {}).get("ready") and report.get("validation_runner", {}).get("ready")
                and report.get("backend_health", {}).get("ready"))
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
