import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Response
from backend.app.schemas.models import SystemStatus, FeedbackRecord
from backend.app.services.analysis import AnalysisService, PRAnalysisRequest, FeatureAnalysisRequest
from backend.app.services.workflow import DeliveryWorkflowService
from backend.app.schemas.models import WorkflowResponse
from backend.app.services.feedback import FeedbackService
from backend.app.services.history import HistoryService
from backend.app.services import metrics
from backend.app.services.ci_tools import tool_readiness

app = FastAPI(title="AI Change Impact Platform Backend", version="1.0.0")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app import auth, product, admin, database, github
database.initialize()
app.include_router(auth.router)
app.include_router(product.router)
app.include_router(admin.router)
app.include_router(github.router)

@app.on_event('startup')
def resume_interrupted_github_jobs():
    product.resume_incomplete_github_analyses()

@app.middleware('http')
async def account_boundary(request, call_next):
    path = request.url.path
    if request.method == 'OPTIONS': return await call_next(request)
    if request.method not in ('GET','HEAD'):
        origin = request.headers.get('origin')
        if (origin and origin not in auth.origins) or request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'detail':'Origin not allowed'}, status_code=403)
        if request.cookies.get(auth.COOKIE) and not origin:
            return JSONResponse({'detail':'Browser mutations require an Origin header'}, status_code=403)
    public = path in ('/health','/api/health','/docs','/openapi.json','/api/github/webhook') or path.startswith('/api/auth/')
    if not public:
        try:
            user = auth.current_user(request)
            # Legacy operations operate on server repositories and shared service state.
            # Keep them administrator-only; browser analyses use owner-scoped product routes.
            safe = path.startswith('/api/product/') or path.startswith('/api/admin/') or path.startswith('/api/github/') or path in ('/api/v1/system/status','/api/model/status')
            if not safe and user.role != 'ADMIN': raise HTTPException(403, 'Administrator access required for legacy operations')
        except HTTPException as exc: return JSONResponse({'detail':exc.detail}, status_code=exc.status_code)
    response = await call_next(request)
    if path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

app.add_middleware(CORSMiddleware, allow_origins=list(auth.origins), allow_credentials=True, allow_methods=['GET','POST','PATCH','DELETE','OPTIONS'], allow_headers=['Content-Type','Last-Event-ID'])
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def safe_validation_error(request, exc):
    # Pydantic includes submitted values by default; never echo credentials.
    return JSONResponse({'detail': [{'loc': error['loc'], 'msg': error['msg'], 'type': error['type']} for error in exc.errors()]}, status_code=422)

service = AnalysisService()
workflow = DeliveryWorkflowService(service)
feedback = workflow.feedback
history = HistoryService(feedback)
monitoring = workflow.monitoring
pipeline_runs = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/health")
def api_health(): return {**health(), "service": "change-impact-backend", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/model/status")
def model_status():
    return {"status": workflow.analysis.risk.status.upper(), "readiness": "READY" if workflow.analysis.risk.status == "available" else "MODEL_NOT_READY", "artifact": str(workflow.analysis.risk.artifact_path)}

@app.get("/metrics")
def prometheus_metrics():
    return Response(content=metrics.render(), media_type=metrics.METRICS_CONTENT_TYPE)

@app.get("/api/v1/monitoring/status")
def monitoring_status(): return monitoring.current()

@app.get("/api/v1/monitoring/metrics-summary")
def monitoring_metrics_summary(): return monitoring.metrics_summary()

@app.post("/api/v1/feedback", response_model=FeedbackRecord)
def create_feedback(record: FeedbackRecord):
    try: return feedback.record(record)
    except Exception as exc: raise HTTPException(status_code=400, detail="Feedback persistence failed") from exc

@app.post("/api/feedback", response_model=FeedbackRecord)
def api_feedback(record: FeedbackRecord): return create_feedback(record)

@app.get("/api/v1/history")
def history_list(limit: int = 50, offset: int = 0): return {"items": history.list(limit, offset), "limit": max(1, min(limit, 100)), "offset": max(0, offset)}

@app.get("/api/v1/history/{workflow_id}")
def history_detail(workflow_id: str):
    item = history.get(workflow_id)
    if not item: raise HTTPException(status_code=404, detail="Workflow not found")
    return item

@app.get("/api/v1/system/status", response_model=SystemStatus)
def system_status():
    risk = workflow.analysis.risk.status
    provider_readiness = workflow.repair.readiness()
    repair = 'available' if provider_readiness['status'] == 'READY' else 'unavailable'
    shap = 'available' if risk == 'available' else 'unavailable'
    monitor = monitoring.current().status
    return SystemStatus(status='ok' if risk == repair == 'available' else 'degraded', model_status=risk, shap_status=shap, services={'analysis':'available', 'risk':risk, 'shap':shap, 'validation':'available', 'repair':repair, 'deployment':'simulation_only', 'monitoring':'available', 'workflow_monitoring':'active', 'external_telemetry':monitor, 'feedback':'available'}, provider_readiness=provider_readiness, tool_readiness=tool_readiness())

@app.post("/api/analyze/diff")
def analyze_diff(request: PRAnalysisRequest): return analyze_pr(request)

@app.post("/api/analyze/repository")
def analyze_repository(request: PRAnalysisRequest): return analyze_pr(request)

@app.post("/api/v1/analyze/pr")
def analyze_pr(request: PRAnalysisRequest):
    try: return service.analyze(request)
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/v1/analyze/features")
def analyze_features(request: FeatureAnalysisRequest):
    try:
        result = service.analyze(request)
        return {"features": result.features, "feature_count": len(result.features), "analysis_complete": result.analysis_complete, "warnings": result.warnings, "errors": result.errors}
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/v1/workflow/analyze", response_model=WorkflowResponse)
def workflow_analyze(request: PRAnalysisRequest):
    try: return workflow.run(request)
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/v1/workflow/validate", response_model=WorkflowResponse)
def workflow_validate(request: PRAnalysisRequest):
    try: return workflow.run(request, validate=True)
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/v1/workflow/repair", response_model=WorkflowResponse)
def workflow_repair(request: PRAnalysisRequest):
    try: return workflow.run(request, mode="proposal", validate=False)
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/v1/workflow/simulate-deployment", response_model=WorkflowResponse)
def workflow_deployment(request: PRAnalysisRequest):
    try: return workflow.run(request, validate=True)
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/risk/predict")
def risk_predict(payload: dict):
    if os.getenv("DEMO_MODE", "false").lower() == "true" and payload.get("mode") == "demo":
        return {"mode":"demo", "synthetic":True, "readiness":"DEMO", "prediction":{"risk_probability":0.42,"risk_class":0}}
    features = payload.get("features") or payload
    missing = [x for x in __import__("dataset.config", fromlist=["ALL_FEATURES"]).ALL_FEATURES if x not in features]
    if missing: raise HTTPException(status_code=422, detail={"code":"FEATURE_CONTRACT_ERROR","missing":missing})
    return {"mode":"production", **workflow.analysis.risk.predict(features).model_dump()}

@app.post("/api/risk/explain")
def risk_explain(payload: dict):
    features = payload.get("features") or {}
    risk = workflow.analysis.risk.predict(features)
    return {"mode":"production", **workflow.analysis.shap.explain(features, risk).model_dump()}

@app.post("/api/repair")
def repair(payload: dict): return workflow.repair.propose(payload).model_dump()

@app.post("/api/tests/generate")
def generate_tests(payload: dict): return workflow.tests.generate(root=payload.get("root", "."), changed_symbols=payload.get("changed_symbols"), impacted_symbols=payload.get("impacted_symbols"), language=payload.get("language")).model_dump()

@app.post("/api/validate")
def validate(payload: dict):
    try:
        return workflow.validation.validate(payload.get("root", "."), payload.get("requested")).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code":"INVALID_REPOSITORY_ROOT", "message":str(exc)}) from exc

@app.post("/api/deployment/{workflow_id}/approve", response_model=WorkflowResponse)
def approve_deployment(workflow_id: str):
    try: return workflow.approve(workflow_id)
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.post("/api/deployment/{workflow_id}/reject", response_model=WorkflowResponse)
def reject_deployment(workflow_id: str):
    try: return workflow.reject(workflow_id)
    except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.post("/api/deployment/decision")
def deployment_decision(payload: dict):
    from backend.app.schemas.models import RiskPrediction, ValidationSummary, ImpactSummary
    args = dict(payload)
    args["risk"] = RiskPrediction.model_validate(args.get("risk", {}))
    args["validation"] = ValidationSummary.model_validate(args["validation"]) if args.get("validation") else None
    args.setdefault("impact_score", (args.get("impact") or {}).get("impact_score"))
    args.setdefault("blast_radius", (args.get("impact") or {}).get("blast_radius", 0))
    args.setdefault("critical_nodes", (args.get("impact") or {}).get("critical_nodes", 0))
    args.setdefault("analysis_complete", True)
    args.pop("impact", None)
    decision = workflow.analysis.decision.decide(**args)
    return {"decision": decision.model_dump(), "deployment": workflow.deployment.simulate(decision, validation=args.get("validation"), risk=args["risk"], impact=payload.get("impact")).model_dump()}

@app.post("/api/pipeline/run")
def pipeline_run(request: PRAnalysisRequest):
    result = workflow.run(request, validate=bool(request.repository_path))
    pipeline_runs[result.workflow_id] = {"status":"COMPLETED", "created_at":datetime.now(timezone.utc).isoformat(), "result":result.model_dump(mode="json")}
    return {"run_id":result.workflow_id, "status":"COMPLETED", "stages":["ANALYZING","IMPACT_ANALYSIS","RISK_PREDICTION","EXPLAINING","DECISION","DEPLOYMENT_DECISION","COMPLETED"], "result":result}

@app.get("/api/pipeline/{run_id}")
def pipeline_get(run_id: str):
    if run_id not in pipeline_runs: raise HTTPException(status_code=404, detail={"code":"PIPELINE_NOT_FOUND"})
    return pipeline_runs[run_id]

@app.get("/api/history")
def api_history(limit: int = 50, offset: int = 0): return history_list(limit, offset)

@app.get("/api/monitoring/summary")
def monitoring_summary(): return monitoring.metrics_summary()
