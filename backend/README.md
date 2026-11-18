# Backend foundation

## Start

From the repository root: `uvicorn backend.app.main:app --reload`.

The API is safe to start while Phase 7F is running. Startup never reads or
writes the producer's caches and never trains a model.

Launch from the repository root with `uvicorn backend.app.main:app --reload`.

Expected configuration is optional for the foundation: `RISK_MODEL_PATH` (future CatBoost artifact), `SHAP_ENABLED` (future explainer switch), `LLM_PROVIDER` / `LLM_MODEL` (future Llama or Qwen repair provider), and repository/GitHub credentials when a remote acquisition adapter is added. With none configured, the API reports explicit unavailable states.

The backend delegates diff parsing, language adapters, common IR, resolver-compatible analyzer components, custom graph/impact components, and the canonical contract in `dataset.config.ALL_FEATURES`. The default graph is intentionally empty when no repository-wide graph acquisition is supplied; no impact facts are fabricated.

`RISK_MODEL_PATH` can later point the risk service at a CatBoost artifact. SHAP will plug into `backend/app/services/shap.py`; Llama/Qwen into `repair.py`. Repository acquisition and validation commands are isolated behind their service boundaries. No automatic deployment or retraining occurs.

Observability is available at `/metrics`, with dashboard-ready JSON APIs at `/api/v1/monitoring/status`, `/api/v1/monitoring/metrics-summary`, `/api/v1/history`, and `/api/v1/history/{workflow_id}`. Prometheus may scrape `http://backend-host:8000/metrics` using a standard `scrape_configs` job. Suggested Grafana panels are analyses/workflows over time, validation and repair outcomes, decision recommendations, duration, monitoring health, and rollback signals.

The frontend may use `/api/health`, `/api/model/status`, `/api/analyze/diff`,
`/api/analyze/repository`, `/api/risk/predict`, `/api/risk/explain`,
`/api/repair`, `/api/tests/generate`, `/api/validate`,
`/api/deployment/decision`, `/api/pipeline/run`, `/api/pipeline/{run_id}`,
`/api/history`, `/api/monitoring/summary`, and `/api/feedback`.

Production risk responses are unavailable until a real CatBoost artifact exists
and are marked `MODEL_NOT_READY`. Set `DEMO_MODE=true` and send
`mode=demo` only for synthetic UI development; demo payloads are marked
`mode=demo` and `synthetic=true` and are never evaluation data.

After Phase 7F completes, validate the parquet with
`analyzer.ml.production.load_final_dataset`. Call
`train_final_catboost` explicitly only after the provenance and class-balance
diagnostics pass. Artifacts belong under
`artifacts/models/deployment_risk/`; the API does not train automatically.

Monitoring thresholds are configured by `MONITORING_ERROR_RATE_THRESHOLD` and `MONITORING_LATENCY_THRESHOLD`; `PROMETHEUS_ENABLED`, `PROMETHEUS_BASE_URL`, `MONITORING_TIMEOUT`, and `FEEDBACK_STORE_PATH` are supported configuration points. Without a runtime provider, external values are explicitly unavailable while internal counters remain scrapeable. Feedback is append-safe SQLite lifecycle evidence, separate from the supervised dataset; collecting feedback does not retrain models automatically.

## Browser accounts

The product layer now requires a real account. See [product correction report](../docs/product-correction-report.md) for authentication, database migration, administrator bootstrap, event streaming, and limitations. Browser submissions use `/api/product/analyses` and owner-scoped history. Legacy server-repository and shared-history APIs now require an authenticated administrator; they are no longer public endpoints.
## GitHub App integration

Configure `GITHUB_APP_ID`, `GITHUB_APP_SLUG`, `GITHUB_APP_PRIVATE_KEY` (or `GITHUB_APP_PRIVATE_KEY_PATH`), and `GITHUB_WEBHOOK_SECRET`. Set the GitHub App setup URL to `/api/github/callback` and its webhook URL to `/api/github/webhook`. The private key and short-lived installation tokens stay server-side; only installation and repository metadata are persisted.

The GitHub App requires read access to repository contents, metadata, and pull requests, plus pull-request webhook events. Repository selection (all or selected repositories) is handled by GitHub's installation screen.
