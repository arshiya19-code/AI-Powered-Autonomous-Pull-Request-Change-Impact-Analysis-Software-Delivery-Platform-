# Product correction pass — implementation and verification

Implemented on 11 September 2026. This is an implementation report, not a claim that visual acceptance is complete.

1. **Visual changes:** Replaced the previous dark stylesheet and shell with translucent cream surfaces, visible blurred lavender/blue/mint/peach fields, rounded cards, quieter typography, and distinct functional areas. Empty states contain an icon, explanation, and contextual analysis action.
2. **Palette:** Lavender `#B7A8FF`, blue `#84B6E8`, cyan `#76C7C8`, sage/mint, coral `#E49A9F`, amber `#E7C477`, cream glass. The graph retains a deep blue/teal atmosphere. Evening atmosphere is an account preference.
3. **Collapsed sidebar:** 68px desktop rail; the product name, navigation text, and account details are not rendered when collapsed. Icons have accessible names and native hover tooltips. Smaller viewports start collapsed.
4. **Expanded sidebar:** 222px, full names and account details. On smaller screens it expands as an overlay. Smooth width transitions and a bottom account menu.
5. **Overview:** Large hero beside a compact latest-analysis/system-health stack; a wider dependency preview beside a peach risk panel; compact delivery sequence below. Statistics use owned database records and actual service readiness.
6. **Graph renderer/layout:** Replaced the hardcoded six-node preview with a custom SVG presentation of the existing backend graph. Depth rings, file-ordered clustering, spherical gradients, halo rings, curved links and a large atmospheric canvas. No graph construction, resolver, language adapter, or impact algorithm was replaced.
7. **Graph interactions:** Pan, zoom, fit, reset, symbol search, selection, hover/connected-edge emphasis, unrelated-node fading, changed/critical filters, maximum-depth and relationship filters. Inspector includes symbols, file/module, relationships, depth, counts and changed origins. Trace animation follows actual backend depths. Missing critical metadata is reported as unavailable, not inferred.
8. **Authentication:** Registration, login, logout, current user, session rotation, profile update, password change, forgot/reset password. Argon2id hashes and opaque random sessions; only token hashes are persisted. HttpOnly, SameSite=Lax cookies; Secure is configurable for HTTPS. Origin checks on mutations, login/recovery throttling, validation errors without submitted credential values.
9. **Database:** SQLAlchemy models for User (profile/preferences embedded), Session, PasswordReset, Analysis (workflow result/events embedded), RepositoryConnection and AuditEvent. DATABASE_URL supports PostgreSQL via psycopg; SQLite fallback for development. No redundant profile/workflow tables merely for complexity.
10. **Profile:** Real display name/avatar/email/role/join date, appearance/start-page/notification preferences, password change and recent activity. Email is read-only; ownership verification for email changes is not implemented.
11. **Admin:** Backend-authorized user list with roles, account status, joined/last-login dates and analysis counts; activate/deactivate and role changes; platform audit activity. Self-demotion/deactivation is rejected. No administrator password is seeded. Use the trusted console bootstrap command below after registering your real account.
12. **Isolation:** Product list/detail/SSE/approval routes enforce ownership. Legacy server-filesystem and shared-state APIs require an administrator. Browser submissions cannot specify arbitrary server paths; source paths are checked and files are materialized in an isolated temporary workspace. Existing analyses are not silently assigned to a new user.
13. **Real-time architecture:** Background jobs persist ordered events to SQL storage. Authenticated SSE supports Last-Event-ID replay and rechecks session validity while streaming. The frontend subscribes and retrieves the durable completed result. No simulated progress percentages.
14. **Events:** ANALYSIS_STARTED, SOURCE_PARSED, GRAPH_BUILT, IMPACT_ANALYZED (includes the real graph before risk finishes), RISK_STARTED, RISK_COMPLETE/UNAVAILABLE, SHAP_COMPLETE/UNAVAILABLE, VALIDATION_STARTED/COMPLETE/UNAVAILABLE, REPAIR_STARTED/COMPLETE/UNAVAILABLE, TEST_STARTED/COMPLETE/UNAVAILABLE, AWAITING_APPROVAL or WORKFLOW_BLOCKED, ANALYSIS_COMPLETE/WORKFLOW_FAILED, and explicit simulation approval/rejection updates. These wrap the existing services. Unavailable stages remain unavailable.
15. **Files changed:** `frontend/src/main.tsx`, `frontend/src/styles.css`, `frontend/src/Account.tsx`, `frontend/src/Graph.tsx`, `frontend/tsconfig.json`; `backend/app/database.py`, `backend/app/auth.py`, `backend/app/product.py`, `backend/app/main.py`, `backend/app/schemas/models.py`, `backend/app/services/analysis.py`, `backend/app/services/workflow.py`; `requirements.txt`, `.env.example`, `scripts/create_admin.py`; `tests/conftest.py`, `tests/test_product_accounts.py`, `tests/test_backend_foundation.py`, `tests/test_monitoring_feedback.py`; this report, account setup notes, and migration files below. Build output was regenerated. This directory has no Git metadata, so a Git diff/status report is unavailable.
16. **Migration files:** `alembic.ini`, `backend/migrations/env.py`, `backend/migrations/versions/0001_accounts.py`. An isolated SQLite migration reached `0001_accounts (head)`. No live PostgreSQL instance was available for a PostgreSQL migration test.
17. **Frontend verification:** Final TypeScript check and Vite production build passed (1,849 modules; 240.13 kB JavaScript, 25.86 kB CSS before gzip). Visual browser verification is blocked: Codex's embedded browser repeatedly timed out attaching its webview, including the alternate tab API. Responsive behavior was implemented but cannot honestly be marked visually verified.
18. **Backend verification:** 199 focused tests passed across account, backend foundation, monitoring/feedback, autonomous delivery, graph, and impact suites. Existing absent-model tests now explicitly use an unavailable artifact fixture, since a real local model artifact exists. Unrelated dataset/training suites were not rerun.
19. **Security verification:** Final security rerun: 7 passed. Registration/hash handling, sessions, logout/password changes, CSRF origin checks, role authorization, account deactivation, user isolation for detail/SSE, source-path traversal, password reset one-use/session revocation, throttling, and approval refusal without passed validation are covered. Existing workflow approval logic remains in control.
20. **Still unavailable / limitations:** SMTP must be configured for reset mail; email verification and GitHub OAuth are not implemented. Browser source submissions do not execute arbitrary test code, so validation can remain not-run; production deployment remains the existing simulation-only capability. The job executor and rate limiter are single-process foundations, not a distributed worker/rate-limit service. In-flight jobs do not resume after process loss. Very large graph performance is not benchmarked. The supplied graph reference recording was reviewed at four points; earlier reference images/current-dashboard video were not available in the attached files. Visual acceptance remains open due to the browser failure.

## Account setup

From the repository root, install requirements and run the migration before starting the API:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Register your account through the UI. To grant that existing account administrator access from the trusted server console:

```powershell
.venv/Scripts/python.exe -m scripts.create_admin your-real-email@example.com
```

For production, supply `DATABASE_URL=postgresql+psycopg://...`, exact HTTPS `APP_ORIGINS`, `APP_URL`, and `COOKIE_SECURE=true`. Environment variables must be loaded into the process (the application does not automatically load `.env.example`). Keep SMTP and database credentials server-side. Use one application worker with this in-process executor. A durable distributed job queue and shared rate limiter are a later operational step.

The local review server uses ports 5174/8011 and a separate `.tmp/visual-review.db`; it does not seed users or substitute demo analysis data.

