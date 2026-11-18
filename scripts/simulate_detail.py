"""
Direct inspection of what the /api/product/analyses/77a3f31b... endpoint returns.
Simulates what the frontend receives.
"""
import sqlite3
import json
import sys

# Simulate the detail() function from product.py
FASTAPI_ID = "77a3f31b-a3a2-42a9-8960-f51d660b466a"

conn = sqlite3.connect("backend/accounts.sqlite3")
cur = conn.cursor()
cur.execute("SELECT id, user_id, repository, pull_request, created_at, status, result, events, updated_at, repository_connection_id, pull_request_id FROM analyses WHERE id=?", (FASTAPI_ID,))
cols = ["id", "user_id", "repository", "pull_request", "created_at", "status", "result", "events", "updated_at", "repository_connection_id", "pull_request_id"]
row_data = cur.fetchone()
if not row_data:
    print("ANALYSIS NOT FOUND")
    sys.exit(1)

row = dict(zip(cols, row_data))
stored_raw = row.get("result")
stored = json.loads(stored_raw) if isinstance(stored_raw, str) else (stored_raw or {})

print("=== DETAIL ENDPOINT SIMULATION ===")
print(f"status: {row['status']}")
print(f"workflow_id in stored: {stored.get('workflow_id')}")
print(f"condition (status==COMPLETE and workflow_id): {row['status'] == 'COMPLETE' and bool(stored.get('workflow_id'))}")
print()

# Simulate the workflow return
workflow = stored if (row['status'] == 'COMPLETE' and stored.get('workflow_id')) else None
print(f"workflow is None: {workflow is None}")
if workflow:
    print(f"workflow.repair present: {bool(workflow.get('repair'))}")
    print(f"workflow.repair.candidate.selectable: {workflow.get('repair', {}).get('candidate', {}).get('selectable')}")
    print(f"workflow.repair.candidate.status: {workflow.get('repair', {}).get('candidate', {}).get('status')}")
    print(f"workflow.repair.patch present: {bool(workflow.get('repair', {}).get('patch'))}")
    print(f"workflow.test_generation present: {bool(workflow.get('test_generation'))}")
    print(f"workflow.candidate_comparison present: {bool(workflow.get('candidate_comparison'))}")

print()
print("=== FRONTEND workflow OBJECT WOULD BE ===")
print("workflow = {...active.workflow, presentation: active.presentation}")
print("workflow.repair = active.workflow.repair = stored.repair")
print()
print(f"stored.repair.status: {stored.get('repair', {}).get('status')}")
print(f"stored.repair.candidate.status: {stored.get('repair', {}).get('candidate', {}).get('status')}")
print(f"stored.repair.candidate.selectable: {stored.get('repair', {}).get('candidate', {}).get('selectable')}")

print()
print("=== FRONTEND selectable COMPUTATION ===")
repair = stored.get("repair", {})
candidate = repair.get("candidate", {})
validation = candidate.get("validation", {})
reanalysis = candidate.get("reanalysis", {})

candidateExists = candidate.get("status") == "validated"
valStatus = str(validation.get("status", "")).lower()
blockingFailures = validation.get("blocking_failures") or []
validationPassed = valStatus == "passed" and not blockingFailures
reanalysisStatus = reanalysis.get("status")
selectable = candidateExists and candidate.get("selectable") is True and validationPassed and reanalysisStatus == "AVAILABLE"

print(f"candidateExists: {candidateExists}")
print(f"candidate.selectable: {candidate.get('selectable')} (type: {type(candidate.get('selectable')).__name__})")
print(f"validationPassed: {validationPassed}")
print(f"reanalysis.status: {reanalysisStatus}")
print(f"Frontend selectable: {selectable}")

print()
print("=== RepairPage INITIAL GUARD CHECK ===")
print("Line 84: if(!repair) return <Empty>")
print(f"repair present: {bool(repair)}")
print("Result: RepairPage RENDERS (not empty)")

print()
print("=== WHERE IS THE TEST GENERATION DATA? ===")
tg = stored.get("test_generation", {})
print(f"stored.test_generation.status: {tg.get('status')}")
gen_tests = tg.get("generated_tests", [])
print(f"stored.test_generation.generated_tests count: {len(gen_tests)}")
if gen_tests:
    t0 = gen_tests[0]
    print(f"  [0].execution_status: {t0.get('execution_status')}")
    print(f"  [0].generated_code present: {bool(t0.get('generated_code'))}")
    print(f"  [0].type: {t0.get('type')}")
    print(f"  [0].test_kind: {t0.get('test_kind')}")

gen_int = tg.get("generated_integration_tests", [])
print(f"stored.test_generation.generated_integration_tests count: {len(gen_int)}")

print()
print("=== CANDIDATE COMPARISON ===")
cc = stored.get("candidate_comparison", {})
if cc:
    print(f"candidate_comparison.recommendation: {cc.get('recommendation')}")
    orig = cc.get("original", {})
    cand = cc.get("candidate", {})
    print(f"comparison.original keys: {list(orig.keys())}")
    print(f"comparison.candidate keys: {list(cand.keys())}")
    print(f"comparison.candidate.validation present: {bool(cand.get('validation'))}")

conn.close()
