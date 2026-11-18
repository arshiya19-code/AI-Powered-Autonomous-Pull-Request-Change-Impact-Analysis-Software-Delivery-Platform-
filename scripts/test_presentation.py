"""
Simulate build_presentation for FastAPI analysis and compare with frontend expectations.
"""
import sys
sys.path.insert(0, '.')

from backend.app.presentation import build_presentation
import sqlite3
import json

FASTAPI_ID = "77a3f31b-a3a2-42a9-8960-f51d660b466a"

conn = sqlite3.connect("backend/accounts.sqlite3")
cur = conn.cursor()
cur.execute("SELECT result, status FROM analyses WHERE id=?", (FASTAPI_ID,))
row = cur.fetchone()
conn.close()

stored = json.loads(row[0])
status = row[1]

print(f"=== RUNNING build_presentation for FastAPI Analysis {FASTAPI_ID} ===\n")
presentation = build_presentation(FASTAPI_ID, status, stored)

print("repair section from presentation:")
repair_p = presentation.get("repair", {})
print(f"  repair.label: {repair_p.get('label')}")
print(f"  repair.message: {repair_p.get('message')}")
print(f"  repair.attempted: {repair_p.get('attempted')}")
print(f"  repair.candidate_status: {repair_p.get('candidate_status')}")

print()
print("=== FRONTEND RepairPage KEY CONDITION (line 84) ===")
print("const repair = workflow?.repair")
repair_stored = stored.get("repair", {})
print(f"workflow.repair is truthy: {bool(repair_stored)}")

print()
print("=== FRONTEND candidateExists (line 84) ===")
candidate = repair_stored.get("candidate", {})
candidateExists = bool(candidate) and candidate.get("status") == "validated"
print(f"candidateExists: {candidateExists}")

print()
print("=== PRESENTATION summary.repair ===")
summary = presentation.get("summary", {})
print(f"summary.repair: {summary.get('repair')}")

print()
print("=== FULL PRESENTATION REPAIR ===")
print(json.dumps(repair_p, indent=2))

print()
print("=== POTENTIAL ISSUE: presentation.repair.candidate_status ===")
print(f"Expected 'validated', got: '{repair_p.get('candidate_status')}'")

print()
print("=== WHAT FRONTEND READS ===")
print("active.workflow = stored (full dict)")
print("active.presentation = build_presentation()")
print("workflow (frontend line 108) = {...active.workflow, presentation: active.presentation}")
print()
print("RepairPage line 84:")
print("  const repair = workflow?.repair  // = stored.repair")
print("  const p = active?.presentation?.repair  // = build_presentation().repair")
print()
print(f"  workflow.repair present: {bool(repair_stored)}")
print(f"  workflow.repair.candidate.selectable: {candidate.get('selectable')}")
print(f"  workflow.repair.candidate.status: {candidate.get('status')}")
print()
print("Does RepairPage show?", "YES - repair is truthy" if repair_stored else "NO - empty guard returns early")
