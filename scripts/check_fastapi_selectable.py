import sqlite3, json

FASTAPI_ID = "77a3f31b-a3a2-42a9-8960-f51d660b466a"
conn = sqlite3.connect("backend/accounts.sqlite3")
cur = conn.cursor()
cur.execute("SELECT result FROM analyses WHERE id=?", (FASTAPI_ID,))
row = cur.fetchone()
stored = json.loads(row[0])

repair = stored.get("repair", {})
candidate = repair.get("candidate", {})
validation = candidate.get("validation", {})
reanalysis = candidate.get("reanalysis", {})

print("=== KEY FIELDS FOR FRONTEND SELECTABLE CHECK ===")
print("repair.status:", repair.get("status"))
print("repair.patch (present):", bool(repair.get("patch")))
print("repair.candidate.status:", candidate.get("status"))
print("repair.candidate.selectable:", candidate.get("selectable"))
print("repair.candidate.validation.status:", validation.get("status"))
print("repair.candidate.validation.blocking_failures:", validation.get("blocking_failures"))
print("repair.candidate.reanalysis (present):", bool(reanalysis))
print("repair.candidate.reanalysis.status:", reanalysis.get("status"))

# Replicate frontend candidateExists check
candidateExists = candidate.get("status") == "validated"
print("\ncandidateExists:", candidateExists)

# validationPassed check
valStatus = str(validation.get("status", "")).lower()
blockingFailures = validation.get("blocking_failures") or []
validationPassed = valStatus == "passed" and not blockingFailures
print("validationPassed:", validationPassed, "(status=" + valStatus + ", blocking_failures=" + str(blockingFailures) + ")")

# selectable in frontend
reanalysisStatus = reanalysis.get("status")
print("reanalysis.status:", reanalysisStatus)
selectable = candidateExists and candidate.get("selectable") == True and validationPassed and reanalysisStatus == "AVAILABLE"
print("Frontend selectable:", selectable)

print()
print("=== TEST GENERATION ===")
test_gen = stored.get("test_generation", {})
print("test_generation present:", bool(test_gen))
if test_gen:
    print("test_generation.status:", test_gen.get("status"))
    gen_tests = test_gen.get("generated_tests", [])
    print("generated_tests count:", len(gen_tests))
    if gen_tests:
        print("generated_tests[0] keys:", list(gen_tests[0].keys()))
        t0 = gen_tests[0]
        print("  execution_status:", t0.get("execution_status"))
        print("  generated_code present:", bool(t0.get("generated_code")))

print()
print("=== CANDIDATE COMPARISON ===")
comparison = stored.get("candidate_comparison", {})
print("candidate_comparison present:", bool(comparison))
if comparison:
    print("comparison keys:", list(comparison.keys()))
    cand_comp = comparison.get("candidate", {})
    print("comparison.candidate.validation present:", bool(cand_comp.get("validation")))

print()
print("=== workflow_id in stored ===")
print("workflow_id:", stored.get("workflow_id"))
print("finalized_demo:", stored.get("finalized_demo"))

conn.close()
