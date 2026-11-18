
import sqlite3
import json
import sys

FASTAPI_ID = "77a3f31b-a3a2-42a9-8960-f51d660b466a"

conn = sqlite3.connect("backend/accounts.sqlite3")
cur = conn.cursor()

# List tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("TABLES:", tables)

# Find the analysis table
analysis_table = None
for t in tables:
    if "analys" in t.lower():
        analysis_table = t
        print(f"Found analysis table: {t}")

if not analysis_table:
    print("No analysis table found!")
    conn.close()
    sys.exit(1)

# Get columns
cur.execute(f"PRAGMA table_info({analysis_table})")
cols = [r[1] for r in cur.fetchall()]
print("Columns:", cols)

# Get FastAPI analysis
cur.execute(f"SELECT * FROM {analysis_table} WHERE id=?", (FASTAPI_ID,))
row = cur.fetchone()
if not row:
    print(f"No row found for {FASTAPI_ID}")
    conn.close()
    sys.exit(1)

row_dict = dict(zip(cols, row))
print(f"\nFastAPI analysis row keys: {list(row_dict.keys())}")
print(f"Status: {row_dict.get('status')}")

# Parse stored JSON
stored_raw = row_dict.get("stored") or row_dict.get("data") or row_dict.get("result")
if stored_raw:
    stored = json.loads(stored_raw) if isinstance(stored_raw, str) else stored_raw
    stored_keys = list(stored.keys())
    print(f"stored keys: {stored_keys}")
    
    # Check workflow
    workflow = stored.get("workflow", {})
    if workflow:
        wf_keys = list(workflow.keys())
        print(f"\nworkflow keys: {wf_keys}")
        
        repair = workflow.get("repair", {})
        if repair:
            print(f"\nrepair keys: {list(repair.keys())}")
            print(f"  repair.status: {repair.get('status')}")
            print(f"  repair.provider: {repair.get('provider')}")
            print(f"  repair.model: {repair.get('model')}")
            print(f"  repair.patch (present): {'patch' in repair}")
            print(f"  repair.production_patch (present): {'production_patch' in repair}")
            
            candidate = repair.get("candidate", {})
            if candidate:
                print(f"\n  repair.candidate keys: {list(candidate.keys())}")
                print(f"  repair.candidate.selectable: {candidate.get('selectable')}")
                print(f"  repair.candidate.status: {candidate.get('status')}")
            else:
                print(f"  repair.candidate: NOT PRESENT")
            
            # Check for test fields
            for f in ["generated_tests", "regression_test", "integration_test", "tests"]:
                if f in repair:
                    val = repair[f]
                    print(f"  repair.{f}: present, type={type(val).__name__}")
        else:
            print("  repair: NOT in workflow")
        
        # Check top-level stored fields
        for key in ["candidate_reanalysis", "comparison", "presentation"]:
            val = stored.get(key) or workflow.get(key)
            print(f"\n{key}: {'present' if val else 'NOT present'}")
    
    # Save full stored to file for inspection
    with open("scripts/fastapi_stored_full.json", "w") as f:
        json.dump(stored, f, indent=2, default=str)
    print("\nFull stored saved to scripts/fastapi_stored_full.json")
else:
    print("No stored data found")

# Now find a working Calculator analysis
print("\n\n=== FINDING WORKING CALCULATOR ANALYSIS ===")
cur.execute(f"SELECT id, status FROM {analysis_table} WHERE id != ? LIMIT 100", (FASTAPI_ID,))
all_rows = cur.fetchall()

calc_ids = []
for aid, astatus in all_rows:
    cur.execute(f"SELECT stored FROM {analysis_table} WHERE id=?", (aid,))
    r = cur.fetchone()
    if r and r[0]:
        try:
            s = json.loads(r[0]) if isinstance(r[0], str) else r[0]
            wf = s.get("workflow", {})
            repair = wf.get("repair", {})
            candidate = repair.get("candidate", {})
            if candidate.get("selectable") == True:
                repo_id = s.get("repository_id", "")
                print(f"  Selectable candidate: id={aid}, status={astatus}")
                calc_ids.append(aid)
                if len(calc_ids) >= 3:
                    break
        except:
            pass

if calc_ids:
    calc_id = calc_ids[0]
    cur.execute(f"SELECT stored FROM {analysis_table} WHERE id=?", (calc_id,))
    r = cur.fetchone()
    calc_stored = json.loads(r[0]) if isinstance(r[0], str) else r[0]
    
    calc_wf = calc_stored.get("workflow", {})
    calc_repair = calc_wf.get("repair", {})
    calc_candidate = calc_repair.get("candidate", {})
    
    print(f"\nCalc analysis {calc_id}:")
    print(f"  repair keys: {list(calc_repair.keys())}")
    print(f"  repair.candidate keys: {list(calc_candidate.keys())}")
    print(f"  repair.candidate.selectable: {calc_candidate.get('selectable')}")
    
    with open("scripts/calc_stored_full.json", "w") as f:
        json.dump(calc_stored, f, indent=2, default=str)
    print(f"  Full stored saved to scripts/calc_stored_full.json")
    
    # Now compare top-level repair structures
    print("\n=== STRUCTURAL COMPARISON ===")
    def flatten_keys(d, prefix=""):
        keys = []
        if isinstance(d, dict):
            for k, v in d.items():
                keys.append(f"{prefix}.{k}" if prefix else k)
                keys.extend(flatten_keys(v, f"{prefix}.{k}" if prefix else k))
        return keys
    
    fastapi_repair = stored.get("workflow", {}).get("repair", {})
    calc_repair2 = calc_stored.get("workflow", {}).get("repair", {})
    
    fkeys = set(flatten_keys(fastapi_repair, "repair"))
    ckeys = set(flatten_keys(calc_repair2, "repair"))
    
    only_in_calc = ckeys - fkeys
    only_in_fastapi = fkeys - ckeys
    
    print(f"\nKeys ONLY in Calculator repair (missing from FastAPI):")
    for k in sorted(only_in_calc):
        print(f"  MISSING: {k}")
    
    print(f"\nKeys ONLY in FastAPI repair (extra vs Calculator):")
    for k in sorted(only_in_fastapi):
        print(f"  EXTRA: {k}")

conn.close()
