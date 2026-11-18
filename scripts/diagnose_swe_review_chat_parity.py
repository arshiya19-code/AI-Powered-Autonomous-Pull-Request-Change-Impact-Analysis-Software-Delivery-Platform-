"""Local, bounded semantic parity check for the preserved spellbook PR.

No network, labels, feature caches, or pilot checkpoints are touched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyzer.ml.dataset_builder import FEATURE_NAMES, _run_full_analysis

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("patch", type=Path, help="saved unified diff")
    ap.add_argument("base_sha")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    patch = args.patch.read_text(encoding="utf-8")
    old_diag, new_diag = {}, {}
    old = _run_full_analysis(args.repo, patch, args.base_sha, old_diag, force_full_repository=True)
    new = _run_full_analysis(args.repo, patch, args.base_sha, new_diag)
    if old is None or new is None:
        raise SystemExit("PARITY_UNAVAILABLE: both analyses must produce features")
    rows = [{"feature": name, "old_value": old.get(name), "new_value": new.get(name),
             "equal": old.get(name) == new.get(name)} for name in FEATURE_NAMES]
    result = {"status": "PASS" if all(r["equal"] for r in rows) else "FAIL",
              "feature_count": len(rows), "features": rows,
              "old_diagnostics": old_diag, "new_diagnostics": new_diag}
    rendered = json.dumps(result, indent=2, sort_keys=False)
    if args.output: args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["status"] != "PASS": raise SystemExit(1)

if __name__ == "__main__": main()
