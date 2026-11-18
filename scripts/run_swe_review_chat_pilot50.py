"""Build and run only the SWE-Review-Chat 50-PR engineering pilot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.swe_review_chat import parse_records
from dataset.swe_review_chat_pilot import (
    PILOT_SIZE, PilotPaths, choose_pilot50, load_language_metadata, preflight_pilot50,
    read_cohort, run_pilot, write_cohort,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the resumable SWE-Review-Chat pilot50 only")
    ap.add_argument("--root", type=Path, default=Path("data/processed/swe_review_chat"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--git-timeout", type=int, default=120)
    ap.add_argument("--cohort-only", action="store_true")
    ap.add_argument("--diagnostic", action="store_true", help="Re-run only selected existing attempts; never writes feature caches")
    ap.add_argument("--diagnostic-identities", nargs="*", help="Exact PR identities; defaults to the first three cohort attempts")
    ap.add_argument("--max-pr-runtime", type=float, default=300)
    ap.add_argument("--max-source-files", type=int, default=2000)
    args = ap.parse_args()
    root = args.root
    paths = PilotPaths(root)
    paths.prepare()
    cohort_path = paths.cohort
    if not cohort_path.exists():
        rows = list(parse_records(root / "label_ledger.jsonl"))
        languages = load_language_metadata(root / "repository_language_summary.json")
        cohort = choose_pilot50(rows, languages)
        write_cohort(cohort, cohort_path)
    else:
        cohort = read_cohort(cohort_path)
    if len(cohort) != PILOT_SIZE or len({r["identity"] for r in cohort}) != PILOT_SIZE:
        raise SystemExit("pilot50_cohort.jsonl does not contain exactly 50 unique PRs")
    print("selected label distribution:", dict(Counter(r["label_name"] for r in cohort)))
    print("selected language distribution:", dict(Counter(r["language"] for r in cohort)))
    print("selected repository count:", len({r["repo_full_name"] for r in cohort}))
    if args.cohort_only:
        return
    # Resolve stale/ineligible persisted rows before any API or git operation.
    # The complete local ledger is the replacement pool; it preserves labels
    # and makes replacement selection reproducible.
    ledger = root / "label_ledger.jsonl"
    candidate_rows = list(parse_records(ledger)) if ledger.exists() else []
    languages = load_language_metadata(root / "repository_language_summary.json") if (root / "repository_language_summary.json").exists() else {}
    for candidate in candidate_rows:
        candidate.setdefault("identity", f"{candidate.get('repo_full_name', '')}#{candidate.get('pr_number', '')}")
        if candidate.get("repo_full_name") in languages:
            candidate["language"] = languages[candidate["repo_full_name"]]
    cohort, preflight = preflight_pilot50(cohort, paths, candidate_rows)
    print("pilot50 pre-flight:", json.dumps(preflight, sort_keys=True))
    audit_path = paths.pilot / "pilot50_cohort_audit.jsonl"
    write_cohort(cohort, audit_path)
    attempted = {str(r["identity"]) for r in cohort if paths.cache.joinpath("per_pr", __import__("hashlib").sha256(str(r["identity"]).encode()).hexdigest() + ".json").exists()}
    diagnostic_ids = set(args.diagnostic_identities or attempted or [str(r["identity"]) for r in cohort[:3]])
    if args.diagnostic:
        cohort = [r for r in cohort if r["identity"] in diagnostic_ids]
    result = run_pilot(cohort, paths, token=os.environ.get("GITHUB_TOKEN"), workers=args.workers, git_timeout=args.git_timeout,
                       diagnostic=args.diagnostic, diagnostic_identities=diagnostic_ids,
                       max_pr_runtime=args.max_pr_runtime, max_source_files=args.max_source_files)
    if args.diagnostic:
        print(json.dumps(result["report"], indent=2, sort_keys=True))
        print("SWE-REVIEW-CHAT PERFORMANCE DIAGNOSTIC COMPLETE")
        return
    # A second pass performs no more than the bounded transient retry; after
    # that retry is exhausted, transient failures remain current failures.
    resumed = run_pilot(cohort, paths, token=os.environ.get("GITHUB_TOKEN"), workers=args.workers, git_timeout=args.git_timeout)
    report = resumed["report"]
    report["cache_resume_verified"] = all(r.get("cache_hit") for r in resumed["results"])
    (paths.pilot / "pilot50_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("SWE-REVIEW-CHAT PILOT50 COMPLETE")


if __name__ == "__main__":
    main()
