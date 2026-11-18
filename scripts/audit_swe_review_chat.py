"""Real, streaming SWE-Review-Chat label audit; never clones or analyzes repos."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.swe_review_chat import SCHEMA_FIELDS, SUPPORTED_LANGUAGES, audit_records, iter_huggingface_records, parse_records

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--dataset", default="Suzhen/SWE-Review-Chat")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/swe_review_chat"))
    args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    records = parse_records(args.input) if args.input else iter_huggingface_records(args.dataset, args.split)
    report = {"status": "audited", "source": str(args.input) if args.input else args.dataset, "split": args.split,
              "schema_fields": list(SCHEMA_FIELDS), "schema_source": "Hugging Face Parquet schema",
              "language_mapping": {"status": "not_completed", "supported": list(SUPPORTED_LANGUAGES),
              "note": "No reliable primary language is exposed; use the separate GitHub metadata command."}}
    report.update(audit_records(records, ledger_path=out / "label_ledger.jsonl"))
    report["policy"] = {"higher_risk": "at least one qualifying human CHANGES_REQUESTED; prior intervention remains positive even if later approved/merged",
                         "lower_risk": "qualifying human APPROVED and merged, with no qualifying human CHANGES_REQUESTED",
                         "unknown": "insufficient, bot/agent-only, malformed, or otherwise ambiguous evidence"}
    per_repo = report["per_repository_label_distribution"]
    both = {r: v for r, v in per_repo.items() if v.get("HIGHER_RISK", 0) and v.get("LOWER_RISK", 0)}
    report["repositories_with_both_classes"] = len(both)
    report["positive_counts_per_repository"] = {r: v.get("HIGHER_RISK", 0) for r, v in per_repo.items() if v.get("HIGHER_RISK", 0)}
    report["candidate_pool_size"] = report["higher_risk"] + report["lower_risk"]
    report["maximum_defensible_candidate_cohort"] = report["candidate_pool_size"]
    report["target_3000_5000_realistic"] = report["candidate_pool_size"] >= 5000
    report["candidate_selection"] = {"seed": 42, "deduplicated": True, "repository_diverse": True,
                                     "development": "balanced/stratified later from genuine labels",
                                     "holdout": "natural-distribution pool retained separately",
                                     "language_filter": "apply after cached GitHub metadata enrichment"}
    (out / "repository_summary.json").write_text(json.dumps({"unique_repositories": report["unique_repositories"], "repositories_with_both_classes": len(both), "per_repository": per_repo, "positive_counts_per_repository": report["positive_counts_per_repository"]}, indent=2, sort_keys=True), encoding="utf-8")
    (out / "reviewer_summary.json").write_text(json.dumps({"reviewer_type_distribution": report["reviewer_type_distribution"], "review_event_type_distribution": report["review_event_type_distribution"], "review_verdict_distribution": report["review_verdict_distribution"]}, indent=2, sort_keys=True), encoding="utf-8")
    (out / "candidate_pool_summary.json").write_text(json.dumps({"candidate_pool_size": report["candidate_pool_size"], "higher_risk": report["higher_risk"], "lower_risk": report["lower_risk"], "unknown": report["unknown"], "repositories_with_both_classes": len(both), "selection": report["candidate_selection"]}, indent=2, sort_keys=True), encoding="utf-8")
    (out / "cheap_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__": main()
