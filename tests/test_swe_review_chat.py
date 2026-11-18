from pathlib import Path
import json
import pytest
from dataset.swe_review_chat import DurablePRCache, audit_records, assert_no_leakage, choose_cohort, grouped_repository_split, label_record
from dataset.config import ALL_FEATURES

def rec(**kw):
    return {"repo_full_name":"o/r", "pr_number":1, "state":"merged", "merged_at":"2025-01-01", "review_conversations":[], **kw}
def review(state, typ="User"): return {"state":state,"reviewer":"r","reviewer_type":typ,"timestamp":"2025-01-01"}
def test_label_policy_and_unknowns():
    assert label_record(rec(review_conversations=[review("CHANGES_REQUESTED")])).label == 1
    assert label_record(rec(review_conversations=[review("APPROVED")])).label == 0
    assert label_record(rec(review_conversations=[])).label is None
    decision = label_record(rec(review_conversations=[review("APPROVED"), review("CHANGES_REQUESTED")]))
    assert decision.label == 1 and decision.reason == "human_changes_requested_with_later_approval"
def test_bot_duplicate_and_malformed_audit():
    rows=[rec(review_conversations=[review("APPROVED","Bot")]), rec(pr_number=1), {"state":"merged"}]
    report=audit_records(rows); assert report["bot_only_rejected"] == 1; assert report["duplicates"] == 1; assert report["malformed_missing_evidence"] == 1
def test_leakage_and_deterministic_selection():
    with pytest.raises(ValueError): assert_no_leakage(["merged_at"])
    rows=[rec(pr_number=i, review_conversations=[review("APPROVED")]) for i in range(10)]
    assert [x["pr_number"] for x in choose_cohort(rows, requested=5)] == [x["pr_number"] for x in choose_cohort(rows, requested=5)]
def test_atomic_cache_checkpoint(tmp_path: Path):
    cache=DurablePRCache(tmp_path); cache.put("o/r#1", {"analysis_mode":"full"}); cache.checkpoint({"completed":1})
    assert cache.get("o/r#1")["analysis_mode"] == "full"; assert not list(tmp_path.rglob("*.tmp"))
def test_grouping_and_canonical_schema_untouched():
    split = grouped_repository_split([rec(repo_full_name=f"o/r{i}") for i in range(20)])
    assert not (set(split["train"]) & set(split["test"])); assert len(ALL_FEATURES) == 42
