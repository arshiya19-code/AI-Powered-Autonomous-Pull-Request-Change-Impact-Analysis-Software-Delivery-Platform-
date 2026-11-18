from datetime import UTC, datetime
import json

from dataset.phase7d_outcome_evidence import OutcomeEvidenceResult
from dataset.phase7d_outcome_policy import outcome_window
from scripts import acquire_aidev_300_outcomes as outcomes


def test_retry_previous_401_accepts_dictionary_backed_ledger_and_retries_only_targets(tmp_path, monkeypatch) -> None:
    merge_at = "2025-01-01T00:00:00Z"
    ledger_rows = [
        {"sample_id": str(index), "repo_full_name": "org/repo", "pr_number": index,
         "acquisition_status": "resolved", "merged_at": merge_at,
         "merge_commit_sha": "merge", "final_pr_head_sha": "head", "changed_files": []}
        for index in range(300)
    ]
    cutoff_at = datetime(2026, 9, 6, tzinfo=UTC)
    labels = [outcomes._label_row(
        row, OutcomeEvidenceResult(outcome_window(datetime(2025, 1, 1, tzinfo=UTC)), (), False, "unmerged"), cutoff_at)
        for row in ledger_rows
    ]
    labels[7]["unknown_or_conflicted_reason"] = outcomes.PREVIOUS_401_PREFIX
    ledger_path = tmp_path / "ledger.jsonl"
    output_dir = tmp_path / "outcomes"
    output_dir.mkdir()
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in ledger_rows), encoding="utf-8")
    (output_dir / "phase7d_labels_300.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    (output_dir / "phase7d_outcome_evidence_300.jsonl").write_text("", encoding="utf-8")

    calls: list[str] = []

    class Provider:
        def __init__(self, *_args) -> None:
            pass

        def collect(self, *, pr_id: str, merge_at: datetime, retry_previous_401: bool, **_kwargs) -> OutcomeEvidenceResult:
            calls.append(pr_id)
            assert retry_previous_401
            return OutcomeEvidenceResult(outcome_window(merge_at), (), True)

    monkeypatch.setattr(outcomes, "GitHubOutcomeEvidenceProvider", Provider)

    report = outcomes.retry_previous_401(
        ledger_path=ledger_path, output_dir=output_dir, cutoff_at=cutoff_at, client=object())

    assert calls == ["7"]
    assert report["targeted_retry"] == [{"sample_id": "7", "repository": "org/repo", "pr_number": 7,
                                         "acquisition_status": "evaluated", "reason": None}]
