from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime

import pytest

import dataset.phase7d_outcome_run as outcome_run
from dataset.phase7c_contract import LabelState
from dataset.phase7d_outcome_evidence import OutcomeEvidenceResult
from dataset.phase7d_outcome_policy import outcome_window
from dataset.phase7f_identity_acquisition import GitHubClient, HTTPResult
from scripts.build_phase7ef_dataset_and_train import _expansion_is_complete


def _ledger(path, count=6):
    rows = [{"sample_id": str(i), "repo_full_name": "o/r", "pr_number": i,
             "acquisition_status": "resolved", "merged_at": "2025-01-01T00:00:00Z",
             "merge_commit_sha": f"sha-{i}", "final_pr_head_sha": "head", "changed_files": [{"path": "a.py"}]}
            for i in range(1, count + 1)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def test_outcome_workers_are_bounded_and_results_are_deterministic(monkeypatch, tmp_path):
    active = 0
    maximum = 0
    lock = threading.Lock()

    class Provider:
        def __init__(self, *_args, **_kwargs): pass
        def collect(self, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep((7 - int(kwargs["pr_id"])) * 0.002)
            with lock:
                active -= 1
            return OutcomeEvidenceResult(outcome_window(datetime(2025, 1, 1, tzinfo=UTC)), (), True)

    monkeypatch.setattr(outcome_run, "GitHubOutcomeEvidenceProvider", Provider)
    ledger = tmp_path / "ledger.jsonl"
    rows = _ledger(ledger)
    outcome_run.build_outcome_artifacts(ledger_path=ledger, output_dir=tmp_path / "outcomes",
                                        cutoff_at=datetime(2026, 1, 1, tzinfo=UTC), client=object(), outcome_workers=2)
    labels = outcome_run.load_jsonl(tmp_path / "outcomes" / "phase7d_labels.jsonl")
    assert maximum <= 2
    assert [row["sample_id"] for row in labels] == [row["sample_id"] for row in rows]
    assert all(row["risk_label"] == LabelState.LOW_OBSERVED.value for row in labels)


def test_incremental_cache_reuse_skips_completed_rows(monkeypatch, tmp_path):
    calls = []

    class Provider:
        def __init__(self, *_args, **_kwargs): pass
        def collect(self, **kwargs):
            calls.append(kwargs["pr_id"])
            return OutcomeEvidenceResult(None, (), False, "temporary")

    monkeypatch.setattr(outcome_run, "GitHubOutcomeEvidenceProvider", Provider)
    ledger = tmp_path / "ledger.jsonl"
    _ledger(ledger, 3)
    output = tmp_path / "outcomes"
    partial = [{"sample_id": "1", "repository": "o/r", "pr_number": 1, "resolved": True,
                "merged": True, "mature": True, "risk_label": "unknown", "confidence": "none",
                "evidence_type": "none", "evidence_references": [], "observation_start": None,
                "observation_end": None, "label_policy_version": "phase7d", "unknown_or_conflicted_reason": "temporary",
                "evidence_acquisition_failure": True}]
    (output).mkdir()
    (output / "phase7d_labels.jsonl").write_text(json.dumps(partial[0]) + "\n", encoding="utf-8")
    outcome_run.build_outcome_artifacts(ledger_path=ledger, output_dir=output,
                                        cutoff_at=datetime(2026, 1, 1, tzinfo=UTC), client=object(), outcome_workers=2)
    assert calls == ["2", "3"]
    assert [row["sample_id"] for row in outcome_run.load_jsonl(output / "phase7d_labels.jsonl")] == ["1", "2", "3"]


def test_permanent_failure_isolated_and_unknown_preserved(monkeypatch, tmp_path):
    class Provider:
        def __init__(self, *_args, **_kwargs): pass
        def collect(self, **kwargs):
            if kwargs["pr_id"] == "2":
                raise RuntimeError("permanent")
            return OutcomeEvidenceResult(None, (), False, "unavailable")

    monkeypatch.setattr(outcome_run, "GitHubOutcomeEvidenceProvider", Provider)
    ledger = tmp_path / "ledger.jsonl"
    _ledger(ledger, 3)
    outcome_run.build_outcome_artifacts(ledger_path=ledger, output_dir=tmp_path / "outcomes",
                                        cutoff_at=datetime(2026, 1, 1, tzinfo=UTC), client=object(), outcome_workers=3)
    labels = outcome_run.load_jsonl(tmp_path / "outcomes" / "phase7d_labels.jsonl")
    assert len(labels) == 3
    assert all(row["risk_label"] == LabelState.UNKNOWN.value for row in labels)


def test_rate_limit_and_transient_failures_back_off_and_retry(tmp_path):
    attempts = []
    responses = [HTTPResult(429, {"retry-after": "0", "x-ratelimit-remaining": "0"}, b'{"message":"secondary rate limit"}', "u"),
                 HTTPResult(503, {"X-RateLimit-Remaining": "12"}, b"{}", "u"),
                 HTTPResult(200, {"X-RateLimit-Remaining": "11"}, b"{}", "u")]
    client = GitHubClient("t", tmp_path, retries=3, transport=lambda *_: (attempts.append(1) or responses.pop(0)), sleep=lambda _: None)
    assert client.get_json("u")[0] == {}
    assert len(attempts) == 3
    assert client.telemetry()["rate_limit_remaining"] == 11


def test_shared_persistence_retries_windows_permission_error_and_writes_per_pr_checkpoint(monkeypatch, tmp_path):
    class Provider:
        def __init__(self, *_args, **_kwargs): pass
        def collect(self, **kwargs):
            return OutcomeEvidenceResult(outcome_window(datetime(2025, 1, 1, tzinfo=UTC)), (), True)

    monkeypatch.setattr(outcome_run, "GitHubOutcomeEvidenceProvider", Provider)
    real_replace = outcome_run.os.replace
    attempts = {"count": 0}

    def flaky_replace(source, destination):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError(5, "Access is denied")
        return real_replace(source, destination)

    monkeypatch.setattr(outcome_run.os, "replace", flaky_replace)
    ledger = tmp_path / "ledger.jsonl"
    _ledger(ledger, 1)
    output = tmp_path / "outcomes"
    outcome_run.build_outcome_artifacts(
        ledger_path=ledger, output_dir=output, cutoff_at=datetime(2026, 1, 1, tzinfo=UTC),
        client=object(), outcome_workers=1,
    )
    assert attempts["count"] >= 2
    assert len(list((output / "per_pr_checkpoints").glob("*.json"))) == 1
    assert not list((output / "per_pr_checkpoints").glob("*.tmp"))


def test_incomplete_expansion_is_not_successful() -> None:
    assert not _expansion_is_complete({"counters": {"accepted": 35}}, 3000)
    assert _expansion_is_complete({"counters": {"accepted": 3000}}, 3000)
