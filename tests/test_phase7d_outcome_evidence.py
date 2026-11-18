from datetime import UTC, datetime

import json

from dataset.phase7d_outcome_evidence import CachedOutcomeEvidenceProvider, GitHubOutcomeEvidenceProvider, OutcomeEvidence, UnavailableOutcomeEvidenceProvider
from dataset.phase7f_identity_acquisition import GitHubClient, HTTPResult
from dataset.phase7d_outcome_policy import OutcomeEvidenceClassification


def test_cache_limits_evidence_to_the_90_day_window() -> None:
    merge = datetime(2025, 1, 1, tzinfo=UTC)
    provider = CachedOutcomeEvidenceProvider((
        OutcomeEvidence(OutcomeEvidenceClassification.REVERT, "a", datetime(2025, 1, 2, tzinfo=UTC), "cache", merge, "explicit", confirmed=True),
        OutcomeEvidence(OutcomeEvidenceClassification.REVERT, "late", datetime(2025, 5, 2, tzinfo=UTC), "cache", merge, "explicit", confirmed=True),
        OutcomeEvidence(OutcomeEvidenceClassification.REVERT, "at-merge", merge, "cache", merge, "explicit", confirmed=True),
        OutcomeEvidence(OutcomeEvidenceClassification.REVERT, "at-end", datetime(2025, 4, 1, tzinfo=UTC), "cache", merge, "explicit", confirmed=True),
    ))
    result = provider.collect(repository="o/r", pr_id="1", merge_sha="m", merge_at=merge)
    assert result.evaluated and [item.reference for item in result.evidence] == ["a", "at-end"]


def test_unavailable_provider_is_explicit() -> None:
    result = UnavailableOutcomeEvidenceProvider("offline").collect(repository="o/r", pr_id="1", merge_sha="m", merge_at=datetime(2025, 1, 1, tzinfo=UTC))
    assert not result.evaluated and result.unavailable_reason == "offline"


def test_github_provider_reuses_immutable_snapshot_and_retains_provenance(tmp_path) -> None:
    merge = datetime(2025, 1, 1, tzinfo=UTC)
    commit_url = "https://api.github.com/search/commits?q=repo%3Ao%2Fr+merge&per_page=100"
    issue_url = "https://api.github.com/search/issues?q=repo%3Ao%2Fr+is%3Apr+is%3Amerged+merged%3A2025-01-01..2025-04-01+%231&per_page=100"
    payloads = {
        commit_url: HTTPResult(200, {}, json.dumps({"items": [{"sha": "revert", "html_url": "https://x/revert", "commit": {"message": "This reverts commit merge", "author": {"date": "2025-01-02T00:00:00Z"}}}]}).encode(), commit_url),
        issue_url: HTTPResult(200, {}, b'{"items": []}', issue_url),
    }
    client = GitHubClient("test", tmp_path / "raw", transport=lambda url, *_: payloads[url], sleep=lambda _: None)
    provider = GitHubOutcomeEvidenceProvider(client, tmp_path / "outcomes")
    first = provider.collect(repository="o/r", pr_id="1", merge_sha="merge", merge_at=merge, original_head_sha="head", original_paths=["a.py"])
    assert first.evaluated and first.evidence[0].original_head_sha == "head"
    assert first.evidence[0].cache_checksum and first.evidence[0].observed_commit_sha == "revert"
    second = GitHubOutcomeEvidenceProvider(GitHubClient("test", tmp_path / "raw", transport=lambda *_: (_ for _ in ()).throw(AssertionError("network used"))), tmp_path / "outcomes").collect(repository="o/r", pr_id="1", merge_sha="merge", merge_at=merge)
    assert second == first


def test_authenticated_retry_preserves_401_snapshot_and_uses_new_retry_snapshot(tmp_path) -> None:
    merge = datetime(2025, 1, 1, tzinfo=UTC)
    commit_url = "https://api.github.com/search/commits?q=repo%3Ao%2Fr+merge&per_page=100"
    failing = GitHubOutcomeEvidenceProvider(GitHubClient("bad", tmp_path / "raw", transport=lambda url, *_: HTTPResult(401, {}, b'{\"message\":\"Bad credentials\"}', url), sleep=lambda _: None), tmp_path / "outcomes")
    first = failing.collect(repository="o/r", pr_id="1", merge_sha="merge", merge_at=merge)
    assert not first.evaluated and "github_http_401" in str(first.unavailable_reason)
    original = next((tmp_path / "outcomes").glob("*.json"))
    payloads = {
        commit_url: HTTPResult(200, {}, b'{\"items\": []}', commit_url),
        "https://api.github.com/search/issues?q=repo%3Ao%2Fr+is%3Apr+is%3Amerged+merged%3A2025-01-01..2025-04-01+%231&per_page=100": HTTPResult(200, {}, b'{\"items\": []}', "issues"),
    }
    retry = GitHubOutcomeEvidenceProvider(GitHubClient("good", tmp_path / "raw", transport=lambda url, *_: payloads[url], sleep=lambda _: None), tmp_path / "outcomes")
    second = retry.collect(repository="o/r", pr_id="1", merge_sha="merge", merge_at=merge, retry_previous_401=True)
    assert second.evaluated
    assert original.exists()
    assert len(list((tmp_path / "outcomes" / "authenticated_retries").glob("*.json"))) == 1
