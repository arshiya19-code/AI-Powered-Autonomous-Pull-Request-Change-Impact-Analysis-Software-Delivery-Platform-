from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from dataset.phase7f_identity_acquisition import (
    AcquisitionError,
    GitHubClient,
    HTTPResult,
    IdentityAcquirer,
    PRIdentity,
    build_report,
)


def _transport(payloads: dict[str, HTTPResult]):
    def send(url: str, headers: dict[str, str], timeout: int) -> HTTPResult:
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"
        return payloads[url]
    return send


def test_acquisition_persists_identity_shas_files_languages_and_raw_cache(tmp_path) -> None:
    base = "https://api.github.com/repos/acme/project/pulls/7"
    pr = {
        "number": 7, "html_url": "https://github.com/acme/project/pull/7", "merged_at": "2025-01-01T00:00:00Z",
        "merge_commit_sha": "merge", "base": {"sha": "base", "repo": {"id": 99, "full_name": "acme/project"}},
        "head": {"sha": "head"},
    }
    files = [{"filename": "src/a.py"}, {"filename": "web/app.ts"}, {"filename": "README.md"}]
    payloads = {
        base: HTTPResult(200, {}, json.dumps(pr).encode(), base),
        base + "/files?per_page=100&page=1": HTTPResult(200, {}, json.dumps(files).encode(), base + "/files?per_page=100&page=1"),
    }
    client = GitHubClient("test", tmp_path / "raw", transport=_transport(payloads), sleep=lambda _: None)
    result = IdentityAcquirer(client, tmp_path, datetime(2026, 6, 1, tzinfo=UTC)).acquire_one(
        PRIdentity("1", "acme/project", "old-id", "1", 7)
    )
    assert result["acquisition_status"] == "resolved"
    assert result["repo_id"] == "99"
    assert result["base_sha"] == "base"
    assert result["final_pr_head_sha"] == "head"
    assert result["merge_commit_sha"] == "merge"
    assert result["changed_language_set"] == ["python", "typescript", "unknown"]
    assert len(result["raw_response_cache"]) == 2
    assert all((tmp_path / path).exists() for path in result["raw_response_cache"])


def test_inaccessible_pr_is_explicit_unresolved_failure(tmp_path) -> None:
    url = "https://api.github.com/repos/acme/private/pulls/8"
    client = GitHubClient("test", tmp_path / "raw", transport=_transport({
        url: HTTPResult(404, {}, b'{"message":"Not Found"}', url),
    }), sleep=lambda _: None)
    result = IdentityAcquirer(client, tmp_path, datetime(2026, 1, 1, tzinfo=UTC)).acquire_one(
        PRIdentity("2", "acme/private", "2", "2", 8)
    )
    assert result["acquisition_status"] == "unresolved"
    assert result["failure_reason"].startswith("github_http_404")
    assert result["base_sha"] is None


def test_manifest_and_ledger_are_never_overwritten(tmp_path) -> None:
    source = tmp_path / "fixed.parquet"
    source.write_bytes(b"fixed")
    identity = PRIdentity("1", "acme/project", "2", "1", 7)
    client = GitHubClient("test", tmp_path / "raw", transport=lambda *_: pytest.fail("not called"))
    acquirer = IdentityAcquirer(client, tmp_path / "out", datetime(2026, 1, 1, tzinfo=UTC))
    acquirer.write_immutable_manifest([identity], source)
    with pytest.raises(AcquisitionError, match="immutable manifest"):
        acquirer.write_immutable_manifest([identity], source)
    (tmp_path / "out" / "identity_ledger.jsonl").write_text("already there\n", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="refusing to overwrite"):
        acquirer.acquire_all([identity])


def test_report_is_acquisition_only_and_excludes_unknown_from_adapter_requirements() -> None:
    report = build_report([
        {"acquisition_status": "resolved", "merged_at": "2025-01-01T00:00:00Z",
         "changed_language_set": ["python"], "changed_files": [{"path": "x.py", "language": "python"}]},
        {"acquisition_status": "resolved", "merged_at": None, "changed_language_set": ["typescript", "unknown"],
         "changed_files": [{"path": "x.ts", "language": "typescript"}, {"path": "README", "language": "unknown"}]},
        {"acquisition_status": "unresolved", "changed_language_set": [], "changed_files": []},
    ], datetime(2026, 6, 1, tzinfo=UTC))
    assert report["total"] == 3
    assert report["successfully_resolved_prs"] == 2
    assert report["unresolved_prs"] == 1
    assert report["mature_prs"] == 1
    assert report["prs_containing_only_currently_supported_python_files"] == 1
    assert report["adapter_languages_required"] == []
    assert "high" not in json.dumps(report).lower()
