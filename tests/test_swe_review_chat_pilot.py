from pathlib import Path

import pytest

from dataset.config import ALL_FEATURES, SUPPORTED_GITHUB_LANGUAGES
from dataset.swe_review_chat_pilot import (
    ANALYSIS_CACHE_VERSION, PilotPaths, assert_feature_contract, cache_entry_is_reusable,
    cache_entry_is_stale_failure, choose_deterministic_replacement, choose_pilot50,
    failure_is_transient, load_language_metadata, transient_retry_allowed,
)


def _row(i: int, label: str, language: str, repo: str | None = None) -> dict:
    repo = repo or f"org/repo-{i}"
    return {"repo_full_name": repo, "pr_number": i, "label": 1 if label == "HIGHER_RISK" else 0,
            "label_name": label, "label_reason": "verified", "provenance": [{"verdict": "approved"}],
            "language": language}


def test_pilot50_exact_balanced_deterministic_and_diverse():
    languages = list(SUPPORTED_GITHUB_LANGUAGES)
    rows = [_row(i, "HIGHER_RISK" if i % 2 else "LOWER_RISK", languages[i % len(languages)]) for i in range(1, 121)]
    metadata = {r["repo_full_name"]: r["language"] for r in rows}
    first = choose_pilot50(rows, metadata)
    second = choose_pilot50(rows, metadata)
    assert [r["identity"] for r in first] == [r["identity"] for r in second]
    assert len(first) == len({r["identity"] for r in first}) == 50
    assert {r["label_name"] for r in first} == {"HIGHER_RISK", "LOWER_RISK"}
    assert set(r["language"] for r in first) == set(languages)


def test_unsupported_and_unknown_are_excluded():
    rows = [_row(i, "HIGHER_RISK", "Python") for i in range(1, 30)] + [_row(100, "UNKNOWN", "Python")]
    with pytest.raises(ValueError):
        choose_pilot50(rows, {r["repo_full_name"]: "Python" for r in rows})


def test_exact_42_features_and_no_label_leakage():
    assert_feature_contract({name: 1.0 for name in ALL_FEATURES})
    with pytest.raises(ValueError):
        assert_feature_contract({**{name: 1.0 for name in ALL_FEATURES}, "label": 1})


def test_old_success_cache_remains_reusable_without_failure_version():
    entry = {"status": "ACCEPTED", "analysis_mode": "full",
             "features": {name: 1.0 for name in ALL_FEATURES}}
    assert cache_entry_is_reusable(entry)


def test_unversioned_and_old_failure_caches_are_stale():
    old = {"status": "CHECKOUT_FAILED"}
    previous = {"status": "CHECKOUT_FAILED", "analysis_cache_version": "older"}
    assert cache_entry_is_stale_failure(old)
    assert cache_entry_is_stale_failure(previous)


def test_current_version_failure_cache_is_reusable():
    entry = {"status": "NO_SUPPORTED_SOURCE_CHANGE", "analysis_cache_version": ANALYSIS_CACHE_VERSION}
    assert cache_entry_is_reusable(entry)
    assert not cache_entry_is_stale_failure(entry)


def test_current_version_transient_failure_is_not_reusable():
    entry = {"status": "CHECKOUT_TIMEOUT", "analysis_cache_version": ANALYSIS_CACHE_VERSION}
    assert failure_is_transient(entry["status"])
    assert not cache_entry_is_reusable(entry)
    assert cache_entry_is_stale_failure(entry)
    assert transient_retry_allowed(entry)
    assert not transient_retry_allowed({**entry, "retry_count": 1})


def test_same_label_replacement_is_deterministic_and_preserves_semantics():
    original = {"identity": "o/r#1", "repo_full_name": "o/r", "language": "Python",
                "label": 1, "label_name": "HIGHER_RISK"}
    candidates = [
        {"identity": "o/r#2", "repo_full_name": "o/r", "language": "Python", "label": 0, "label_name": "LOWER_RISK"},
        {"identity": "o/other#3", "repo_full_name": "o/other", "language": "Python", "label": 1, "label_name": "HIGHER_RISK"},
    ]
    replacement = choose_deterministic_replacement(original, candidates, occupied={"o/r#1"})
    assert replacement["identity"] == "o/other#3"
    assert replacement["label_name"] == original["label_name"]
    assert replacement["label"] == original["label"]


def test_paths_are_isolated(tmp_path: Path):
    paths = PilotPaths(tmp_path / "swe_review_chat")
    paths.prepare()
    assert all(p.exists() for p in (paths.cache, paths.checkpoints, paths.checkouts, paths.pilot))


def test_checkout_identity_uses_distinct_pr_refs(monkeypatch, tmp_path: Path):
    import analyzer.ml.dataset_builder as builder
    calls = []

    def fake_git(args, cwd=None, timeout=120):
        calls.append(args)
        if args[:2] == ["rev-parse", "refs/pilot/pr/77/base"]:
            return True, "base-sha"
        if args[:2] == ["rev-parse", "refs/pilot/pr/77/head"]:
            return True, "head-sha"
        return True, ""

    monkeypatch.setattr(builder, "_run_git", fake_git)
    monkeypatch.setattr(builder, "_checkout_head_commit", lambda *args, **kwargs: (True, ""))
    ok, head, base, error = builder._checkout_pr_head(tmp_path, "fallback", 77, 10)
    assert ok and head == "head-sha" and base == "base-sha" and not error
    assert any("pull/77/head:refs/pilot/pr/77/head" in item for call in calls for item in call)
    assert any("pull/77/base:refs/pilot/pr/77/base" in item for call in calls for item in call)


def test_exact_api_shas_are_preferred_for_closed_pr(monkeypatch, tmp_path: Path):
    import analyzer.ml.dataset_builder as builder
    calls = []

    def fake_git(args, cwd=None, timeout=120):
        calls.append(args)
        return (True, "") if args[0] != "checkout" else (True, "")

    monkeypatch.setattr(builder, "_run_git", fake_git)
    monkeypatch.setattr(builder, "_checkout_head_commit", lambda *args, **kwargs: (True, ""))
    ok, head, base, error = builder._checkout_pr_head(
        tmp_path, "head-sha", 12, 10, base_sha="base-sha",
        head_repo_full_name="owner/repo",
    )
    assert ok and head == "head-sha" and base == "base-sha" and not error
    assert any(call[-2:] == ["origin", "head-sha"] for call in calls)
    assert any(call[-2:] == ["origin", "base-sha"] for call in calls)
    assert not any("pull/12/head" in item for call in calls for item in call)


def test_deleted_fork_head_is_classified_without_fabricating_sha(monkeypatch, tmp_path: Path):
    import analyzer.ml.dataset_builder as builder

    def unavailable(args, cwd=None, timeout=120):
        if args[0] == "fetch":
            return False, "fatal: remote error: upload-pack: not our ref"
        return False, "fatal: reference is not a tree"

    monkeypatch.setattr(builder, "_run_git", unavailable)
    ok, head, _, error = builder._checkout_pr_head(
        tmp_path, "head-sha", 13, 10, base_sha="base-sha",
        head_repo_full_name="deleted-owner/repo",
    )
    assert not ok and head == "head-sha"
    assert error.startswith("HEAD_COMMIT_UNAVAILABLE:")
