"""Focused behavioural tests for the cheap-first Phase 7F expansion path."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime

import polars as pl

import dataset.phase7f_expansion as expansion
from analyzer.ml.dataset_builder import FEATURE_NAMES, FeatureRow


def _pr(identifier: int, repo_id: int = 1, number: int | None = None) -> dict[str, object]:
    return {"id": identifier, "repo_id": repo_id, "number": number or identifier, "merged_at": "2025-01-01T00:00:00Z"}


def _full_row(identifier: int, repo: str = "acme/repo") -> dict[str, object]:
    row = FeatureRow(identifier, 1, repo, identifier, "2025-01-01T00:00:00Z", None, 0)
    for name in FEATURE_NAMES:
        setattr(row, name, 1.0)
    row.analysis_mode = "full"
    return asdict(row)


class _DiscoveryBuilder:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def _load_eligible_repos(self) -> pl.DataFrame:
        return pl.DataFrame({"id": [1], "full_name": ["acme/repo"]})

    def _load_eligible_prs(self, _: list[int]) -> pl.DataFrame:
        return pl.DataFrame([_pr(1), _pr(2), _pr(3), _pr(4)])

    def _load_patches(self, _: list[int]) -> pl.DataFrame:
        return pl.DataFrame({
            "pr_id": [1, 2, 3, 4], "filename": ["src/a.py", "src/a.py", "README", "src/a.py"],
            "status": ["modified"] * 4, "patch": ["@@ -1 +1 @@\n-a\n+b", "@@ -1 +1 @@\n-a\n+b", "@@ -1 +1 @@\n-a\n+b", "@@ -1 +1 @@\n-a\n+b"],
        })

    def _load_commits(self, _: list[int]) -> pl.DataFrame:
        return pl.DataFrame({"pr_id": [1, 2, 3, 4], "sha": ["a", "b", "c", "d"]})


def test_discovery_uses_cheap_gates_before_expensive_work(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(expansion, "AIDevDatasetBuilder", _DiscoveryBuilder)
    expander = expansion.CheapFirstExpander(output_dir=tmp_path, exclude_pr_ids={1}, cutoff_at=datetime.now(UTC), client=None)
    candidates = expander.discover(4)
    # Discovery order is the deterministic seeded candidate order, rather
    # than ascending pull-request ID order.
    assert [item["pr"]["id"] for item in candidates] == [4, 2]
    assert expander.stats["baseline_reject"] == 1
    assert expander.stats["unknown_language_reject"] == 1
    assert expander.stats["entering_expensive"] == 0
    assert expander.outcomes(candidates)[0] == []
    assert expander.stats["outcome_unknown_reject"] == 2


def test_versioned_cache_reuses_only_matching_version(tmp_path) -> None:
    first = expansion.VersionedCache(tmp_path, "feature", "v1")
    first.put({"pr": 7}, {"value": 1})
    assert first.get({"pr": 7}) == {"value": 1}
    assert first.hits == 1
    changed = expansion.VersionedCache(tmp_path, "feature", "v2")
    assert changed.get({"pr": 7}) is None
    assert changed.misses == 1


def test_cached_analysis_is_resumable_and_deterministic(monkeypatch, tmp_path) -> None:
    class _NeverAnalyze:
        def __init__(self, **_: object) -> None: pass
        def _process_pr(self, *_: object) -> object: raise AssertionError("cache should avoid analysis")
    monkeypatch.setattr(expansion, "AIDevDatasetBuilder", _NeverAnalyze)
    expander = expansion.CheapFirstExpander(output_dir=tmp_path, exclude_pr_ids=set(), cutoff_at=datetime.now(UTC), client=None, workers=2)
    cache = expansion.VersionedCache(tmp_path / "cache", "features", expansion.ANALYSIS_VERSION)
    for identifier in (2, 1):
        cache.put({"repo": "acme/repo", "pr": identifier, "head": str(identifier), "analysis": expansion.ANALYSIS_VERSION}, _full_row(identifier))
    candidates = [{"pr": _pr(identifier), "repo": "acme/repo", "patch": "patch", "head_sha": str(identifier), "languages": ["python"]} for identifier in (2, 1)]
    rows = expander.analyze(candidates, 2)
    assert [row["pr_id"] for row in rows] == [1, 2]
    assert expander.stats["feature_cache_hits"] == 2
    assert expander.stats["entering_expensive"] == 0


def test_analysis_workers_are_bounded(monkeypatch, tmp_path) -> None:
    observed: list[int] = []
    original = expansion.ThreadPoolExecutor
    class _Builder:
        def __init__(self, **_: object) -> None: pass
        def _process_pr(self, pr: dict[str, object], repo: str, *_: object) -> FeatureRow:
            row = FeatureRow(int(pr["id"]), 1, repo, int(pr["number"]), "", None, 0)
            for name in FEATURE_NAMES: setattr(row, name, 1.0)
            row.analysis_mode = "full"
            return row
    class _Executor(original):
        def __init__(self, *args: object, **kwargs: object) -> None:
            observed.append(int(kwargs["max_workers"])); super().__init__(*args, **kwargs)
    monkeypatch.setattr(expansion, "AIDevDatasetBuilder", _Builder)
    monkeypatch.setattr(expansion, "ThreadPoolExecutor", _Executor)
    expander = expansion.CheapFirstExpander(output_dir=tmp_path, exclude_pr_ids=set(), cutoff_at=datetime.now(UTC), client=None, workers=2)
    candidates = [{"pr": _pr(i), "repo": "acme/repo", "patch": "patch", "head_sha": str(i), "languages": ["typescript"]} for i in range(1, 5)]
    assert len(expander.analyze(candidates, 4)) == 4
    assert observed == [2]


def test_outcomes_are_batched_checkpointed_and_resumable(monkeypatch, tmp_path) -> None:
    calls: list[int] = []

    class _Client: pass
    class _Acquirer:
        def __init__(self, *_: object, **__: object) -> None: pass
        def acquire_one(self, identity: expansion.PRIdentity) -> dict[str, object]:
            calls.append(identity.pr_number)
            return {"sample_id": identity.sample_id, "repo_full_name": identity.repo_full_name, "pr_number": identity.pr_number,
                    "acquisition_status": "resolved", "merged_at": "2025-01-01T00:00:00Z", "merge_commit_sha": "m",
                    "final_pr_head_sha": "h", "changed_files": [{"path": "src/a.py"}]}

    def _labels(*, ledger_path, output_dir, labels_name, **_: object) -> dict[str, object]:
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
        (output_dir / labels_name).write_text("".join(json.dumps({"sample_id": row["sample_id"], "risk_label": "low_observed"}) + "\n" for row in rows))
        return {}

    monkeypatch.setattr(expansion, "IdentityAcquirer", _Acquirer)
    monkeypatch.setattr(expansion, "build_outcome_artifacts", _labels)
    candidates = [{"pr": _pr(i), "repo": "acme/repo", "repo_id": 1, "patch": "patch", "head_sha": str(i), "languages": ["python"]} for i in range(1, 6)]
    expander = expansion.CheapFirstExpander(output_dir=tmp_path, exclude_pr_ids=set(), cutoff_at=datetime.now(UTC), client=_Client(), workers=2, batch_size=2)
    eligible, _, labels_path = expander.outcomes(candidates)
    assert len(eligible) == 5
    assert calls == [1, 2, 3, 4, 5]
    assert expander.stats["identity_batches_completed"] == 3
    assert expander.stats["outcome_batches_completed"] == 3
    checkpoint = json.loads((tmp_path / "phase7f_checkpoint.json").read_text())
    assert checkpoint["stage"] == "outcomes" and checkpoint["completed"] == 5
    assert labels_path.exists()

    calls.clear()
    resumed = expansion.CheapFirstExpander(output_dir=tmp_path, exclude_pr_ids=set(), cutoff_at=datetime.now(UTC), client=_Client(), workers=2, batch_size=2)
    eligible, _, _ = resumed.outcomes(candidates)
    assert len(eligible) == 5
    assert calls == []


def test_supported_language_routing_has_no_python_fallback() -> None:
    expected = {"python", "javascript", "typescript", "csharp", "rust", "cpp"}
    assert set(expansion.SUPPORTED_CHANGED_LANGUAGES) == expected
    detector = expansion.LanguageDetector()
    assert {detector.detect(f"src/file{suffix}").language for suffix in (".py", ".js", ".ts", ".cs", ".rs", ".cpp")} == expected
    assert detector.detect("README").language == "unknown"
