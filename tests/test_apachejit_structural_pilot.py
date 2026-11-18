from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.run_apachejit_structural_pilot as pilot
from scripts.run_apachejit_structural_pilot import choose_pilot


def test_choose_pilot_is_development_only_stratified_and_deterministic() -> None:
    rows = []
    for project_index, project in enumerate(("p1", "p2", "p3", "p4")):
        split = "validation" if project_index == 3 else "train"
        for i in range(200):
            rows.append({"project": project, "commit_id": f"{project}-{i}", "buggy": int(i % 5 == 0), "split": split})
    rows.extend({"project": "test-project", "commit_id": f"t-{i}", "buggy": i % 2, "split": "test"} for i in range(100))
    frame = pd.DataFrame(rows)
    first = choose_pilot(frame, projects=("p1", "p2", "p3", "p4"), size=100)
    second = choose_pilot(frame, projects=("p1", "p2", "p3", "p4"), size=100)
    assert first.commit_id.tolist() == second.commit_id.tolist()
    assert len(first) == 100
    assert set(first["split"]) == {"train", "validation"}
    assert "test-project" not in set(first.project)
    assert first.groupby("project")["buggy"].nunique().eq(2).all()
    assert first.groupby("project")["buggy"].mean().eq(.2).all()


def test_unsupported_change_is_explicit_without_materializing_patch(monkeypatch, tmp_path: Path) -> None:
    bare = tmp_path / "repo.git"
    bare.mkdir()
    monkeypatch.setattr("repository.git_object_cache.ensure_bare_repo", lambda *a, **k: (bare, True))
    monkeypatch.setattr("repository.git_object_cache.has_commit", lambda *a, **k: True)
    monkeypatch.setattr(pilot, "_repository_lease", lambda *a, **k: __import__("contextlib").nullcontext())

    def fake_git(args, **kwargs):
        if args[:2] == ["rev-parse", "head^"]:
            return True, "parent"
        if args[:2] == ["cat-file", "-t"]:
            return True, "commit"
        if "--name-only" in args:
            return True, "src/Main.java"
        raise AssertionError(f"textual diff must not be generated for unsupported changes: {args}")

    monkeypatch.setattr(pilot, "_run_git", fake_git)
    row = {"sample_order": 1, "project": "apache/example", "commit_id": "head", "buggy": 0, "split": "train"}
    result = pilot.process_commit(row, cache_dir=tmp_path, analyze_supported=False)
    assert result["checkout_success"] is True
    assert result["analysis_status"] == "analysis_failed"
    assert result["analysis_failure_reason"] == "NO_SUPPORTED_SOURCE_CHANGE (detected: java)"
