"""Focused tests for the bounded Phase 7F checkout optimization."""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import analyzer.ml.dataset_builder as builder
from analyzer.scoped_analysis import build_repository_index, clear_index_cache, select_dependency_neighborhood


def test_checkout_identity_is_short_and_deterministic(monkeypatch, tmp_path) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_git(args, cwd=None, timeout=120):
        calls.append((args, cwd))
        if args[0] == "clone":
            path = tmp_path / "c" / "r" / hashlib.sha256("owner/repository".encode()).hexdigest()[:16]
            (path / ".git").mkdir(parents=True)
        if args[0] == "fetch" and "pull/17/head" in args:
            return True, ""
        if args[0] == "fetch" and "pull/17/base" in args:
            return True, ""
        if args[0] == "rev-parse":
            return True, "a" * 40
        if args[0] == "checkout":
            return True, ""
        return True, ""

    monkeypatch.setattr(builder, "_run_git", fake_git)
    ok, path, _, error = builder._checkout_repo("owner/repository", "b" * 40, tmp_path / "c", pr_number=17)
    assert ok and not error
    assert path is not None
    assert path.name == hashlib.sha256("owner/repository".encode()).hexdigest()[:16]
    assert len(str(path.relative_to(tmp_path))) <= 20
    assert any("--no-checkout" in args for args, _ in calls)
    assert (tmp_path / "c" / "m" / f"{path.name}.json").exists()


def test_repository_clone_is_reused_for_multiple_prs(monkeypatch, tmp_path) -> None:
    clone_calls = 0

    def fake_git(args, cwd=None, timeout=120):
        nonlocal clone_calls
        if args[0] == "clone":
            clone_calls += 1
            Path(cwd or tmp_path)  # keep the fake signature honest
            target = tmp_path / "c" / "r" / hashlib.sha256("owner/repository".encode()).hexdigest()[:16]
            (target / ".git").mkdir(parents=True)
        if args[0] == "fetch" and "pull/17/head" in args:
            return True, ""
        if args[0] == "fetch" and "pull/17/base" in args:
            return True, ""
        if args[0] == "rev-parse":
            return True, "a" * 40
        return True, ""

    monkeypatch.setattr(builder, "_run_git", fake_git)
    for number in (17, 18):
        ok, _, _, error = builder._checkout_repo("owner/repository", "a" * 40, tmp_path / "c", pr_number=number)
        assert ok and not error
    assert clone_calls == 1


def test_checkout_failure_preserves_precise_reason_and_isolated(monkeypatch, tmp_path) -> None:
    def fail_git(args, cwd=None, timeout=120):
        if args[0] == "clone":
            return False, "Filename too long"
        return True, ""

    monkeypatch.setattr(builder, "_run_git", fail_git)
    ok, path, base, error = builder._checkout_repo("owner/repository", "a" * 40, tmp_path / "c")
    assert not ok and path is None and base is None
    assert "clone failed: Filename too long" in error
    assert list((tmp_path / "c" / "r").glob("*")) == []


def test_repository_lease_serializes_same_repo(tmp_path) -> None:
    events: list[str] = []
    entered = threading.Event()

    def first() -> None:
        with builder._repository_lease(tmp_path, "owner/repository", 2):
            events.append("first-enter")
            entered.set()
            time.sleep(0.15)
            events.append("first-exit")

    def second() -> None:
        entered.wait(1)
        with builder._repository_lease(tmp_path, "owner/repository", 2):
            events.append("second-enter")

    a, b = threading.Thread(target=first), threading.Thread(target=second)
    a.start(); b.start(); a.join(); b.join()
    assert events == ["first-enter", "first-exit", "second-enter"]


def test_changed_file_first_transitive_imports_and_dependents(tmp_path) -> None:
    (tmp_path / "app.py").write_text("from service import run\ndef entry(): return run()\n", encoding="utf-8")
    (tmp_path / "service.py").write_text("from base import Base\nclass Service(Base):\n    @decorator\n    def run(self): return helper()\n", encoding="utf-8")
    (tmp_path / "base.py").write_text("class Base: pass\n", encoding="utf-8")
    (tmp_path / "decorators.py").write_text("def decorator(f): return f\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("def helper(): return 1\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("def unrelated(): return 0\n", encoding="utf-8")
    clear_index_cache()
    index = build_repository_index(tmp_path, revision="r1")
    selection = select_dependency_neighborhood(index, ["service.py"])
    assert selection.closure_complete
    assert {p.name for p in selection.files} >= {"service.py", "base.py", "app.py"}
    assert "unrelated.py" not in {p.name for p in selection.files}


def test_closure_limit_is_precise_and_not_repository_size(tmp_path) -> None:
    for i in range(20):
        (tmp_path / f"m{i}.py").write_text(f"from m{i+1} import value\ndef f{i}(): return value\n" if i < 19 else "def value(): return 1\n", encoding="utf-8")
    clear_index_cache()
    selection = select_dependency_neighborhood(build_repository_index(tmp_path, "r2"), ["m0.py"], max_files=3)
    assert not selection.closure_complete
    assert selection.reason == "DEPENDENCY_CLOSURE_LIMIT_EXCEEDED"


def test_repository_index_cache_reuse(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    clear_index_cache()
    first = build_repository_index(tmp_path, "same")
    second = build_repository_index(tmp_path, "same")
    assert not first.cache_hit
    assert second.cache_hit


def test_scoped_full_analysis_keeps_exact_42_features(tmp_path) -> None:
    (tmp_path / "app.py").write_text("from util import helper\ndef run(): return helper() + 1\n", encoding="utf-8")
    (tmp_path / "util.py").write_text("def helper(): return 1\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("def noise(): return 0\n", encoding="utf-8")
    diff = """diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 from util import helper
-def run(): return helper()
+def run(): return helper() + 1
"""
    diagnostics: dict = {}
    features = builder._run_full_analysis(tmp_path, diff, "not-a-real-sha", diagnostics=diagnostics)
    assert features is not None
    assert list(features) == builder.FEATURE_NAMES
    assert len(features) == 42
    assert diagnostics["repository_supported_source_files"] == 3
    assert diagnostics["changed_supported_files"] == 1
    assert diagnostics["parsed_files"] < diagnostics["repository_supported_source_files"]
    assert diagnostics["closure_complete"]
