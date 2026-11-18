from pathlib import Path
import subprocess

import pytest

from repository import git_object_cache as cache


def git(*args, cwd: Path) -> str:
    p = subprocess.run(["git", *args], cwd=str(cwd), text=True,
                       capture_output=True, check=True)
    return p.stdout.strip()


def make_bare(tmp_path: Path):
    work = tmp_path / "normal"
    work.mkdir()
    git("init", cwd=work)
    git("config", "user.email", "test@example.invalid", cwd=work)
    git("config", "user.name", "Test", cwd=work)
    (work / "value.txt").write_text("base\n")
    git("add", ".", cwd=work)
    git("commit", "-m", "base", cwd=work)
    base = git("rev-parse", "HEAD", cwd=work)
    (work / "value.txt").write_text("head\n")
    git("commit", "-am", "head", cwd=work)
    head = git("rev-parse", "HEAD", cwd=work)
    bare = tmp_path / "cache.git"
    git("clone", "--bare", str(work), str(bare), cwd=tmp_path)
    return work, bare, base, head


def test_valid_bare_repo_and_detached_worktree(tmp_path):
    _, bare, base, head = make_bare(tmp_path)
    ok, path, diag = cache.create_worktree(bare, tmp_path, "owner/repo", 1, "head", head)
    assert ok and path and diag["commit_present"]
    detached = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=str(path),
                              capture_output=True, text=True)
    assert detached.returncode != 0 and detached.stdout == ""
    assert (path / "value.txt").read_text() == "head\n"


def test_separate_base_head_worktrees_and_contents(tmp_path):
    _, bare, base, head = make_bare(tmp_path)
    a = cache.create_worktree(bare, tmp_path, "owner/repo", 2, "base", base)[1]
    b = cache.create_worktree(bare, tmp_path, "owner/repo", 2, "head", head)[1]
    assert a != b and (a / "value.txt").read_text() == "base\n" and (b / "value.txt").read_text() == "head\n"
    git("worktree", "remove", "--force", str(a), cwd=bare)
    git("worktree", "remove", "--force", str(b), cwd=bare)
    git("worktree", "prune", cwd=bare)


def test_existing_empty_directory_is_replaced(tmp_path):
    _, bare, _, head = make_bare(tmp_path)
    path = cache.worktree_path(tmp_path, "owner/repo", 3, "head", head)
    path.mkdir(parents=True)
    ok, _, _ = cache.create_worktree(bare, tmp_path, "owner/repo", 3, "head", head)
    assert ok


def test_existing_nonempty_directory_is_replaced(tmp_path):
    _, bare, _, head = make_bare(tmp_path)
    path = cache.worktree_path(tmp_path, "owner/repo", 4, "head", head)
    path.mkdir(parents=True)
    (path / "sentinel").write_text("temporary")
    ok, _, diag = cache.create_worktree(bare, tmp_path, "owner/repo", 4, "head", head)
    assert ok and diag["stale_metadata_detected"]


def test_missing_commit_has_exact_diagnostic(tmp_path):
    _, bare, _, _ = make_bare(tmp_path)
    ok, _, diag = cache.create_worktree(bare, tmp_path, "owner/repo", 5, "head", "0" * 40)
    assert not ok and not diag["commit_present"] and diag["worktree_role"] == "head"
    assert "cat-file" in " ".join(diag["git_command"])


def test_windows_style_paths_are_path_objects(tmp_path):
    _, bare, _, head = make_bare(tmp_path)
    path = cache.worktree_path(tmp_path, "owner/repo", 6, "head", head)
    assert isinstance(path, Path) and len(str(path)) < 240


def test_retry_succeeds_after_stale_metadata(monkeypatch, tmp_path):
    _, bare, _, head = make_bare(tmp_path)
    path = cache.worktree_path(tmp_path, "owner/repo", 7, "head", head)
    path.mkdir(parents=True)
    calls = []
    real = cache._git_result
    def wrapped(args, cwd, timeout):
        calls.append(args)
        return real(args, cwd, timeout)
    monkeypatch.setattr(cache, "_git_result", wrapped)
    ok, _, diag = cache.create_worktree(bare, tmp_path, "owner/repo", 7, "head", head)
    assert ok and diag["retry_attempted"] and any(a[:2] == ["worktree", "prune"] for a in calls)


def test_retry_failure_preserves_worktree_diagnostics(monkeypatch, tmp_path):
    bare = tmp_path / "repo.git"
    bare.mkdir()
    n = {"add": 0}
    def fake(args, cwd, timeout):
        if args[:2] == ["cat-file", "-e"]:
            return {"git_command": ["git", *args], "cwd": str(cwd), "stdout": "", "stderr": "", "exit_code": 0}
        if args[:2] == ["worktree", "list"]:
            return {"git_command": ["git", *args], "cwd": str(cwd), "stdout": "", "stderr": "", "exit_code": 0}
        if args[:2] == ["worktree", "add"]:
            n["add"] += 1
            return {"git_command": ["git", *args], "cwd": str(cwd), "stdout": "out", "stderr": "fatal: checkout failed", "exit_code": 128}
        return {"git_command": ["git", *args], "cwd": str(cwd), "stdout": "", "stderr": "", "exit_code": 0}
    monkeypatch.setattr(cache, "_git_result", fake)
    ok, _, diag = cache.create_worktree(bare, tmp_path, "owner/repo", 8, "head", "a" * 40, fetch_remote="origin")
    assert not ok and n["add"] == 2 and diag["retry_attempted"] and diag["stderr_tail"] == "fatal: checkout failed"


def test_partial_object_retry_succeeds(monkeypatch, tmp_path):
    bare = tmp_path / "repo.git"
    bare.mkdir()
    n = {"add": 0}
    def fake(args, cwd, timeout):
        if args[:2] == ["cat-file", "-e"] or args[:2] == ["worktree", "list"]:
            return {"git_command": ["git", *args], "cwd": str(cwd), "stdout": "", "stderr": "", "exit_code": 0}
        if args[:2] == ["worktree", "add"]:
            n["add"] += 1
            return {"git_command": ["git", *args], "cwd": str(cwd), "stdout": "ok" if n["add"] == 2 else "", "stderr": "missing blob" if n["add"] == 1 else "", "exit_code": 0 if n["add"] == 2 else 128}
        return {"git_command": ["git", *args], "cwd": str(cwd), "stdout": "", "stderr": "", "exit_code": 0}
    monkeypatch.setattr(cache, "_git_result", fake)
    ok, _, diag = cache.create_worktree(bare, tmp_path, "owner/repo", 9, "head", "b" * 40, fetch_remote="origin")
    assert ok and diag["retry_attempted"] and n["add"] == 2
