"""Shared bare Git object caches for historical PR analysis.

The cache owns objects only; PR analysis gets an isolated linked worktree.
This keeps network acquisition repository-scoped and prevents one failed PR
from invalidating a healthy cache.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _safe_sha(sha: str) -> str:
    value = sha.strip().lower()
    if not value or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"invalid commit SHA: {sha!r}")
    return value


def worktree_path(root: Path, full_name: str, pr_number: int | None, role: str, sha: str) -> Path:
    """Return a short, collision-resistant path for one immutable checkout."""
    if role not in {"base", "head"}:
        raise ValueError("worktree role must be base or head")
    key = repository_key(full_name)
    pr = str(pr_number or 0)
    return (root / "worktrees" / key / f"pr-{pr}-{role}-{_safe_sha(sha)[:16]}").resolve()


def _git_result(args: list[str], cwd: Path, timeout: int) -> dict:
    """Run Git with argument arrays and retain stdout/stderr/exit status."""
    command = ["git", "-c", "core.longpaths=true", *args]
    try:
        p = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return {"git_command": command, "cwd": str(cwd), "stdout": p.stdout,
                "stderr": p.stderr, "exit_code": p.returncode}
    except subprocess.TimeoutExpired as exc:
        return {"git_command": command, "cwd": str(cwd), "stdout": exc.stdout or "",
                "stderr": exc.stderr or f"git timeout after {timeout}s", "exit_code": None}
    except Exception as exc:
        return {"git_command": command, "cwd": str(cwd), "stdout": "",
                "stderr": str(exc), "exit_code": None}


def _tail(value: str, limit: int = 4000) -> str:
    return value[-limit:].strip()


def _remove_temporary_worktree(path: Path, root: Path) -> None:
    """Remove only a generated temporary worktree, never an arbitrary path."""
    path = path.resolve()
    root = root.resolve()
    if root not in path.parents:
        raise ValueError(f"refusing to remove worktree outside root: {path}")
    if path.exists():
        shutil.rmtree(path)


def create_worktree(repo_git: Path, root: Path, full_name: str, pr_number: int | None,
                    role: str, sha: str, *, timeout: int = 120,
                    fetch_remote: str | None = None) -> tuple[bool, Path | None, dict]:
    """Create one detached linked worktree, retrying stale/object failures once."""
    sha = _safe_sha(sha)
    target = worktree_path(root, full_name, pr_number, role, sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = {"worktree_role": role, "worktree_path": str(target), "target_sha": sha,
                   "commit_present": False, "retry_attempted": False,
                   "stale_metadata_detected": False}
    present = _git_result(["cat-file", "-e", f"{sha}^{{commit}}"], repo_git, timeout)
    diagnostics["commit_present"] = present["exit_code"] == 0
    if not diagnostics["commit_present"]:
        diagnostics.update(git_command=present["git_command"], exit_code=present["exit_code"],
                           stdout_tail=_tail(present["stdout"]), stderr_tail=_tail(present["stderr"]))
        return False, None, diagnostics

    for attempt in range(2):
        diagnostics["retry_attempted"] = attempt == 1
        if target.exists():
            diagnostics["stale_metadata_detected"] = True
            _remove_temporary_worktree(target, root)
        listing = _git_result(["worktree", "list", "--porcelain"], repo_git, timeout)
        if str(target) in listing["stdout"]:
            diagnostics["stale_metadata_detected"] = True
        if diagnostics["stale_metadata_detected"]:
            _git_result(["worktree", "prune"], repo_git, timeout)
            if attempt == 0:
                continue
        result = _git_result(["worktree", "add", "--detach", str(target), sha], repo_git, timeout)
        if result["exit_code"] == 0:
            diagnostics.update(git_command=result["git_command"], exit_code=0,
                               stdout_tail=_tail(result["stdout"]), stderr_tail=_tail(result["stderr"]))
            return True, target, diagnostics
        # A filtered cache may contain the commit/tree but not checkout blobs.
        if attempt == 0 and fetch_remote:
            _git_result(["fetch", "--no-tags", fetch_remote, sha], repo_git, timeout)
            continue
        diagnostics.update(git_command=result["git_command"], exit_code=result["exit_code"],
                           stdout_tail=_tail(result["stdout"]), stderr_tail=_tail(result["stderr"]))
        return False, None, diagnostics
    return False, None, diagnostics


def _git(args: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[bool, str]:
    # Imported lazily so this utility can be used by the pilot without an
    # import cycle during dataset_builder startup.
    from analyzer.ml.dataset_builder import _run_git
    return _run_git(args, cwd=cwd, timeout=timeout)


def repository_key(full_name: str) -> str:
    return hashlib.sha256(full_name.strip().lower().encode("utf-8")).hexdigest()[:16]


def cache_path(root: Path, full_name: str) -> Path:
    return root / "repos" / repository_key(full_name) / "repo.git"


@contextmanager
def repository_lock(root: Path, full_name: str, timeout: float = 120) -> Iterator[None]:
    lock = root / "repos" / repository_key(full_name) / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("REPOSITORY_LOCK_TIMEOUT")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def valid_bare_repo(path: Path, timeout: int = 30) -> bool:
    if not path.exists():
        return False
    ok, value = _git(["rev-parse", "--is-bare-repository"], cwd=str(path), timeout=timeout)
    return ok and value.strip().lower() == "true"


def ensure_bare_repo(root: Path, full_name: str, *, timeout: int = 120) -> tuple[Path, bool]:
    """Create/reuse a bare cache. Returns (path, cache_hit)."""
    path = cache_path(root, full_name)
    with repository_lock(root, full_name, timeout):
        if valid_bare_repo(path):
            return path, True
        if path.exists():
            # Only an invalid/incomplete object-store entry is repaired.
            for child in path.iterdir():
                if child.is_dir():
                    import shutil
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, err = _git(["init", "--bare", str(path)], timeout=timeout)
        if not ok:
            raise RuntimeError("REPOSITORY_CACHE_CORRUPT: " + err)
        return path, False


def has_commit(repo_git: Path, sha: str, timeout: int = 30) -> bool:
    ok, _ = _git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=str(repo_git), timeout=timeout)
    return ok


def fetch(repo_git: Path, remote: str, refspec: str, *, timeout: int = 120, depth: int | None = 2,
          filter_blobs: bool = True) -> tuple[bool, str]:
    args = ["fetch", "--no-tags"]
    if depth:
        args += ["--depth", str(depth)]
    if filter_blobs:
        args += ["--filter=blob:none"]
    args += [remote, refspec]
    ok, output = _git(args, cwd=str(repo_git), timeout=timeout)
    if not ok and filter_blobs and ("filter" in output.lower() or "unsupported" in output.lower()):
        fallback = [a for a in args if a != "--filter=blob:none"]
        return _git(fallback, cwd=str(repo_git), timeout=timeout)
    return ok, output


def prepare_pr_objects(root: Path, full_name: str, head_sha: str, base_sha: str, *,
                       pr_number: int | None = None, timeout: int = 120) -> tuple[bool, Path | None, str]:
    """Ensure exact PR objects exist in the bare cache without creating a worktree.

    This is the cheap boundary used to discover changed paths.  It intentionally
    holds the repository lock only while the object store is changed, leaving
    parsing and graph analysis free to run concurrently.
    """
    origin = f"https://github.com/{full_name}.git"
    try:
        # The caller owns the repository mutation lease.  ``ensure_bare_repo``
        # has its own short initialization lock; fetches remain protected by
        # that outer lease and never cover semantic analysis.
        bare, _ = ensure_bare_repo(root, full_name, timeout=timeout)
        for sha, fallback_ref, label in (
            (head_sha, f"refs/pull/{pr_number}/head:refs/pilot/{pr_number}/head" if pr_number else None, "HEAD"),
            (base_sha, None, "BASE"),
        ):
            if has_commit(bare, sha, timeout=30):
                continue
            ok, error = fetch(bare, origin, sha, timeout=timeout, depth=2)
            if not ok and fallback_ref:
                ok, error = fetch(bare, origin, fallback_ref, timeout=timeout, depth=2)
            if not ok:
                return False, None, f"{label}_COMMIT_UNAVAILABLE: {error or 'exact commit could not be fetched'}"
        return True, bare, ""
    except TimeoutError:
        return False, None, "REPOSITORY_LOCK_TIMEOUT"
    except Exception as exc:
        return False, None, f"REPOSITORY_CACHE_CORRUPT: {exc}"
