"""
analyzer/ml/dataset_builder.py — Phase 8B AIDev Dataset Ingestion Pipeline

Pipeline:
    AIDev PR → filter → sample → per-PR:
        → patch extraction
        → label assignment
        → [optional] repository checkout
        → analyzer (change analysis + impact scoring)
        → 42 feature vector
        → Parquet row

DESIGN RULES (NON-NEGOTIABLE):
1. No synthetic data. Every row comes from a real AIDev PR.
2. If checkout fails → record failure, skip Cat-C features (NaN), do NOT zero-fill.
3. Leakage fields NEVER enter the feature vector.
4. Polars lazy scanning — never load full 1.3 GB table.
5. Sequential, bounded repository processing (max N repos on disk).
6. Deterministic sampling with seed=42.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for optional ML deps
# ---------------------------------------------------------------------------

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False
    pl = None  # type: ignore[assignment]

try:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False
    pa = None
    pq = None

# ---------------------------------------------------------------------------
# Feature name list (canonical 42-feature order from dataset/config.py)
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # Cat A
    "files_changed", "lines_added", "lines_deleted", "total_lines_changed",
    # Cat B
    "functions_changed", "classes_changed", "methods_changed",
    "dependencies_added", "dependencies_removed", "dependencies_changed",
    # Cat B/C
    "dependency_change_ratio",
    # Cat C
    "direct_nodes_changed", "impacted_nodes", "impacted_modules",
    "impacted_functions", "impacted_classes", "impacted_methods",
    "maximum_impact_depth", "impact_ratio", "average_impact_depth",
    "changed_modules", "cross_module_impact_count", "cross_module_ratio",
    "average_changed_node_degree", "maximum_changed_node_degree",
    "average_changed_node_dependents", "maximum_changed_node_dependents",
    "average_changed_node_dependencies", "maximum_changed_node_dependencies",
    "import_propagation_count", "from_import_propagation_count",
    "call_propagation_count", "inheritance_propagation_count",
    "decoration_propagation_count", "total_propagation_edges",
    "critical_node_count", "nodes_at_depth_0", "nodes_at_depth_1",
    "nodes_at_depth_2", "nodes_at_depth_3_plus",
    "impact_score", "impact_level_numeric",
]

assert len(FEATURE_NAMES) == 42

# Cat C feature names (set for fast lookup)
_CAT_C_FEATURES: Set[str] = {
    "direct_nodes_changed", "impacted_nodes", "impacted_modules",
    "impacted_functions", "impacted_classes", "impacted_methods",
    "maximum_impact_depth", "impact_ratio", "average_impact_depth",
    "changed_modules", "cross_module_impact_count", "cross_module_ratio",
    "average_changed_node_degree", "maximum_changed_node_degree",
    "average_changed_node_dependents", "maximum_changed_node_dependents",
    "average_changed_node_dependencies", "maximum_changed_node_dependencies",
    "import_propagation_count", "from_import_propagation_count",
    "call_propagation_count", "inheritance_propagation_count",
    "decoration_propagation_count", "total_propagation_edges",
    "critical_node_count", "nodes_at_depth_0", "nodes_at_depth_1",
    "nodes_at_depth_2", "nodes_at_depth_3_plus",
    "impact_score", "impact_level_numeric",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FilterCounters:
    """Tracks how many PRs were filtered at each stage."""

    considered: int = 0
    excluded_language: int = 0
    excluded_license: int = 0
    excluded_fork: int = 0
    excluded_no_patch: int = 0
    excluded_no_review: int = 0
    excluded_not_closed: int = 0
    excluded_not_merged: int = 0
    excluded_revert: int = 0
    checkout_failed: int = 0
    analysis_failed: int = 0
    successfully_extracted: int = 0

    def summary(self) -> str:
        return (
            f"FilterCounters:\n"
            f"  Considered       : {self.considered}\n"
            f"  -language        : {self.excluded_language}\n"
            f"  -license         : {self.excluded_license}\n"
            f"  -fork            : {self.excluded_fork}\n"
            f"  -no_patch        : {self.excluded_no_patch}\n"
            f"  -no_review       : {self.excluded_no_review}\n"
            f"  -not_closed      : {self.excluded_not_closed}\n"
            f"  -not_merged      : {self.excluded_not_merged}\n"
            f"  -revert          : {self.excluded_revert}\n"
            f"  checkout_failed  : {self.checkout_failed}\n"
            f"  analysis_failed  : {self.analysis_failed}\n"
            f"  Extracted        : {self.successfully_extracted}"
        )


@dataclass
class FeatureRow:
    """One row of the ML dataset."""

    pr_id: int
    repo_id: int
    repo_full_name: str
    pr_number: int
    created_at: str
    agent: Optional[str]
    high_risk: int

    # 42 features — NaN = analysis failure (Cat C) or unavailable
    files_changed: float = float("nan")
    lines_added: float = float("nan")
    lines_deleted: float = float("nan")
    total_lines_changed: float = float("nan")
    functions_changed: float = float("nan")
    classes_changed: float = float("nan")
    methods_changed: float = float("nan")
    dependencies_added: float = float("nan")
    dependencies_removed: float = float("nan")
    dependencies_changed: float = float("nan")
    dependency_change_ratio: float = float("nan")
    direct_nodes_changed: float = float("nan")
    impacted_nodes: float = float("nan")
    impacted_modules: float = float("nan")
    impacted_functions: float = float("nan")
    impacted_classes: float = float("nan")
    impacted_methods: float = float("nan")
    maximum_impact_depth: float = float("nan")
    impact_ratio: float = float("nan")
    average_impact_depth: float = float("nan")
    changed_modules: float = float("nan")
    cross_module_impact_count: float = float("nan")
    cross_module_ratio: float = float("nan")
    average_changed_node_degree: float = float("nan")
    maximum_changed_node_degree: float = float("nan")
    average_changed_node_dependents: float = float("nan")
    maximum_changed_node_dependents: float = float("nan")
    average_changed_node_dependencies: float = float("nan")
    maximum_changed_node_dependencies: float = float("nan")
    import_propagation_count: float = float("nan")
    from_import_propagation_count: float = float("nan")
    call_propagation_count: float = float("nan")
    inheritance_propagation_count: float = float("nan")
    decoration_propagation_count: float = float("nan")
    total_propagation_edges: float = float("nan")
    critical_node_count: float = float("nan")
    nodes_at_depth_0: float = float("nan")
    nodes_at_depth_1: float = float("nan")
    nodes_at_depth_2: float = float("nan")
    nodes_at_depth_3_plus: float = float("nan")
    impact_score: float = float("nan")
    impact_level_numeric: float = float("nan")

    # Extraction metadata
    checkout_success: bool = False
    analysis_mode: str = "none"   # "full" | "patch_only" | "none"
    extraction_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def feature_dict(self) -> Dict[str, Any]:
        """Return only the ML feature columns + metadata (no leakage fields)."""
        d = self.to_dict()
        # Remove leakage fields
        for lf in ("high_risk",):
            pass  # high_risk is the label, included separately
        return d


# ---------------------------------------------------------------------------
# Patch parsing utilities
# ---------------------------------------------------------------------------

_FUNC_DEF_RE = re.compile(
    r"^[+\-]\s*def\s+\w+", re.MULTILINE
)
_CLASS_DEF_RE = re.compile(
    r"^[+\-]\s*class\s+\w+", re.MULTILINE
)
_IMPORT_ADD_RE = re.compile(
    r"^\+\s*(?:import|from)\s+", re.MULTILINE
)
_IMPORT_DEL_RE = re.compile(
    r"^-\s*(?:import|from)\s+", re.MULTILINE
)


def _extract_patch_features(patch_text: str) -> Dict[str, float]:
    """
    Extract Category B features from a raw unified-diff patch.

    Uses our existing analyzer where possible; falls back to regex counts.

    Args:
        patch_text: Concatenated patch for all files in a PR.

    Returns:
        Dict with Cat B features.
    """
    # Use our Phase 5 analyzer for accurate results
    try:
        from analyzer.change_analyzer import analyze_diff
        cs = analyze_diff(patch_text)
        deps_added = len(cs.dependencies_added)
        deps_removed = len(cs.dependencies_removed)

        # When analyze_diff returns 0 dep changes (happens in patch-only context
        # where full file AST is unavailable), augment with regex import counts.
        # Regex catches "+import X" / "+from X import Y" lines in the diff.
        if deps_added == 0 and deps_removed == 0:
            deps_added = len(_IMPORT_ADD_RE.findall(patch_text))
            deps_removed = len(_IMPORT_DEL_RE.findall(patch_text))

        deps_changed = deps_added + deps_removed
        direct_nodes = cs.files_changed_count
        return {
            "files_changed": float(cs.files_changed_count),
            "lines_added": float(cs.lines_added),
            "lines_deleted": float(cs.lines_deleted),
            "total_lines_changed": float(cs.total_lines_changed),
            "functions_changed": float(len(cs.changed_functions)),
            "classes_changed": float(len(cs.changed_classes)),
            "methods_changed": float(len(cs.changed_methods)),
            "dependencies_added": float(deps_added),
            "dependencies_removed": float(deps_removed),
            "dependencies_changed": float(deps_changed),
            "dependency_change_ratio": deps_changed / max(direct_nodes + 1, 1),
        }
    except Exception as exc:
        logger.debug(f"Patch analyzer failed ({exc}); falling back to regex.")

    # Regex fallback (less accurate but never fails silently)
    lines = patch_text.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    deleted = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    funcs = len(_FUNC_DEF_RE.findall(patch_text))
    classes = len(_CLASS_DEF_RE.findall(patch_text))
    imports_added = len(_IMPORT_ADD_RE.findall(patch_text))
    imports_removed = len(_IMPORT_DEL_RE.findall(patch_text))

    # Estimate files from diff headers
    file_headers = [l for l in lines if l.startswith("diff --git")]
    n_files = max(len(file_headers), 1)

    return {
        "files_changed": float(n_files),
        "lines_added": float(added),
        "lines_deleted": float(deleted),
        "total_lines_changed": float(added + deleted),
        "functions_changed": float(funcs),
        "classes_changed": float(classes),
        "methods_changed": 0.0,
        "dependencies_added": float(imports_added),
        "dependencies_removed": float(imports_removed),
        "dependencies_changed": float(imports_added + imports_removed),
        "dependency_change_ratio": (imports_added + imports_removed) / max(n_files + 1, 1),
    }



# ---------------------------------------------------------------------------
# Repository checkout utilities
# ---------------------------------------------------------------------------


def _normalize_license_standalone(raw: Optional[str]) -> Optional[str]:
    """Module-level license normalization (for testing without instantiating builder)."""
    if not raw:
        return None
    raw = raw.strip().lower()
    aliases = {
        "mit license": "mit",
        "apache license 2.0": "apache-2.0",
        "apache 2.0": "apache-2.0",
        "apache-2": "apache-2.0",
        "bsd 2-clause": "bsd-2-clause",
        "bsd 3-clause": "bsd-3-clause",
        "isc license": "isc",
    }
    return aliases.get(raw, raw)


def _run_git(args: List[str], cwd: Optional[str] = None, timeout: int = 120) -> Tuple[bool, str]:
    """Run a git command, returning (success, stderr_or_stdout)."""
    try:
        result = subprocess.run(
            # Keep long-path support local to this invocation.  This avoids
            # requiring unsafe machine-wide Git configuration on Windows.
            ["git", "-c", "core.longpaths=true"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if result.returncode != 0:
            # Git can omit stderr for transport or binary-diff failures.
            # Preserve a meaningful analysis failure instead of raising while
            # formatting the error and obscuring the pipeline stage.
            message = (result.stderr or result.stdout or "git command failed").strip()
            return False, message
        return True, (result.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return False, f"git timeout after {timeout}s"
    except FileNotFoundError:
        return False, "git not found in PATH"
    except Exception as exc:
        return False, str(exc)


def _checkout_repo(
    repo_full_name: str,
    commit_sha: str,
    base_dir: Path,
    pr_number: Optional[int] = None,
    clone_depth: int = 50,
    timeout: int = 120,
    max_size_mb: int = 500,
    base_sha: Optional[str] = None,
    head_repo_full_name: Optional[str] = None,
) -> Tuple[bool, Optional[Path], Optional[str], str]:
    """
    Clone a GitHub repository and checkout the PR head commit.

    Args:
        repo_full_name: "owner/repo" string.
        commit_sha: First commit SHA of the PR (head commit).
        base_dir: Directory to clone into.
        clone_depth: git clone --depth value.
        timeout: git operation timeout in seconds.
        max_size_mb: Skip repos larger than this.

    Returns:
        (success, repo_path_or_None, base_sha_or_None, error_message)
        base_sha is the PR base commit from refs/pull/<number>/base when available.
    """
    # Production path: acquire objects once per repository into a bare cache,
    # then create a local linked worktree with no network access.  The legacy
    # branch below remains as a compatibility fallback for callers/tests that
    # provide a non-Git fake runner.
    try:
        from repository.git_object_cache import create_worktree, ensure_bare_repo, fetch, has_commit
        bare, cache_hit = ensure_bare_repo(base_dir, repo_full_name, timeout=timeout)
        if not bare.exists():
            raise RuntimeError("cache initializer did not create a repository")
        origin = f"https://github.com/{repo_full_name}.git"
        if not has_commit(bare, commit_sha, timeout=30):
            ok, err = fetch(bare, origin, commit_sha, timeout=timeout, depth=2)
            if not ok and pr_number:
                ok, err = fetch(bare, origin, f"refs/pull/{pr_number}/head:refs/pilot/{pr_number}/head", timeout=timeout, depth=2)
            if not ok:
                return False, None, None, "HEAD_COMMIT_UNAVAILABLE: " + (err or "exact commit could not be fetched")
        if base_sha and not has_commit(bare, base_sha, timeout=30):
            ok, err = fetch(bare, origin, base_sha, timeout=timeout, depth=2)
            if not ok:
                logger.debug("base exact fetch failed: %s", err)
        # Materialize immutable, detached, role-specific worktrees directly
        # from the bare cache. Never reuse one directory for base and head.
        remote = f"https://github.com/{repo_full_name}.git"
        ok, linked_dir, head_diag = create_worktree(
            bare, base_dir, repo_full_name, pr_number, "head", commit_sha,
            timeout=timeout, fetch_remote=remote,
        )
        if not ok:
            return False, None, None, "WORKTREE_FAILED: " + json.dumps(head_diag, sort_keys=True)
        if base_sha:
            ok_base, _, base_diag = create_worktree(
                bare, base_dir, repo_full_name, pr_number, "base", base_sha,
                timeout=timeout, fetch_remote=remote,
            )
            if not ok_base:
                return False, None, None, "WORKTREE_FAILED: " + json.dumps(base_diag, sort_keys=True)
        return True, linked_dir, base_sha, ""
    except TimeoutError:
        return False, None, None, "REPOSITORY_LOCK_TIMEOUT"
    except Exception as exc:
        logger.debug("bare cache path unavailable; using compatibility checkout: %s", exc)

    # The checkout root is intentionally short.  The repository name is kept
    # in the deterministic metadata file and in the per-PR feature cache key,
    # rather than being repeated in the filesystem path.
    repo_key = hashlib.sha256(repo_full_name.strip().lower().encode("utf-8")).hexdigest()[:16]
    repo_dir = base_dir / "r" / repo_key
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata_dir = base_dir / "m"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{repo_key}.json"
    if not metadata_path.exists():
        temp_metadata = metadata_dir / f".{repo_key}.{os.getpid()}.{threading.get_ident()}.tmp"
        temp_metadata.write_text(json.dumps({"repo": repo_full_name, "repo_key": repo_key}, sort_keys=True), encoding="utf-8")
        try:
            temp_metadata.replace(metadata_path)
        except FileExistsError:
            temp_metadata.unlink(missing_ok=True)

    # If already cloned, reuse
    cached_repo_valid = False
    if repo_dir.exists() and (repo_dir / ".git").exists():
        cached_repo_valid, _ = _run_git(
            ["rev-parse", "--git-dir"], cwd=str(repo_dir), timeout=timeout
        )
    if cached_repo_valid:
        logger.debug(f"Reusing cached clone: {repo_dir}")
        ok, checked_out_head, fetched_base, checkout_error = _checkout_pr_head(
            repo_dir, commit_sha, pr_number, timeout, base_sha=base_sha,
            head_repo_full_name=head_repo_full_name,
        )
        if ok:
            return True, repo_dir, fetched_base, ""
        # Preserve a valid object store/checkout for later repair.  A failed
        # historical acquisition is not evidence that the repository cache is
        # corrupt, and deleting it causes the expensive clone loop we are
        # eliminating.
        logger.info("Cached checkout could not acquire this commit: %s", checkout_error or "unavailable")
        return False, None, None, checkout_error or "HEAD_COMMIT_UNAVAILABLE"

    # Remove only this deterministic, incomplete cache entry.  This makes a
    # prior interrupted clone retryable without touching other repositories.
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)

    clone_url = f"https://github.com/{repo_full_name}.git"
    logger.info(f"Cloning {clone_url} (depth={clone_depth})")

    ok, err = _run_git(
        ["clone", "--no-checkout", "--depth", str(clone_depth), "--no-single-branch",
         clone_url, str(repo_dir)],
        timeout=timeout,
    )
    if not ok:
        return False, None, None, f"clone failed: {err}"

    # AIDev commits are often older than the shallow default branch history.
    # Fetch the exact commit before attempting checkout instead of assuming a
    # depth-limited clone happens to contain it.
    ok, checked_out_head, fetched_base, checkout_error = _checkout_pr_head(
        repo_dir, commit_sha, pr_number, timeout, base_sha=base_sha,
        head_repo_full_name=head_repo_full_name,
    )
    if not ok:
        shutil.rmtree(repo_dir, ignore_errors=True)
        return False, None, None, f"checkout failed: {checkout_error or 'PR ref unavailable'}"

    return True, repo_dir, fetched_base, ""


_REPO_LOCKS: Dict[str, threading.Lock] = {}
_REPO_LOCKS_GUARD = threading.Lock()


@contextmanager
def _repository_lease(repo_cache_dir: Path, repo_full_name: str, timeout: int):
    """Serialize checkout and analysis for one repo across threads/processes."""
    key = hashlib.sha256(repo_full_name.strip().lower().encode("utf-8")).hexdigest()[:16]
    with _REPO_LOCKS_GUARD:
        lock = _REPO_LOCKS.setdefault(str(repo_cache_dir.resolve()) + ":" + key, threading.Lock())
    lock_path = repo_cache_dir / "l" / f"{key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1, timeout)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"repository_lock_timeout:{repo_full_name}")
            time.sleep(0.05)
    try:
        acquired = lock.acquire(timeout=max(1, int(deadline - time.monotonic())))
        if not acquired:
            raise TimeoutError(f"repository_lock_timeout:{repo_full_name}")
        try:
            yield
        finally:
            lock.release()
    finally:
        Path(lock_path).unlink(missing_ok=True)


def _checkout_head_commit(
    repo_dir: Path,
    head_sha: str,
    timeout: int = 120,
) -> Tuple[bool, str]:
    """Checkout an exact PR head commit, fetching it if necessary."""
    ok, err = _run_git(["checkout", "--detach", head_sha], cwd=str(repo_dir), timeout=timeout)
    if ok:
        return True, ""
    _run_git(["fetch", "--depth", "50", "origin", head_sha], cwd=str(repo_dir), timeout=timeout)
    return _run_git(["checkout", "--detach", head_sha], cwd=str(repo_dir), timeout=timeout)


def _checkout_pr_head(
    repo_dir: Path,
    aidev_sha: str,
    pr_number: Optional[int],
    timeout: int,
    base_sha: Optional[str] = None,
    head_repo_full_name: Optional[str] = None,
) -> Tuple[bool, str, Optional[str], str]:
    """Acquire a historical PR head using immutable metadata before refs.

    ``pr_commits`` does not guarantee that its first row is the final PR head;
    fetching ``refs/pull/<number>/head`` gives the immutable PR snapshot used
    to derive the actual change.  The AIDev SHA remains a compatibility
    fallback for hosts that do not expose pull refs.

    Returns:
        (success, checked_out_head_sha, base_sha_or_None)
        base_sha is the PR base commit from refs/pull/<number>/base when available,
        otherwise None (caller must fall back to head^).
    """
    def fetch_exact(remote: str, sha: str) -> Tuple[bool, str]:
        ok_fetch, fetch_error = _run_git(
            ["fetch", "--depth", "2", remote, sha], cwd=str(repo_dir), timeout=timeout
        )
        return ok_fetch, fetch_error

    def fetch_and_checkout(remote: str, sha: str) -> Tuple[bool, str]:
        ok_fetch, fetch_error = fetch_exact(remote, sha)
        if not ok_fetch:
            return False, fetch_error
        return _checkout_head_commit(repo_dir, sha, timeout)

    # The API's head SHA is immutable and remains authoritative after a PR is
    # closed or merged.  GitHub may retain it in the base repository even after
    # the source branch (including a fork branch) has been deleted.
    if base_sha and base_sha != aidev_sha:
        ok_head, head_error = fetch_and_checkout("origin", aidev_sha)
        if ok_head:
            ok_base, base_error = fetch_exact("origin", base_sha)
            if ok_base:
                return True, aidev_sha, base_sha, ""
            # The head is usable; continue to PR-base-ref fallback to obtain
            # the authoritative base object without losing the exact head.
            logger.debug("exact PR base SHA fetch failed: %s", base_error)
        else:
            logger.debug("exact PR head SHA fetch failed: %s", head_error)
    elif base_sha:
        ok_head, head_error = fetch_and_checkout("origin", aidev_sha)
        if ok_head:
            ok_base, base_error = fetch_exact("origin", base_sha)
            if ok_base:
                return True, aidev_sha, base_sha, ""
            logger.debug("exact PR base SHA fetch failed: %s", base_error)

    # A fork's deleted branch may still be reachable from its base PR ref.
    if pr_number and pr_number > 0:
        # Fetch both head and base refs for the PR
        # Store the two moving FETCH_HEAD results under explicit refs.  A
        # second fetch overwrites FETCH_HEAD, so reading it after fetching the
        # base can otherwise accidentally checkout the base as the PR head.
        head_ref = f"refs/pilot/pr/{pr_number}/head"
        base_ref = f"refs/pilot/pr/{pr_number}/base"
        ok_head, err = _run_git(
            ["fetch", "--depth", "2", "origin", f"pull/{pr_number}/head:{head_ref}"],
            cwd=str(repo_dir), timeout=timeout,
        )
        ok_base, base_err = _run_git(
            ["fetch", "--depth", "2", "origin", f"pull/{pr_number}/base:{base_ref}"],
            cwd=str(repo_dir), timeout=timeout,
        )

        base_sha: Optional[str] = None
        if ok_base:
            ok_base_rev, base_sha_candidate = _run_git(
                ["rev-parse", base_ref], cwd=str(repo_dir), timeout=timeout
            )
            if ok_base_rev:
                base_sha = base_sha_candidate

        if ok_head:
            ok_head_rev, pr_head = _run_git(
                ["rev-parse", head_ref], cwd=str(repo_dir), timeout=timeout
            )
            if ok_head_rev:
                ok_checkout, err = _checkout_head_commit(repo_dir, pr_head, timeout)
                if ok_checkout:
                    return True, pr_head, base_sha, ""
        else:
            logger.debug("PR head ref fetch failed for #%s: %s", pr_number, err)
            if ok_base:
                logger.debug("PR base ref fetch succeeded for #%s: %s", pr_number, base_sha)

    # A fork remote is useful only while the fork repository still exists and
    # retains the historical object.  Do not invent a replacement SHA.
    if head_repo_full_name and head_repo_full_name != "":
        fork_url = f"https://github.com/{head_repo_full_name}.git"
        ok_fork, fork_error = fetch_and_checkout(fork_url, aidev_sha)
        if ok_fork:
            return True, aidev_sha, base_sha, ""
        logger.debug("fork exact head fetch failed: %s", fork_error)

    ok_checkout, err = fetch_and_checkout("origin", aidev_sha)
    if ok_checkout:
        return True, aidev_sha, base_sha, ""
    return False, aidev_sha, base_sha, "HEAD_COMMIT_UNAVAILABLE: " + (err or "exact commit could not be fetched")


def _get_checked_out_pr_diff(repo_dir: Path, head_sha: str, timeout: int = 120) -> Tuple[bool, str, str]:
    """Return (ok, base_sha, diff) for the checked-out PR head.

    The AIDev ``patch`` column stores independent file hunks and omits the
    file headers required by our UnifiedDiffParser.  Git is therefore the
    authoritative source for the complete, parseable PR diff.
    """
    ok, base_sha = _run_git(["rev-parse", f"{head_sha}^"], cwd=str(repo_dir), timeout=timeout)
    if not ok:
        return False, "", f"base resolution failed: {base_sha}"
    ok, diff = _run_git(
        ["diff", "--find-renames", "--no-ext-diff", base_sha, head_sha],
        cwd=str(repo_dir), timeout=timeout,
    )
    if not ok:
        return False, "", f"git diff failed: {diff}"
    if not diff.strip():
        return False, "", "git diff was empty"
    return True, base_sha, diff


def _git_source_provider(
    repo_dir: Path, base_sha: str,
) -> Callable[[str, str], Optional[str]]:
    """Provide old source from the base and new source from the head tree."""
    def provider(file_path: str, version: str) -> Optional[str]:
        if version == "new":
            path = repo_dir / file_path
            try:
                return path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return None
        ok, source = _run_git(["show", f"{base_sha}:{file_path}"], cwd=str(repo_dir))
        return source if ok else None
    return provider


# ---------------------------------------------------------------------------
# Full pipeline: repo → graph → impact score → 42 features
# ---------------------------------------------------------------------------


def _run_full_analysis(
    repo_dir: Path,
    patch_text: str,
    base_sha: str,
    diagnostics: Optional[Dict[str, Any]] = None,
    max_source_files: Optional[int] = None,
    max_total_runtime: Optional[float] = None,
    max_dependency_neighborhood_files: int = 5000,
    max_parsed_files: int = 5000,
    max_graph_nodes: int = 100000,
    force_full_repository: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Run the complete analysis pipeline on a checked-out repository.

    Steps:
    1. Detect supported source files and build the dependency graph through
       the language-adapter / ParsedSource IR path.
    2. Run ChangeAnalyzer on the patch.
    3. Run ImpactAnalyzer through the graph.
    4. Run ImpactScorer.
    5. Return the 42-feature dict.

    Args:
        repo_dir: Path to the checked-out repository.
        patch_text: Unified diff patch text for this PR.

    Returns:
        Dict with 42 features, or None if analysis fails.
    """
    started = time.monotonic()
    timings = diagnostics if diagnostics is not None else {}
    timings.setdefault("stages", {})
    def mark(name: str, before: float) -> float:
        now = time.monotonic()
        timings["stages"][name] = round((now - before) * 1000, 3)
        return now
    def guard() -> None:
        if max_total_runtime is not None and time.monotonic() - started > max_total_runtime:
            raise TimeoutError("ANALYSIS_RUNTIME_LIMIT_EXCEEDED")
    try:
        from analyzer.graph_engine import GraphBuilder, CustomDependencyGraph
        from analyzer.change_analyzer import ChangeAnalyzer
        from analyzer.impact_scorer import analyze_and_score
        from analyzer.languages.detector import LanguageDetector
        from analyzer.languages.registry import default_registry
        from analyzer.resolver_coordinator import resolve_project
        from analyzer.scoped_analysis import build_repository_index, select_dependency_neighborhood

        # Detection and support are deliberately distinct: a recognized but
        # unsupported language is not passed to the Python adapter.
        detector = LanguageDetector()
        # The repository-wide pass is lexical indexing only.  It records paths,
        # modules, symbols, imports and tokens; adapters see only the relevant
        # dependency neighborhood.  max_source_files is retained for API
        # compatibility but is intentionally not a normal rejection rule.
        enum_started = time.monotonic()
        head_revision_ok, head_revision = _run_git(["rev-parse", "HEAD"], cwd=str(repo_dir), timeout=30)
        index_revision = head_revision if head_revision_ok else base_sha
        index = build_repository_index(repo_dir, revision=index_revision, lazy=True)
        diff_paths = []
        for line in patch_text.splitlines():
            if line.startswith("+++ ") and line[4:].strip() != "/dev/null":
                diff_paths.append(line[4:].strip().removeprefix("b/"))
        selection = select_dependency_neighborhood(
            index, diff_paths, max_files=max_dependency_neighborhood_files
        )
        mark("changed-file discovery", enum_started)
        timings.update({
            "repository_supported_source_files": index.supported_files,
            "changed_supported_files": sum(1 for p in selection.changed_files if p in index.entries),
            "indexed_files": index.indexed_files,
            "dependency_neighborhood_files": selection.dependency_neighborhood_files,
            "closure_depth": selection.closure_depth,
            "closure_complete": selection.closure_complete,
            "index_cache_hit": index.cache_hit,
            "changed_file_list": list(selection.changed_files),
            "changed_supported_file_list": sorted(p for p in selection.changed_files if p in index.entries),
            "supported_head_files": sorted(index.entries),
            "scoped_neighborhood_files": sorted(p.relative_to(index.root).as_posix() for p in selection.files),
        })
        if not selection.closure_complete:
            timings["safety_failure"] = selection.reason or "DEPENDENCY_CLOSURE_LIMIT_EXCEEDED"
            raise RuntimeError(timings["safety_failure"])
        source_files = [index.root / p for p in sorted(index.entries)] if force_full_repository else selection.files

        parsed_sources = []
        load_started = time.monotonic()
        parse_started = time.monotonic()
        captured_warning_count = 0
        for source_file in source_files:
            detected = detector.detect(str(source_file))
            adapter = default_registry.get(detected.language)
            if adapter is None:
                continue
            try:
                source_text = source_file.read_text(encoding="utf-8")
                # Repository code is untrusted input.  SyntaxWarning noise is
                # expected in historical third-party files; analyzer errors
                # and parse failures remain visible through normal diagnostics.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", SyntaxWarning)
                    parsed = adapter.parse(source_text, str(source_file))
                captured_warning_count += sum(1 for item in caught
                                              if issubclass(item.category, SyntaxWarning))
                if not parsed.has_errors:
                    parsed_sources.append(parsed)
            except (OSError, UnicodeDecodeError):
                continue
            if len(parsed_sources) > max_parsed_files:
                raise RuntimeError("GRAPH_RESOURCE_LIMIT_EXCEEDED")
            guard()
        mark("source file loading", load_started)
        mark("parsing/adapters", parse_started)
        timings["analyzed_source_files"] = len(parsed_sources)
        timings["parsed_files"] = len(parsed_sources)
        timings["repository_source_files"] = index.supported_files
        timings["repository_syntax_warning_count"] = captured_warning_count

        if not parsed_sources:
            logger.warning(f"No parseable supported sources in {repo_dir}")
            if not timings.get("changed_supported_file_list"):
                timings["safety_failure"] = "NO_SUPPORTED_SOURCE_CHANGE"
                timings["failure_reason"] = "PR changes no supported-language source files"
            else:
                timings["failure_reason"] = "supported source files were selected but none were parseable"
            return None

        step = time.monotonic()
        resolved_sources = resolve_project(repo_dir, parsed_sources)
        mark("resolver construction", step)
        timings["stages"]["dependency extraction"] = timings["stages"]["resolver construction"]
        step = time.monotonic()
        graph = GraphBuilder(repo_dir).build_from_ir(resolved_sources)
        mark("graph construction", step)
        if not isinstance(graph, CustomDependencyGraph):
            logger.warning("Graph builder returned unexpected type")
            return None

        # Analyze against actual old/new source snapshots so symbol and import
        # extraction is tied to the checked-out PR rather than partial hunk text.
        change_analyzer = ChangeAnalyzer(
            source_provider=_git_source_provider(repo_dir, base_sha),
            language_registry=default_registry,
        )
        step = time.monotonic()
        change_set = change_analyzer.analyze_diff(patch_text)
        mark("ChangeAnalyzer", step)

        # Run Phase 5 + 6 pipeline
        step = time.monotonic()
        scoring_result = analyze_and_score(change_set, graph)
        mark("impact propagation/scoring", step)

        # Extract the 42-feature dict from the scorer
        features: Dict[str, float] = dict(scoring_result.features)

        # Validate all 42 features present
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            logger.warning(f"Missing features from scorer: {missing}")
            for m in missing:
                features[m] = float("nan")

        timings["graph_node_count"] = len(getattr(graph, "nodes", {}) or {})
        edges = getattr(graph, "edges", {}) or {}
        timings["graph_edge_count"] = (sum(len(v) for v in edges.values())
                                        if hasattr(edges, "values") else len(edges))
        timings["graph_nodes"] = timings["graph_node_count"]
        timings["graph_edges"] = timings["graph_edge_count"]
        if timings["graph_node_count"] > max_graph_nodes:
            raise RuntimeError("GRAPH_RESOURCE_LIMIT_EXCEEDED")
        timings["analysis_runtime_ms"] = round((time.monotonic() - started) * 1000, 3)
        timings["feature_count"] = len(features)
        return features

    except Exception as exc:
        logger.warning(f"Full analysis failed: {exc}")
        if diagnostics is not None and str(exc) in {"ANALYSIS_TIMEOUT", "ANALYSIS_RUNTIME_LIMIT_EXCEEDED", "REPOSITORY_TOO_LARGE", "GRAPH_LIMIT_EXCEEDED", "GRAPH_RESOURCE_LIMIT_EXCEEDED", "DEPENDENCY_CLOSURE_LIMIT_EXCEEDED"}:
            diagnostics["safety_failure"] = str(exc)
        if diagnostics is not None:
            diagnostics.setdefault("analysis_runtime_ms", round((time.monotonic() - started) * 1000, 3))
        return None


# ---------------------------------------------------------------------------
# AIDev Dataset Builder
# ---------------------------------------------------------------------------


class AIDevDatasetBuilder:
    """
    Builds an ML dataset from real AIDev Pull Request data.

    Pipeline:
        1. Load repository metadata (filter by language, license, fork).
        2. Load PR metadata (filter by state=closed).
        3. Load review data (filter: require at least one review).
        4. Load task types (filter: exclude revert).
        5. Deterministic sampling (seed=42).
        6. Per-PR: extract patch, label, features (patch_only or full checkout).
        7. Save to Parquet.
    """

    HF_BASE = "hf://datasets/hao-li/AIDev"

    # Confirmed flat-file paths (NOT split-based):
    # hf://datasets/hao-li/AIDev/repository.parquet         — 6,673 repos
    # hf://datasets/hao-li/AIDev/pull_request.parquet        — curated subset
    # hf://datasets/hao-li/AIDev/pr_reviews.parquet
    # hf://datasets/hao-li/AIDev/pr_commits.parquet
    # hf://datasets/hao-li/AIDev/pr_commit_details.parquet   — ~1.3 GB
    # NOTE: pr_task_type.parquet does NOT exist in AIDev v4; revert filter is skipped.
    _TASK_TYPE_AVAILABLE: bool = False  # set False: confirmed absent from dataset

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        language_filter: Union[str, Sequence[str]] = "Python",
        allowed_licenses: Optional[Set[str]] = None,
        exclude_forks: bool = True,
        random_seed: int = 42,
        clone_depth: int = 50,
        git_timeout: int = 120,
        max_repos_on_disk: int = 5,
        max_repo_size_mb: int = 500,
        checkout_enabled: bool = True,
        repo_cache_dir: Optional[Path] = None,
        require_review: bool = True,
        require_merged: bool = False,
        require_full_analysis: bool = False,
        exclude_pr_ids: Optional[Set[int]] = None,
    ) -> None:
        if not _POLARS_AVAILABLE:
            raise ImportError("polars is required. pip install polars")

        self.output_dir = output_dir or (Path(__file__).parent.parent.parent / "data" / "processed")
        if isinstance(language_filter, str):
            self.language_filters: Tuple[str, ...] = (language_filter,)
            self.language_filter = language_filter
        else:
            self.language_filters = tuple(language_filter)
            self.language_filter = self.language_filters[0] if self.language_filters else "Python"
        self.require_review = require_review
        self.require_merged = require_merged
        self.require_full_analysis = require_full_analysis
        self.exclude_pr_ids: Set[int] = set(exclude_pr_ids or ())
        self.allowed_licenses: Set[str] = allowed_licenses or {
            "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"
        }
        self.exclude_forks = exclude_forks
        self.random_seed = random_seed
        self.clone_depth = clone_depth
        self.git_timeout = git_timeout
        self.max_repos_on_disk = max_repos_on_disk
        self.max_repo_size_mb = max_repo_size_mb
        self.checkout_enabled = checkout_enabled
        self.counters = FilterCounters()

        # Temporary directory for repo clones
        if repo_cache_dir:
            self.repo_cache_dir = Path(repo_cache_dir)
        else:
            self.repo_cache_dir = Path(tempfile.mkdtemp(prefix="aidev_repos_"))
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        self._cloned_repos: List[str] = []  # FIFO queue for eviction

    def _normalize_license(self, raw: Optional[str]) -> Optional[str]:
        """Normalize license string to lowercase SPDX-like identifier."""
        if not raw:
            return None
        raw = raw.strip().lower()
        # Common aliases
        aliases = {
            "mit license": "mit",
            "apache license 2.0": "apache-2.0",
            "apache 2.0": "apache-2.0",
            "apache-2": "apache-2.0",
            "bsd 2-clause": "bsd-2-clause",
            "bsd 3-clause": "bsd-3-clause",
            "isc license": "isc",
        }
        return aliases.get(raw, raw)

    def _load_eligible_repos(self) -> "pl.DataFrame":
        """
        Load repository table and filter to eligible repos.

        Returns Polars DataFrame with columns: id, full_name, language, license, is_forked.
        """
        logger.info("Loading repository metadata from AIDev...")
        repo_path = f"{self.HF_BASE}/repository.parquet"
        try:
            repo_df = pl.read_parquet(
                repo_path,
                columns=["id", "full_name", "language", "license", "is_forked", "stars"],
            )
        except Exception as exc:
            logger.warning(f"Could not load curated 'repository' table ({exc}); trying all_repository.")
            repo_path = f"{self.HF_BASE}/all_repository.parquet"
            repo_df = pl.read_parquet(
                repo_path,
                columns=["id", "full_name", "language", "license", "is_forked"],
            )

        # Language filter — a single GitHub language or the validated adapter set.
        before = len(repo_df)
        repo_df = repo_df.filter(pl.col("language").is_in(list(self.language_filters)))
        logger.info(f"Language filter ({list(self.language_filters)}): {before} → {len(repo_df)}")
        self.counters.excluded_language = before - len(repo_df)

        # License filter
        before = len(repo_df)

        def _license_ok(lic: Optional[str]) -> bool:
            normalized = self._normalize_license(lic)
            if normalized is None:
                return False
            return any(normalized == a or normalized.startswith(a) for a in self.allowed_licenses)

        repo_df = repo_df.filter(
            pl.col("license").map_elements(_license_ok, return_dtype=pl.Boolean)
        )
        logger.info(f"License filter: {before} → {len(repo_df)}")

        # Fork filter
        if self.exclude_forks:
            before = len(repo_df)
            repo_df = repo_df.filter(pl.col("is_forked") == False)  # noqa: E712
            logger.info(f"Fork filter: {before} → {len(repo_df)}")

        logger.info(f"Eligible repositories: {len(repo_df)}")
        return repo_df

    def _load_eligible_prs(self, eligible_repo_ids: List[int]) -> "pl.DataFrame":
        """Load PR table filtered to eligible repos and closed state."""
        logger.info("Loading PR metadata from AIDev...")
        pr_path = f"{self.HF_BASE}/pull_request.parquet"
        try:
            pr_df = (
                pl.scan_parquet(
                    pr_path,
                    # Note: merged_at and closed_at needed only for label — NOT features
                )
                .select(["id", "number", "repo_id", "state", "created_at",
                         "merged_at", "closed_at", "agent"])
                .filter(
                    pl.col("repo_id").is_in(eligible_repo_ids)
                    & (pl.col("state") == "closed")
                )
                .collect()
            )
        except Exception as exc:
            logger.warning(f"Could not scan 'pull_request' ({exc}); trying all_pull_request.")
            pr_path = f"{self.HF_BASE}/all_pull_request.parquet"
            pr_df = (
                pl.scan_parquet(pr_path)
                .select(["id", "number", "repo_id", "state", "created_at",
                         "merged_at", "closed_at", "agent"])
                .filter(
                    pl.col("repo_id").is_in(eligible_repo_ids)
                    & (pl.col("state") == "closed")
                )
                .collect()
            )
        if self.require_merged:
            before = len(pr_df)
            pr_df = pr_df.filter(pl.col("merged_at").is_not_null())
            self.counters.excluded_not_merged = before - len(pr_df)
        if self.exclude_pr_ids:
            pr_df = pr_df.filter(~pl.col("id").is_in(list(self.exclude_pr_ids)))
        logger.info(f"Closed PRs in eligible repos: {len(pr_df)}")
        return pr_df

    def _load_reviews(self, pr_ids: List[int]) -> "pl.DataFrame":
        """Load review data for the given PR IDs."""
        logger.info(f"Loading reviews for {len(pr_ids)} PRs...")
        review_path = f"{self.HF_BASE}/pr_reviews.parquet"
        try:
            reviews = (
                pl.scan_parquet(
                    review_path,
                )
                .select(["pr_id", "state"])
                .filter(pl.col("pr_id").is_in(pr_ids))
                .collect()
            )
        except Exception as exc:
            logger.error(f"Could not load reviews: {exc}")
            reviews = pl.DataFrame({"pr_id": [], "state": []})
        logger.info(f"Reviews loaded: {len(reviews)} rows")
        return reviews

    def _load_task_types(self, pr_ids: List[int]) -> "pl.DataFrame":
        """Load task type data for the given PR IDs.

        NOTE: pr_task_type.parquet is confirmed absent from AIDev v4.
        This method returns an empty DataFrame, making the revert filter a no-op.
        If the table becomes available in a future dataset version, it will be
        picked up automatically by the try/except path.
        """
        if not self._TASK_TYPE_AVAILABLE:
            # Confirmed absent — skip attempt to avoid confusing HuggingFace 404 errors.
            logger.info(
                "pr_task_type.parquet is absent from AIDev v4; "
                "revert filter skipped (no revert PRs will be excluded)."
            )
            return pl.DataFrame({"pr_id": pl.Series([], dtype=pl.Int64),
                                  "task_type": pl.Series([], dtype=pl.String)})

        task_path = f"{self.HF_BASE}/pr_task_type.parquet"
        try:
            tasks = (
                pl.scan_parquet(task_path)
                .select(["pr_id", "task_type"])
                .filter(pl.col("pr_id").is_in(pr_ids))
                .collect()
            )
        except Exception as exc:
            logger.info(f"Could not load task types ({type(exc).__name__}); skipping revert filter.")
            tasks = pl.DataFrame({"pr_id": pl.Series([], dtype=pl.Int64),
                                   "task_type": pl.Series([], dtype=pl.String)})
        return tasks

    def _load_patches(self, pr_ids: List[int]) -> "pl.DataFrame":
        """
        Load pr_commit_details for the given PR IDs.
        Only selected columns; patch text is mandatory.
        """
        logger.info(f"Loading patches for {len(pr_ids)} PRs...")
        detail_path = f"{self.HF_BASE}/pr_commit_details.parquet"
        try:
            details = (
                pl.scan_parquet(detail_path)
                .select(["pr_id", "sha", "filename", "status",
                         "additions", "deletions", "changes", "patch"])
                .filter(pl.col("pr_id").is_in(pr_ids))
                .collect()
            )
        except Exception as exc:
            logger.error(f"Could not load pr_commit_details: {exc}")
            details = pl.DataFrame({
                "pr_id": [], "sha": [], "filename": [], "status": [],
                "additions": [], "deletions": [], "changes": [], "patch": [],
            })
        logger.info(f"Commit detail rows loaded: {len(details)}")
        return details

    def _load_commits(self, pr_ids: List[int]) -> "pl.DataFrame":
        """Load pr_commits for the given PR IDs (to get head SHA)."""
        commit_path = f"{self.HF_BASE}/pr_commits.parquet"
        try:
            commits = (
                pl.scan_parquet(commit_path)
                .select(["pr_id", "sha"])
                .filter(pl.col("pr_id").is_in(pr_ids))
                .collect()
            )
        except Exception as exc:
            logger.warning(f"Could not load pr_commits: {exc}")
            commits = pl.DataFrame({"pr_id": [], "sha": []})
        return commits

    def _build_label_map(
        self, reviews_df: "pl.DataFrame"
    ) -> Dict[int, int]:
        """Build pr_id → high_risk label dict from reviews.

        Uses case-insensitive comparison to match labels.py semantics.
        Any pr_id with at least one CHANGES_REQUESTED review → high_risk=1.
        """
        label_map: Dict[int, int] = {}
        # Group by pr_id; any CHANGES_REQUESTED → 1
        for row in reviews_df.iter_rows(named=True):
            pid = int(row["pr_id"])
            if pid not in label_map:
                label_map[pid] = 0
            state = row.get("state")
            if state and state.upper() == "CHANGES_REQUESTED":
                label_map[pid] = 1
        return label_map

    def _evict_oldest_repo(self) -> None:
        """Remove the oldest cloned repository from disk."""
        if self._cloned_repos:
            oldest = self._cloned_repos.pop(0)
            oldest_path = self.repo_cache_dir / "r" / oldest
            if oldest_path.exists():
                shutil.rmtree(oldest_path, ignore_errors=True)
                logger.debug(f"Evicted cached repo: {oldest}")

    def _process_pr(
        self,
        pr_row: Dict[str, Any],
        repo_full_name: str,
        patch_text: str,
        head_sha: str,
        label: int,
    ) -> Optional[FeatureRow]:
        # A shared repository cache is safe only when checkout, diff, and
        # analysis are one critical section.  This prevents another PR from
        # changing the working tree between checkout and source extraction.
        if self.checkout_enabled and head_sha:
            try:
                with _repository_lease(self.repo_cache_dir, repo_full_name, self.git_timeout):
                    return self._process_pr_unlocked(pr_row, repo_full_name, patch_text, head_sha, label)
            except TimeoutError as exc:
                row = self._make_feature_row(pr_row, repo_full_name, label)
                row.extraction_error = f"checkout: {exc}"
                self.counters.checkout_failed += 1
                return row
        return self._process_pr_unlocked(pr_row, repo_full_name, patch_text, head_sha, label)

    def _make_feature_row(self, pr_row: Dict[str, Any], repo_full_name: str, label: int) -> FeatureRow:
        return FeatureRow(
            pr_id=int(pr_row["id"]), repo_id=int(pr_row["repo_id"]),
            repo_full_name=repo_full_name, pr_number=int(pr_row.get("number", 0)),
            created_at=str(pr_row.get("created_at", "")), agent=pr_row.get("agent"),
            high_risk=label,
        )

    def _process_pr_unlocked(
        self,
        pr_row: Dict[str, Any],
        repo_full_name: str,
        patch_text: str,
        head_sha: str,
        label: int,
    ) -> Optional[FeatureRow]:
        """
        Process a single PR: extract features and build a FeatureRow.

        Args:
            pr_row: PR metadata dict (from pull_request table).
            repo_full_name: "owner/repo".
            patch_text: Concatenated unified diff for the PR.
            head_sha: First commit SHA of the PR.
            label: high_risk label (0 or 1).

        Returns:
            FeatureRow or None if the PR should be skipped.
        """
        pr_id = int(pr_row["id"])
        repo_id = int(pr_row["repo_id"])

        row = self._make_feature_row(pr_row, repo_full_name, label)

        # --- Step 1: Always extract Cat A + B from patch ---
        if not patch_text.strip():
            self.counters.analysis_failed += 1
            row.extraction_error = "empty_patch"
            return None

        try:
            patch_feats = _extract_patch_features(patch_text)
            row.files_changed = patch_feats.get("files_changed", float("nan"))
            row.lines_added = patch_feats.get("lines_added", float("nan"))
            row.lines_deleted = patch_feats.get("lines_deleted", float("nan"))
            row.total_lines_changed = patch_feats.get("total_lines_changed", float("nan"))
            row.functions_changed = patch_feats.get("functions_changed", float("nan"))
            row.classes_changed = patch_feats.get("classes_changed", float("nan"))
            row.methods_changed = patch_feats.get("methods_changed", float("nan"))
            row.dependencies_added = patch_feats.get("dependencies_added", float("nan"))
            row.dependencies_removed = patch_feats.get("dependencies_removed", float("nan"))
            row.dependencies_changed = patch_feats.get("dependencies_changed", float("nan"))
            row.dependency_change_ratio = patch_feats.get("dependency_change_ratio", float("nan"))
            row.analysis_mode = "patch_only"
        except Exception as exc:
            logger.warning(f"PR {pr_id}: patch feature extraction failed: {exc}")
            row.extraction_error = f"patch_failed: {exc}"
            # Cat A/B remain NaN — not zeroed

        # --- Step 2: Full checkout for Cat C features ---
        if self.checkout_enabled and head_sha:
            # Evict repos if at capacity
            while len(self._cloned_repos) >= self.max_repos_on_disk:
                self._evict_oldest_repo()

            ok, repo_dir, base_sha, err = _checkout_repo(
                repo_full_name,
                head_sha,
                self.repo_cache_dir,
                pr_number=int(pr_row.get("number", 0)),
                clone_depth=self.clone_depth,
                timeout=self.git_timeout,
                max_size_mb=self.max_repo_size_mb,
            )
            if not ok:
                logger.warning(f"PR {pr_id} checkout failed: {err}")
                self.counters.checkout_failed += 1
                row.checkout_success = False
                row.extraction_error = f"checkout: {err}"
                # Cat C features remain NaN — analysis failure is NOT zero
            else:
                row.checkout_success = True
                repo_key = hashlib.sha256(repo_full_name.strip().lower().encode("utf-8")).hexdigest()[:16]
                if repo_key not in self._cloned_repos:
                    self._cloned_repos.append(repo_key)

                # repo_dir is non-None when ok=True (guaranteed by _checkout_repo)
                assert repo_dir is not None

                # Use the PR base SHA from refs/pull/<number>/base if available,
                # otherwise fall back to head^ (first parent of checked-out head).
                if base_sha is None:
                    head_ok, checked_out_head = _run_git(
                        ["rev-parse", "HEAD"], cwd=str(repo_dir), timeout=self.git_timeout
                    )
                    if not head_ok:
                        self.counters.analysis_failed += 1
                        row.extraction_error = f"analysis_head: {checked_out_head}"
                        return row
                    diff_ok, base_sha_fallback, full_diff_or_error = _get_checked_out_pr_diff(
                        repo_dir, checked_out_head, self.git_timeout
                    )
                    if not diff_ok:
                        self.counters.analysis_failed += 1
                        row.extraction_error = f"analysis_diff: {full_diff_or_error}"
                        return row
                    base_sha = base_sha_fallback
                else:
                    # We have the PR base SHA from refs/pull/<number>/base.
                    # Compute the diff between base and the checked-out head.
                    head_ok, checked_out_head = _run_git(
                        ["rev-parse", "HEAD"], cwd=str(repo_dir), timeout=self.git_timeout
                    )
                    if not head_ok:
                        self.counters.analysis_failed += 1
                        row.extraction_error = f"analysis_head: {checked_out_head}"
                        return row
                    diff_ok, full_diff_or_error = _run_git(
                        ["diff", "--find-renames", "--no-ext-diff", base_sha, checked_out_head],
                        cwd=str(repo_dir), timeout=self.git_timeout,
                    )
                    if not diff_ok:
                        self.counters.analysis_failed += 1
                        row.extraction_error = f"analysis_diff: {full_diff_or_error}"
                        return row
                    if not full_diff_or_error.strip():
                        self.counters.analysis_failed += 1
                        row.extraction_error = "analysis_diff: git diff was empty (PR base vs head)"
                        return row

                # Run Phase 5 change analysis, resolver/graph construction,
                # and Phase 6 impact scoring on the real source snapshots.
                all_features = _run_full_analysis(repo_dir, full_diff_or_error, base_sha)
                if all_features is not None:
                    # Overwrite Cat A/B with more accurate values from full analysis
                    for feat_name in FEATURE_NAMES:
                        val = all_features.get(feat_name)
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            setattr(row, feat_name, float(val))
                    row.analysis_mode = "full"
                else:
                    self.counters.analysis_failed += 1
                    row.extraction_error = "analysis_failed"
                    # Cat C remain NaN

        return row

    def build(
        self,
        target_size: int = 3000,
        pilot_mode: bool = False,
        pilot_size: int = 25,
    ) -> List[FeatureRow]:
        """
        Build the dataset by processing AIDev PRs.

        Args:
            target_size: Target number of successfully extracted PRs.
            pilot_mode: If True, only process pilot_size PRs (for validation).
            pilot_size: Number of PRs to process in pilot mode.

        Returns:
            List of FeatureRow objects (one per successfully extracted PR).
        """
        sample_cap = pilot_size if pilot_mode else target_size

        # === Phase 1: Load eligible repositories ===
        repo_df = self._load_eligible_repos()
        eligible_repo_ids = repo_df["id"].to_list()
        # Build id→full_name map
        repo_name_map: Dict[int, str] = {}
        for row in repo_df.iter_rows(named=True):
            repo_name_map[int(row["id"])] = str(row["full_name"])

        if not eligible_repo_ids:
            logger.error("No eligible repositories found. Check filters.")
            return []

        # === Phase 2: Load eligible PRs ===
        pr_df = self._load_eligible_prs(eligible_repo_ids)
        pr_ids_all = pr_df["id"].to_list()
        self.counters.considered = len(pr_ids_all)

        # === Phase 3: Load reviews (needed for label and review filter) ===
        reviews_df = self._load_reviews(pr_ids_all)

        if self.require_review:
            reviewed_pr_ids = set(reviews_df["pr_id"].to_list())
            pr_df = pr_df.filter(pl.col("id").is_in(list(reviewed_pr_ids)))
            self.counters.excluded_no_review = self.counters.considered - len(pr_df)
            logger.info(f"PRs with at least one review: {len(pr_df)}")
        else:
            self.counters.excluded_no_review = 0
            logger.info("Review filter disabled (Phase 7D labels do not use review friction).")

        pr_ids_reviewed = pr_df["id"].to_list()

        # === Phase 4: Load task types, filter reverts ===
        tasks_df = self._load_task_types(pr_ids_reviewed)
        revert_pr_ids: Set[int] = set()
        for row in tasks_df.iter_rows(named=True):
            if row.get("task_type") == "revert":
                revert_pr_ids.add(int(row["pr_id"]))
        pr_df = pr_df.filter(~pl.col("id").is_in(list(revert_pr_ids)))
        self.counters.excluded_revert = len(revert_pr_ids)
        logger.info(f"After excluding reverts: {len(pr_df)} PRs")

        pr_ids_filtered = pr_df["id"].to_list()

        # === Phase 5: Deterministic sampling ===
        rng = np.random.RandomState(self.random_seed)
        pool_multiplier = 12 if self.require_full_analysis else 3
        if len(pr_ids_filtered) > sample_cap * pool_multiplier:
            # Sample a pool larger than the target to allow for checkout/analysis failures
            candidate_ids = rng.choice(
                pr_ids_filtered,
                size=min(sample_cap * pool_multiplier, len(pr_ids_filtered)),
                replace=False,
            ).tolist()
        else:
            candidate_ids = list(pr_ids_filtered)
            rng.shuffle(candidate_ids)

        logger.info(f"Candidate PR pool: {len(candidate_ids)} (target: {sample_cap})")

        # === Phase 6: Load patches for candidates ===
        patches_df = self._load_patches(candidate_ids)

        # Build pr_id → concatenated patch text map
        patch_map: Dict[int, str] = {}
        for row in patches_df.iter_rows(named=True):
            pid = int(row["pr_id"])
            patch = row.get("patch") or ""
            if pid not in patch_map:
                patch_map[pid] = ""
            # AIDev stores GitHub's per-file hunk body, which deliberately
            # omits ``diff --git``/---/+++ headers.  Recreate those headers so
            # the existing Phase 5 UnifiedDiffParser can read the patch even
            # when a repository checkout cannot be analyzed.
            filename = str(row.get("filename") or "")
            status = str(row.get("status") or "modified")
            if filename and patch:
                old_path = "/dev/null" if status == "added" else f"a/{filename}"
                new_path = "/dev/null" if status == "removed" else f"b/{filename}"
                patch_map[pid] += (
                    f"diff --git a/{filename} b/{filename}\n"
                    f"--- {old_path}\n"
                    f"+++ {new_path}\n"
                    f"{patch}\n"
                )

        # === Phase 7: Load head commit SHAs ===
        commits_df = self._load_commits(candidate_ids)
        sha_map: Dict[int, str] = {}
        for row in commits_df.iter_rows(named=True):
            pid = int(row["pr_id"])
            if pid not in sha_map:
                sha_map[pid] = str(row.get("sha", ""))

        # === Phase 8: Build label map ===
        label_map = self._build_label_map(
            reviews_df.filter(pl.col("pr_id").is_in(candidate_ids))
        )

        # === Phase 9: Per-PR processing ===
        pr_meta: Dict[int, Dict[str, Any]] = {
            int(r["id"]): dict(r)
            for r in pr_df.filter(pl.col("id").is_in(candidate_ids)).iter_rows(named=True)
        }

        results: List[FeatureRow] = []

        for pr_id in candidate_ids:
            if len(results) >= sample_cap:
                break

            # Skip PRs without patches
            patch_text = patch_map.get(pr_id, "")
            if not patch_text.strip():
                self.counters.excluded_no_patch += 1
                continue

            pr_meta_row = pr_meta.get(pr_id)
            if pr_meta_row is None:
                continue

            repo_id = int(pr_meta_row.get("repo_id", 0))
            repo_full_name = repo_name_map.get(repo_id, "")
            if not repo_full_name:
                continue

            label = label_map.get(pr_id, 0)
            head_sha = sha_map.get(pr_id, "")

            logger.info(
                f"[{len(results)+1}/{sample_cap}] Processing PR {pr_id} "
                f"({repo_full_name}) label={label}"
            )

            feature_row = self._process_pr(
                pr_row=pr_meta_row,
                repo_full_name=repo_full_name,
                patch_text=patch_text,
                head_sha=head_sha,
                label=label,
            )

            if feature_row is not None:
                if self.require_full_analysis and feature_row.analysis_mode != "full":
                    continue
                results.append(feature_row)
                self.counters.successfully_extracted += 1

        logger.info(self.counters.summary())
        return results

    def save_parquet(
        self,
        rows: List[FeatureRow],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Save feature rows to a Parquet file.

        Args:
            rows: List of FeatureRow objects.
            output_path: Optional output file path.

        Returns:
            Path to the saved Parquet file.
        """
        if not _PYARROW_AVAILABLE:
            raise ImportError("pyarrow is required. pip install pyarrow")

        if output_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "aidev_features.parquet"

        dicts = [r.to_dict() for r in rows]

        # Convert to pyarrow table and save
        table = pa.Table.from_pylist(dicts)
        pq.write_table(table, str(output_path))
        logger.info(f"Dataset saved to {output_path} ({len(rows)} rows)")
        return output_path

    def cleanup(self) -> None:
        """Remove all cloned repositories from disk."""
        if self.repo_cache_dir.exists():
            shutil.rmtree(self.repo_cache_dir, ignore_errors=True)
            logger.info(f"Cleaned up repo cache: {self.repo_cache_dir}")
