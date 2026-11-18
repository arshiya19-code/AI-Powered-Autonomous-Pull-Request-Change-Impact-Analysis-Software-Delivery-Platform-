"""
Repository Manager - Handles cloning, accessing, and managing Git repositories.
Provides a clean interface for repository operations needed by the analysis pipeline.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import git
from git import Repo, GitCommandError, InvalidGitRepositoryError

from config.settings import get_settings
from repository.language_detector import detect_languages, LanguageStats


@dataclass
class RepositoryInfo:
    """Information about a repository."""
    path: Path
    name: str
    owner: str
    full_name: str
    default_branch: str
    current_branch: str
    commit_sha: str
    commit_message: str
    commit_author: str
    commit_date: datetime
    is_shallow: bool
    remote_url: str
    languages: LanguageStats
    total_files: int
    total_lines: int
    size_bytes: int


@dataclass
class CommitInfo:
    """Information about a specific commit."""
    sha: str
    message: str
    author: str
    email: str
    date: datetime
    parents: List[str]
    files_changed: List[str]
    lines_added: int
    lines_deleted: int


class RepositoryManager:
    """
    Manages Git repository operations including cloning, checkout, and metadata extraction.
    """

    def __init__(self, repo_path: Optional[Path] = None, clone_url: Optional[str] = None):
        """
        Initialize repository manager.

        Args:
            repo_path: Path to existing local repository
            clone_url: URL to clone repository from (if repo_path not provided)
        """
        self.settings = get_settings()
        self._repo: Optional[Repo] = None
        self._repo_path: Optional[Path] = None
        self._temp_dir: Optional[Path] = None
        self._repo_info: Optional[RepositoryInfo] = None

        if repo_path:
            self._repo_path = Path(repo_path).resolve()
        elif clone_url:
            self._clone_url = clone_url
        else:
            raise ValueError("Either repo_path or clone_url must be provided")

    def __enter__(self) -> "RepositoryManager":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup if we created a temp clone."""
        self.cleanup()

    def initialize(self) -> Repo:
        """
        Initialize the repository - either open existing or clone new.

        Returns:
            GitPython Repo object
        """
        if self._repo is not None:
            return self._repo

        if self._repo_path and self._repo_path.exists():
            # Open existing repository
            self._repo = self._open_repository(self._repo_path)
        elif hasattr(self, '_clone_url') and self._clone_url:
            # Clone repository
            self._repo, self._repo_path = self._clone_repository(self._clone_url)
        else:
            raise ValueError("No valid repository path or clone URL")

        # Load repository info
        self._repo_info = self._extract_repository_info()
        return self._repo

    def _open_repository(self, path: Path) -> Repo:
        """Open an existing Git repository."""
        try:
            repo = Repo(path)
            if repo.bare:
                raise InvalidGitRepositoryError(f"Repository at {path} is bare")
            return repo
        except InvalidGitRepositoryError as e:
            raise InvalidGitRepositoryError(f"Invalid Git repository at {path}: {e}")

    def _clone_repository(self, clone_url: str) -> tuple[Repo, Path]:
        """Clone a repository to a temporary directory."""
        # Create temporary directory
        self._temp_dir = Path(tempfile.mkdtemp(prefix="ai_impact_repo_"))
        repo_name = clone_url.rstrip("/").split("/")[-1].replace(".git", "")
        clone_path = self._temp_dir / repo_name

        # Prepare clone options
        clone_kwargs = {
            "depth": 1,  # Shallow clone for speed
            "single_branch": True,
        }

        # Add authentication if token provided
        if self.settings.github.token:
            # Insert token into URL for HTTPS authentication
            if clone_url.startswith("https://"):
                clone_url = clone_url.replace("https://", f"https://{self.settings.github.token}@")

        try:
            repo = Repo.clone_from(clone_url, clone_path, **clone_kwargs)
            return repo, clone_path
        except GitCommandError as e:
            # Cleanup on failure
            if self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            raise GitCommandError(f"Failed to clone repository: {e}")

    def _extract_repository_info(self) -> RepositoryInfo:
        """Extract comprehensive repository metadata."""
        if not self._repo:
            raise RuntimeError("Repository not initialized")

        # Get remote URL
        remote_url = ""
        if self._repo.remotes:
            remote_url = self._repo.remotes[0].url

        # Parse owner/repo from remote URL
        owner, name = self._parse_repo_identity(remote_url)

        # Get current commit info
        head_commit = self._repo.head.commit

        # Detect languages
        languages = detect_languages(self._repo_path)

        # Count files and lines
        total_files, total_lines = self._count_files_and_lines()

        # Get repository size
        size_bytes = self._get_repo_size()

        return RepositoryInfo(
            path=self._repo_path,
            name=name,
            owner=owner,
            full_name=f"{owner}/{name}" if owner else name,
            default_branch=self._get_default_branch(),
            current_branch=self._get_current_branch(),
            commit_sha=head_commit.hexsha,
            commit_message=head_commit.message.strip(),
            commit_author=head_commit.author.name,
            commit_date=datetime.fromtimestamp(head_commit.committed_date),
            is_shallow=self._is_shallow(),
            remote_url=remote_url,
            languages=languages,
            total_files=total_files,
            total_lines=total_lines,
            size_bytes=size_bytes
        )

    def _parse_repo_identity(self, remote_url: str) -> tuple[str, str]:
        """Extract owner and repo name from remote URL."""
        owner = ""
        name = ""

        # Handle various URL formats
        # https://github.com/owner/repo.git
        # git@github.com:owner/repo.git
        # https://token@github.com/owner/repo.git
        import re

        patterns = [
            r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?",
            r"github\.com[:/]([^/]+)/([^/.]+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, remote_url)
            if match:
                owner, name = match.groups()
                break

        return owner, name

    def _get_default_branch(self) -> str:
        """Get the default branch name."""
        try:
            # Try to get from remote HEAD
            if self._repo.remotes:
                origin = self._repo.remotes.origin
                refs = origin.refs
                for ref in refs:
                    if ref.name == "origin/HEAD":
                        return ref.ref.name.replace("origin/", "")
        except Exception:
            pass

        # Fallback to common defaults
        for branch in ["main", "master", "develop"]:
            try:
                self._repo.refs[branch]
                return branch
            except IndexError:
                continue

        # Last resort: current branch
        return self._get_current_branch()

    def _get_current_branch(self) -> str:
        """Get the current branch name."""
        try:
            return self._repo.active_branch.name
        except TypeError:
            # Detached HEAD state
            return f"detached-{self._repo.head.commit.hexsha[:8]}"

    def _is_shallow(self) -> bool:
        """Check if repository is shallow."""
        try:
            git_dir = Path(self._repo.git_dir)
            shallow_file = git_dir / "shallow"
            return shallow_file.exists()
        except Exception:
            return False

    def _count_files_and_lines(self) -> tuple[int, int]:
        """Count total files and lines in the repository."""
        total_files = 0
        total_lines = 0

        exclude_dirs = set(self.settings.analysis.exclude_dirs)
        analyze_extensions = set(self.settings.analysis.analyze_extensions)

        for file_path in self._repo_path.rglob("*"):
            # Skip excluded directories
            if any(part in exclude_dirs for part in file_path.parts):
                continue

            if file_path.is_file() and file_path.suffix in analyze_extensions:
                # Check file size
                try:
                    if file_path.stat().st_size > self.settings.analysis.max_file_size_kb * 1024:
                        continue
                except OSError:
                    continue

                total_files += 1
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        total_lines += sum(1 for _ in f)
                except Exception:
                    pass

        return total_files, total_lines

    def _get_repo_size(self) -> int:
        """Get repository size in bytes."""
        total_size = 0
        try:
            for file_path in self._repo_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size

    @property
    def repo(self) -> Repo:
        """Get the GitPython Repo object."""
        if self._repo is None:
            self.initialize()
        return self._repo

    @property
    def repo_path(self) -> Path:
        """Get the repository path."""
        if self._repo_path is None:
            self.initialize()
        return self._repo_path

    @property
    def repo_info(self) -> RepositoryInfo:
        """Get repository information."""
        if self._repo_info is None:
            self.initialize()
        return self._repo_info

    def get_commit_info(self, commit_sha: str) -> CommitInfo:
        """
        Get detailed information about a specific commit.

        Args:
            commit_sha: Commit SHA to analyze

        Returns:
            CommitInfo object
        """
        commit = self._repo.commit(commit_sha)

        # Get diff stats
        lines_added = 0
        lines_deleted = 0
        files_changed = []

        if commit.parents:
            parent = commit.parents[0]
            diffs = parent.diff(commit)
            for diff in diffs:
                files_changed.append(diff.a_path or diff.b_path)
                # Get detailed stats
                try:
                    stats = diff.diff.decode("utf-8", errors="ignore")
                    for line in stats.split("\n"):
                        if line.startswith("+") and not line.startswith("+++"):
                            lines_added += 1
                        elif line.startswith("-") and not line.startswith("---"):
                            lines_deleted += 1
                except Exception:
                    pass
        else:
            # Initial commit - all files are additions
            for item in commit.tree.traverse():
                if item.type == "blob":
                    files_changed.append(item.path)
                    try:
                        content = item.data_stream.read().decode("utf-8", errors="ignore")
                        lines_added += content.count("\n")
                    except Exception:
                        pass

        return CommitInfo(
            sha=commit.hexsha,
            message=commit.message.strip(),
            author=commit.author.name,
            email=commit.author.email,
            date=datetime.fromtimestamp(commit.committed_date),
            parents=[p.hexsha for p in commit.parents],
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted
        )

    def get_diff(self, base_sha: str, head_sha: str) -> List[Dict[str, Any]]:
        """
        Get detailed diff between two commits.

        Args:
            base_sha: Base commit SHA
            head_sha: Head commit SHA

        Returns:
            List of file diffs with detailed changes
        """
        base_commit = self._repo.commit(base_sha)
        head_commit = self._repo.commit(head_sha)

        diffs = base_commit.diff(head_commit)
        result = []

        for diff in diffs:
            file_info = {
                "path": diff.a_path or diff.b_path,
                "old_path": diff.a_path,
                "new_path": diff.b_path,
                "change_type": diff.change_type,  # A, D, M, R, C, T
                "is_binary": diff.diff is not None and b'\x00' in diff.diff,
                "diff": diff.diff.decode("utf-8", errors="ignore") if diff.diff else "",
            }

            # Parse diff for line changes
            if file_info["diff"]:
                lines_added = 0
                lines_deleted = 0
                for line in file_info["diff"].split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        lines_added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        lines_deleted += 1
                file_info["lines_added"] = lines_added
                file_info["lines_deleted"] = lines_deleted

            result.append(file_info)

        return result

    def checkout(self, ref: str) -> bool:
        """
        Checkout a specific branch, tag, or commit.

        Args:
            ref: Branch name, tag, or commit SHA

        Returns:
            True if successful
        """
        try:
            self._repo.git.checkout(ref)
            # Refresh repo info after checkout
            self._repo_info = self._extract_repository_info()
            return True
        except GitCommandError as e:
            raise GitCommandError(f"Failed to checkout {ref}: {e}")

    def get_file_content(self, file_path: str, commit_sha: Optional[str] = None) -> str:
        """
        Get file content at a specific commit.

        Args:
            file_path: Path to file relative to repo root
            commit_sha: Commit SHA (default: current HEAD)

        Returns:
            File content as string
        """
        if commit_sha:
            commit = self._repo.commit(commit_sha)
            try:
                blob = commit.tree / file_path
                return blob.data_stream.read().decode("utf-8", errors="ignore")
            except Exception:
                return ""
        else:
            full_path = self._repo_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
                except Exception:
                    return ""
            return ""

    def list_files(self, extension: Optional[str] = None,
                   exclude_dirs: Optional[List[str]] = None) -> List[Path]:
        """
        List files in the repository.

        Args:
            extension: Filter by file extension (e.g., '.py')
            exclude_dirs: Additional directories to exclude

        Returns:
            List of file paths relative to repo root
        """
        exclude = set(self.settings.analysis.exclude_dirs)
        if exclude_dirs:
            exclude.update(exclude_dirs)

        files = []
        for file_path in self._repo_path.rglob("*"):
            if any(part in exclude for part in file_path.parts):
                continue
            if file_path.is_file():
                if extension is None or file_path.suffix == extension:
                    files.append(file_path.relative_to(self._repo_path))

        return sorted(files)

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None
            self._repo = None
            self._repo_path = None
            self._repo_info = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def open_repository(path: Path) -> RepositoryManager:
    """
    Convenience function to open an existing repository.

    Args:
        path: Path to repository

    Returns:
        Initialized RepositoryManager
    """
    manager = RepositoryManager(repo_path=path)
    manager.initialize()
    return manager


def clone_repository(url: str) -> RepositoryManager:
    """
    Convenience function to clone a repository.

    Args:
        url: Repository clone URL

    Returns:
        Initialized RepositoryManager
    """
    manager = RepositoryManager(clone_url=url)
    manager.initialize()
    return manager