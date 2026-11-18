"""
Repository Scanner - Scans repository for source files and extracts
basic metadata needed for analysis.
"""

import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import fnmatch

from config.settings import get_settings


@dataclass
class FileMetadata:
    """Metadata for a source file."""
    path: Path
    relative_path: Path
    size_bytes: int
    lines: int
    extension: str
    language: str
    encoding: str = "utf-8"
    is_test: bool = False
    is_config: bool = False
    is_generated: bool = False


@dataclass
class ScanResult:
    """Result of repository scan."""
    files: List[FileMetadata]
    total_files: int
    total_lines: int
    total_bytes: int
    languages: Dict[str, int]  # language -> file count
    by_extension: Dict[str, int]  # extension -> file count
    test_files: List[FileMetadata]
    config_files: List[FileMetadata]
    excluded_count: int


class RepositoryScanner:
    """
    Scans a repository to discover source files and collect metadata.
    Supports filtering by language, directory, and file patterns.
    """

    # Test file patterns
    TEST_PATTERNS = [
        "*test*.py",
        "*_test.py",
        "test_*.py",
        "*_spec.py",
        "spec_*.py",
        "conftest.py",
        "pytest.ini",
        "tox.ini",
    ]

    # Config file patterns
    CONFIG_PATTERNS = [
        "*.ini", "*.cfg", "*.conf", "*.config",
        "*.yaml", "*.yml",
        "*.toml",
        "*.json",
        "*.xml",
        "requirements*.txt",
        "setup.py", "setup.cfg", "pyproject.toml",
        "Pipfile", "Pipfile.lock",
        "poetry.lock",
        ".env*", "*.env",
        "Dockerfile*", "docker-compose*.yml", "docker-compose*.yaml",
        "Makefile", "makefile",
        "CMakeLists.txt", "*.cmake",
        "*.gradle", "pom.xml",
        "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "Cargo.toml", "Cargo.lock",
        "go.mod", "go.sum",
    ]

    # Generated file patterns
    GENERATED_PATTERNS = [
        "*_pb2.py", "*_pb2_grpc.py",  # Protocol buffers
        "*.generated.py",
        "*.min.js", "*.min.css",
        "__init__.py",  # Often empty/generated
    ]

    def __init__(self, repo_path: Path, exclude_dirs: Optional[List[str]] = None,
                 extensions: Optional[List[str]] = None, max_file_size_kb: Optional[int] = None):
        """
        Initialize repository scanner.

        Args:
            repo_path: Path to repository root
            exclude_dirs: Directories to exclude (uses settings default if None)
            extensions: File extensions to include (uses settings default if None)
            max_file_size_kb: Maximum file size to process (uses settings default if None)
        """
        self.repo_path = Path(repo_path).resolve()
        self.settings = get_settings()

        self.exclude_dirs = set(exclude_dirs or self.settings.analysis.exclude_dirs)
        self.include_extensions = set(ext.lower() for ext in (extensions or self.settings.analysis.analyze_extensions))
        self.max_file_size = (max_file_size_kb or self.settings.analysis.max_file_size_kb) * 1024

        # Results
        self._files: List[FileMetadata] = []
        self._test_files: List[FileMetadata] = []
        self._config_files: List[FileMetadata] = []
        self._excluded_count = 0

    def scan(self) -> ScanResult:
        """
        Scan the repository and collect file metadata.

        Returns:
            ScanResult with all discovered files and statistics
        """
        self._files = []
        self._test_files = []
        self._config_files = []
        self._excluded_count = 0

        for file_path in self._iterate_files():
            metadata = self._extract_metadata(file_path)
            if metadata:
                self._files.append(metadata)
                self._categorize_file(metadata)

        return self._build_result()

    def _iterate_files(self) -> Iterator[Path]:
        """Iterate over all files in repository."""
        for file_path in self.repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check excluded directories
            if self._is_excluded(file_path):
                self._excluded_count += 1
                continue

            # Check file size
            try:
                if file_path.stat().st_size > self.max_file_size:
                    self._excluded_count += 1
                    continue
            except OSError:
                self._excluded_count += 1
                continue

            # Check extension
            if self.include_extensions and file_path.suffix.lower() not in self.include_extensions:
                self._excluded_count += 1
                continue

            yield file_path

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if file is in an excluded directory."""
        try:
            rel_path = file_path.relative_to(self.repo_path)
            for part in rel_path.parts[:-1]:  # Exclude filename itself
                if part in self.exclude_dirs:
                    return True
        except ValueError:
            return True
        return False

    def _extract_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Extract metadata from a file."""
        try:
            stat = file_path.stat()
            size = stat.st_size
            rel_path = file_path.relative_to(self.repo_path)

            # Count lines
            lines = self._count_lines(file_path)

            # Detect encoding
            encoding = self._detect_encoding(file_path)

            # Determine language from extension
            language = self._detect_language(file_path)

            # Check if test file
            is_test = self._is_test_file(file_path)

            # Check if config file
            is_config = self._is_config_file(file_path)

            # Check if generated
            is_generated = self._is_generated_file(file_path)

            return FileMetadata(
                path=file_path,
                relative_path=rel_path,
                size_bytes=size,
                lines=lines,
                extension=file_path.suffix.lower(),
                language=language,
                encoding=encoding,
                is_test=is_test,
                is_config=is_config,
                is_generated=is_generated
            )
        except Exception as e:
            # Log error but continue scanning
            print(f"Warning: Failed to process {file_path}: {e}")
            return None

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw)
                return result.get("encoding", "utf-8") or "utf-8"
        except Exception:
            return "utf-8"

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "Python", ".pyw": "Python", ".pyi": "Python",
            ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
            ".ts": "TypeScript", ".tsx": "TypeScript",
            ".java": "Java",
            ".cs": "C#",
            ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++",
            ".c": "C", ".h": "C",
            ".go": "Go",
            ".rs": "Rust",
            ".rb": "Ruby",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin", ".kts": "Kotlin",
            ".scala": "Scala",
            ".r": "R", ".R": "R",
            ".lua": "Lua",
            ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
            ".ps1": "PowerShell",
        }
        return extension_map.get(file_path.suffix.lower(), "Unknown")

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        name = file_path.name.lower()
        for pattern in self.TEST_PATTERNS:
            if fnmatch.fnmatch(name, pattern.lower()):
                return True

        # Check if in test directory
        for part in file_path.parts:
            if part.lower() in ("test", "tests", "testing", "spec", "specs"):
                return True

        return False

    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file."""
        name = file_path.name.lower()
        for pattern in self.CONFIG_PATTERNS:
            if fnmatch.fnmatch(name, pattern.lower()):
                return True
        return False

    def _is_generated_file(self, file_path: Path) -> bool:
        """Check if file appears to be generated."""
        name = file_path.name.lower()
        for pattern in self.GENERATED_PATTERNS:
            if fnmatch.fnmatch(name, pattern.lower()):
                return True
        return False

    def _categorize_file(self, metadata: FileMetadata) -> None:
        """Categorize file into test, config, or regular."""
        if metadata.is_test:
            self._test_files.append(metadata)
        elif metadata.is_config:
            self._config_files.append(metadata)

    def _build_result(self) -> ScanResult:
        """Build scan result from collected data."""
        languages = defaultdict(int)
        by_extension = defaultdict(int)

        for meta in self._files:
            languages[meta.language] += 1
            by_extension[meta.extension] += 1

        return ScanResult(
            files=self._files,
            total_files=len(self._files),
            total_lines=sum(f.lines for f in self._files),
            total_bytes=sum(f.size_bytes for f in self._files),
            languages=dict(languages),
            by_extension=dict(by_extension),
            test_files=self._test_files,
            config_files=self._config_files,
            excluded_count=self._excluded_count
        )

    def get_source_files(self, include_tests: bool = False) -> List[FileMetadata]:
        """Get source files, optionally including test files."""
        if include_tests:
            return self._files
        return [f for f in self._files if not f.is_test and not f.is_config and not f.is_generated]

    def get_files_by_language(self, language: str) -> List[FileMetadata]:
        """Get files for a specific language."""
        return [f for f in self._files if f.language == language]

    def get_python_files(self, include_tests: bool = False) -> List[FileMetadata]:
        """Get Python source files."""
        return self.get_files_by_language("Python")


def scan_repository(repo_path: Path, **kwargs) -> ScanResult:
    """
    Convenience function to scan a repository.

    Args:
        repo_path: Path to repository
        **kwargs: Additional arguments for RepositoryScanner

    Returns:
        ScanResult
    """
    scanner = RepositoryScanner(repo_path, **kwargs)
    return scanner.scan()


def get_python_files(repo_path: Path, include_tests: bool = False) -> List[FileMetadata]:
    """
    Convenience function to get Python files from a repository.

    Args:
        repo_path: Path to repository
        include_tests: Whether to include test files

    Returns:
        List of Python FileMetadata
    """
    scanner = RepositoryScanner(repo_path)
    scanner.scan()
    return scanner.get_python_files(include_tests=include_tests)