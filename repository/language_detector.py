"""
Language Detector - Detects programming languages in a repository
based on file extensions and content analysis.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import json


# Language extension mapping
LANGUAGE_EXTENSIONS: Dict[str, Set[str]] = {
    "Python": {".py", ".pyw", ".pyi", ".pyx", ".pxd", ".pxi"},
    "JavaScript": {".js", ".jsx", ".mjs", ".cjs"},
    "TypeScript": {".ts", ".tsx"},
    "Java": {".java"},
    "C#": {".cs"},
    "C++": {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++"},
    "C": {".c", ".h"},
    "Go": {".go"},
    "Rust": {".rs"},
    "Ruby": {".rb", ".rbw"},
    "PHP": {".php", ".phtml", ".php3", ".php4", ".php5", ".phps"},
    "Swift": {".swift"},
    "Kotlin": {".kt", ".kts"},
    "Scala": {".scala", ".sc"},
    "R": {".r", ".R"},
    "MATLAB": {".m"},
    "Julia": {".jl"},
    "Lua": {".lua"},
    "Perl": {".pl", ".pm", ".pod"},
    "Shell": {".sh", ".bash", ".zsh", ".fish"},
    "PowerShell": {".ps1", ".psm1", ".psd1"},
    "HTML": {".html", ".htm", ".xhtml"},
    "CSS": {".css", ".scss", ".sass", ".less"},
    "SQL": {".sql"},
    "YAML": {".yaml", ".yml"},
    "JSON": {".json"},
    "TOML": {".toml"},
    "XML": {".xml", ".xsd", ".xsl"},
    "Markdown": {".md", ".markdown", ".mdown", ".mkd"},
    "Dockerfile": {"Dockerfile", "dockerfile", ".dockerfile"},
    "Makefile": {"Makefile", "makefile", "GNUmakefile"},
    "CMake": {".cmake", "CMakeLists.txt"},
    "Gradle": {".gradle"},
    "Maven": {"pom.xml"},
    "Bazel": {"BUILD", "BUILD.bazel", ".bzl"},
}


# Shebang patterns for language detection
SHEBANG_PATTERNS: Dict[str, str] = {
    "Python": r"python",
    "Node.js": r"node",
    "Ruby": r"ruby",
    "Perl": r"perl",
    "Bash": r"bash",
    "Shell": r"sh",
    "PHP": r"php",
    "Lua": r"lua",
    "Rust": r"rust",
}


@dataclass
class LanguageStats:
    """Statistics for a detected language."""
    name: str
    files: int = 0
    lines: int = 0
    bytes: int = 0
    percentage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "files": self.files,
            "lines": self.lines,
            "bytes": self.bytes,
            "percentage": round(self.percentage, 2)
        }


class LanguageDetector:
    """
    Detects programming languages in a repository by analyzing
    file extensions, shebangs, and file content.
    """

    def __init__(self, repo_path: Path, exclude_dirs: Optional[List[str]] = None):
        """
        Initialize language detector.

        Args:
            repo_path: Path to repository root
            exclude_dirs: Directories to exclude from analysis
        """
        self.repo_path = Path(repo_path)
        self.exclude_dirs = set(exclude_dirs or [
            "__pycache__", ".git", "venv", "env", ".venv", ".env",
            "node_modules", "dist", "build", ".pytest_cache",
            ".mypy_cache", "htmlcov", ".coverage", ".idea", ".vscode"
        ])
        self._stats: Dict[str, LanguageStats] = {}
        self._total_files = 0
        self._total_lines = 0
        self._total_bytes = 0

    def detect(self) -> Dict[str, LanguageStats]:
        """
        Detect languages in the repository.

        Returns:
            Dictionary mapping language name to LanguageStats
        """
        self._stats = {}
        self._total_files = 0
        self._total_lines = 0
        self._total_bytes = 0

        # First pass: count by extension
        self._count_by_extension()

        # Second pass: refine with shebang/content analysis for ambiguous files
        self._refine_with_content()

        # Calculate percentages
        self._calculate_percentages()

        return self._stats

    def _count_by_extension(self) -> None:
        """Count files, lines, and bytes by file extension."""
        extension_to_lang = {}
        for lang, extensions in LANGUAGE_EXTENSIONS.items():
            for ext in extensions:
                extension_to_lang[ext.lower()] = lang

        for file_path in self.repo_path.rglob("*"):
            if not self._should_process_file(file_path):
                continue

            try:
                stat = file_path.stat()
                size = stat.st_size
                lines = self._count_lines(file_path)
            except (OSError, PermissionError):
                continue

            # Determine language from extension
            lang = None
            suffix = file_path.suffix.lower()

            # Check exact filename matches (e.g., Dockerfile, Makefile)
            for language, extensions in LANGUAGE_EXTENSIONS.items():
                if file_path.name in extensions:
                    lang = language
                    break

            # Check extension
            if not lang and suffix in extension_to_lang:
                lang = extension_to_lang[suffix]

            if lang:
                self._add_file_stats(lang, lines, size)
            else:
                # Unknown extension - categorize as "Other"
                self._add_file_stats("Other", lines, size)

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        if not file_path.is_file():
            return False

        # Skip excluded directories
        for part in file_path.parts:
            if part in self.exclude_dirs:
                return False

        # Skip hidden files
        if file_path.name.startswith("."):
            return False

        # Skip very large files (> 1MB)
        try:
            if file_path.stat().st_size > 1_000_000:
                return False
        except OSError:
            return False

        return True

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _add_file_stats(self, language: str, lines: int, bytes_: int) -> None:
        """Add file statistics to language totals."""
        if language not in self._stats:
            self._stats[language] = LanguageStats(name=language)

        stats = self._stats[language]
        stats.files += 1
        stats.lines += lines
        stats.bytes += bytes_

        self._total_files += 1
        self._total_lines += lines
        self._total_bytes += bytes_

    def _refine_with_content(self) -> None:
        """Refine language detection using file content (shebangs, etc.)."""
        # This could be expanded to analyze file content for better detection
        # For now, extension-based detection is sufficient
        pass

    def _calculate_percentages(self) -> None:
        """Calculate percentage for each language."""
        if self._total_bytes == 0:
            return

        for stats in self._stats.values():
            stats.percentage = (stats.bytes / self._total_bytes) * 100

    def get_primary_language(self) -> Optional[str]:
        """Get the primary language by bytes."""
        if not self._stats:
            return None
        return max(self._stats.values(), key=lambda s: s.bytes).name

    def get_language_summary(self) -> Dict[str, any]:
        """Get a summary of language statistics."""
        return {
            "total_files": self._total_files,
            "total_lines": self._total_lines,
            "total_bytes": self._total_bytes,
            "languages": {name: stats.to_dict() for name, stats in self._stats.items()},
            "primary_language": self.get_primary_language()
        }

    def to_json(self) -> str:
        """Export statistics as JSON."""
        return json.dumps(self.get_language_summary(), indent=2)


def detect_languages(repo_path: Path, exclude_dirs: Optional[List[str]] = None) -> LanguageStats:
    """
    Convenience function to detect languages and return aggregate stats.

    Args:
        repo_path: Path to repository
        exclude_dirs: Directories to exclude

    Returns:
        LanguageStats for the primary language
    """
    detector = LanguageDetector(repo_path, exclude_dirs)
    stats = detector.detect()
    primary = detector.get_primary_language()
    return stats.get(primary, LanguageStats(name="Unknown")) if primary else LanguageStats(name="Unknown")


def get_language_stats(repo_path: Path, exclude_dirs: Optional[List[str]] = None) -> Dict[str, LanguageStats]:
    """
    Get detailed language statistics for all detected languages.

    Args:
        repo_path: Path to repository
        exclude_dirs: Directories to exclude

    Returns:
        Dictionary mapping language name to LanguageStats
    """
    detector = LanguageDetector(repo_path, exclude_dirs)
    return detector.detect()