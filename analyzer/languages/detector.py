"""
Language Detector (Phase 7).

Detects the programming language of a source file from its extension.

Detection does NOT imply that an adapter is available for that language.
Call ``LanguageRegistry.is_supported(language)`` to check adapter availability.

Supported detection mappings
-----------------------------
.py           → python
.java         → java
.js           → javascript
.jsx          → javascript
.ts           → typescript
.tsx          → typescript
.go           → go
.c            → c
.h            → c
.cpp          → cpp
.hpp          → cpp
.cc           → cpp
.cxx          → cpp
.rb           → ruby
.rs           → rust
.swift        → swift
.kt           → kotlin
.kts          → kotlin
.cs           → csharp
.php          → php
.r, .R        → r
.scala        → scala
(unknown)     → unknown
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ============================================================================
# Extension mapping
# ============================================================================

_EXTENSION_MAP: Dict[str, str] = {
    ".py":    "python",
    ".pyw":   "python",
    ".java":  "java",
    ".js":    "javascript",
    ".jsx":   "javascript",
    ".mjs":   "javascript",
    ".cjs":   "javascript",
    ".ts":    "typescript",
    ".tsx":   "typescript",
    ".mts":   "typescript",
    ".cts":   "typescript",
    ".go":    "go",
    ".c":     "c",
    ".h":     "c",
    ".cpp":   "cpp",
    ".hpp":   "cpp",
    ".cc":    "cpp",
    ".cxx":   "cpp",
    ".hh":    "cpp",
    ".hxx":   "cpp",
    ".rb":    "ruby",
    ".rs":    "rust",
    ".swift": "swift",
    ".kt":    "kotlin",
    ".kts":   "kotlin",
    ".cs":    "csharp",
    ".php":   "php",
    ".r":     "r",
    ".R":     "r",
    ".scala": "scala",
}

# ============================================================================
# Result dataclass
# ============================================================================

@dataclass(frozen=True)
class DetectedLanguage:
    """
    Result of language detection for a single file.

    Attributes
    ----------
    file_path    : The file that was analysed.
    language     : Lower-case language identifier, e.g. ``"python"``, or
                   ``"unknown"`` when the extension is not recognised.
    is_supported : Whether a language adapter is registered for this language.
                   Always ``False`` until checked against a registry.
    extension    : The file extension that triggered the detection (with dot).
    """
    file_path:    str
    language:     str
    is_supported: bool
    extension:    str

    def __str__(self) -> str:
        support = "supported" if self.is_supported else "not supported"
        return f"DetectedLanguage({self.file_path!r} → {self.language!r} [{support}])"


# ============================================================================
# Detector
# ============================================================================

class LanguageDetector:
    """
    Detects the programming language of source files from their extension.

    The detector is intentionally simple and relies solely on file extensions.
    Shebang-line or content-based detection may be added in a future phase.

    Usage::

        detector = LanguageDetector()
        result = detector.detect("services/payment.py")
        # DetectedLanguage(file_path='services/payment.py',
        #                  language='python', is_supported=False, extension='.py')

    Note: ``is_supported`` is always ``False`` from the detector alone.
    Use ``LanguageRegistry.mark_supported(result)`` or call
    ``registry.is_supported(result.language)`` to check real support.
    """

    def __init__(self) -> None:
        self._extension_map: Dict[str, str] = dict(_EXTENSION_MAP)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, path: str) -> DetectedLanguage:
        """
        Detect the language of a single file.

        Args:
            path: File path (does not need to exist on disk).

        Returns:
            ``DetectedLanguage`` with ``is_supported=False`` (registry
            must be consulted separately for support status).
        """
        ext = Path(path).suffix          # e.g. ".py"
        language = self._extension_map.get(ext, "unknown")
        return DetectedLanguage(
            file_path=path,
            language=language,
            is_supported=False,          # caller must check against registry
            extension=ext,
        )

    def detect_many(self, paths: List[str]) -> Dict[str, DetectedLanguage]:
        """
        Detect languages for a list of file paths.

        Args:
            paths: List of file paths.

        Returns:
            Dict mapping each path to its ``DetectedLanguage``.
        """
        return {p: self.detect(p) for p in paths}

    def detect_with_support(
        self,
        path: str,
        supported_languages: List[str],
    ) -> DetectedLanguage:
        """
        Detect language and immediately resolve support status.

        Args:
            path:                File path.
            supported_languages: List of language identifiers that have
                                 registered adapters.

        Returns:
            ``DetectedLanguage`` with ``is_supported`` set correctly.
        """
        result = self.detect(path)
        return DetectedLanguage(
            file_path=result.file_path,
            language=result.language,
            is_supported=result.language in supported_languages,
            extension=result.extension,
        )

    def known_extensions(self) -> List[str]:
        """Return all recognised file extensions (with dot)."""
        return sorted(self._extension_map.keys())

    def known_languages(self) -> List[str]:
        """Return all distinct language identifiers the detector can produce."""
        return sorted(set(self._extension_map.values()))
