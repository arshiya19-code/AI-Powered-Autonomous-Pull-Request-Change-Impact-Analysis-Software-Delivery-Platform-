"""
Language Abstraction Layer — Phase 7 Public API.

This package provides the multi-language architecture foundation:

    LanguageAdapter     Protocol that all language adapters must satisfy.
    ParsedSource        Language-neutral parsed file representation.
    Symbol              Language-neutral symbol (class/function/method/module).
    Dependency          Language-neutral dependency edge.
    LanguageDependencyType  Enum of dependency relationship types.
    SymbolKind          Enum of symbol kinds.
    SourceLocation      Immutable language-neutral source location.
    ProjectContext      Project-wide context for symbol resolution.

    DetectedLanguage    Result of language detection for a single file.
    LanguageDetector    Detects language from file extension.

    LanguageRegistry    Registry of registered language adapters.
    DuplicateAdapterError   Raised on conflicting adapter registration.
    UnsupportedLanguageError  Raised when no adapter is available.

    default_registry    Module-level singleton pre-registered with Python.

    PythonLanguageAdapter   Language Adapter #1 — wraps Phase 1-3 pipeline.

Currently supported languages
------------------------------
  python  — full adapter (ASTParser + DependencyExtractor + NameResolver)

Detected but not yet implemented
----------------------------------
  java, javascript, typescript, go, c, cpp (detected; adapters planned)
"""

from analyzer.languages.base import (
    Dependency,
    LanguageAdapter,
    LanguageDependencyType,
    ParsedSource,
    ProjectContext,
    SourceLocation,
    Symbol,
    SymbolKind,
)

from analyzer.languages.detector import (
    DetectedLanguage,
    LanguageDetector,
)

from analyzer.languages.registry import (
    DuplicateAdapterError,
    LanguageRegistry,
    UnsupportedLanguageError,
    default_registry,
)

from analyzer.languages.python.adapter import PythonLanguageAdapter
from analyzer.languages.structured import (
    CppLanguageAdapter, CSharpLanguageAdapter, JavaScriptLanguageAdapter,
    RustLanguageAdapter, TypeScriptLanguageAdapter,
)

__all__ = [
    # Base types
    "LanguageAdapter",
    "ParsedSource",
    "Symbol",
    "SymbolKind",
    "Dependency",
    "LanguageDependencyType",
    "SourceLocation",
    "ProjectContext",
    # Detection
    "DetectedLanguage",
    "LanguageDetector",
    # Registry
    "LanguageRegistry",
    "DuplicateAdapterError",
    "UnsupportedLanguageError",
    "default_registry",
    # Adapters
    "PythonLanguageAdapter",
    "TypeScriptLanguageAdapter",
    "JavaScriptLanguageAdapter",
    "CSharpLanguageAdapter",
    "RustLanguageAdapter",
    "CppLanguageAdapter",
]
