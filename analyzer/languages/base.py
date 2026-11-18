"""
Language Abstraction Layer — Base Types (Phase 7).

Defines the language-neutral intermediate representations and the
``LanguageAdapter`` Protocol that all language implementations must satisfy.

Design principles
-----------------
* ``ParsedSource`` replaces language-specific types (e.g. ``ParsedModule``)
  as the currency between the language layer and the core pipeline.
* ``LanguageAdapter`` is a ``typing.Protocol`` so that existing classes can
  satisfy it via structural subtyping without inheriting from anything.
* All types are mypy ``--strict`` compatible.
* No NetworkX, Neo4j, or external graph libraries.
* No AST types from any language are exposed through this module.

Unified representation hierarchy
---------------------------------
ParsedSource
    ├── file_path: str
    ├── language: str
    ├── symbols: List[Symbol]
    ├── dependencies: List[Dependency]
    ├── errors: List[str]
    └── metadata: Dict[str, Any]

Symbol
    ├── name, qualified_name, symbol_type
    ├── file_path, start_line, end_line
    ├── language
    └── metadata

Dependency
    ├── source, target, dep_type
    ├── source_file, source_line
    ├── confidence, language
    └── metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ============================================================================
# Language-neutral enumerations
# ============================================================================

class LanguageDependencyType(Enum):
    """
    Language-neutral dependency relationship types.

    These map to language-specific concepts:
        IMPORTS     Python ``import`` / Java ``import`` / JS ``require``/``import``
        CALLS       Function/method call
        INHERITS    Class inheritance (Python, Java, JS/TS, C++)
        ANNOTATES   Python decorator / Java annotation / TS decorator
        REFERENCES  Generic module or symbol reference
    """
    IMPORTS    = "imports"
    CALLS      = "calls"
    INHERITS   = "inherits"
    ANNOTATES  = "annotates"
    REFERENCES = "references"


class SymbolKind(Enum):
    """
    Language-neutral symbol kinds.

    MODULE   – a file or package
    CLASS    – a class or interface or struct
    FUNCTION – a standalone function
    METHOD   – a function defined inside a class
    VARIABLE – a module-level variable / constant
    """
    MODULE   = "module"
    CLASS    = "class"
    FUNCTION = "function"
    METHOD   = "method"
    VARIABLE = "variable"


# ============================================================================
# Unified intermediate representations
# ============================================================================

@dataclass(frozen=True)
class SourceLocation:
    """
    Language-neutral source code location.

    All line and column numbers are 1-indexed.
    ``end_col=0`` means the end column is unknown.
    """
    file_path:    str
    start_line:   int
    end_line:     int
    start_col:    int = 0
    end_col:      int = 0

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class Symbol:
    """
    A language-neutral symbol (class, function, method, module, variable).

    ``qualified_name`` is a dotted identifier that is unique within a project.
    For Python ``services.payment.PaymentService.process`` is an example.
    For Java it would follow the same dotted convention.

    ``language`` stores the originating language (e.g. ``"python"``).
    ``metadata`` holds any language-specific extra information.
    """
    name:           str
    qualified_name: str
    symbol_kind:    SymbolKind
    file_path:      str
    start_line:     int = 0
    end_line:       int = 0
    language:       str = ""
    parent:         Optional[str] = None   # qualified_name of enclosing symbol
    is_public:      bool = True
    metadata:       Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Symbol({self.symbol_kind.value}:{self.qualified_name})"


@dataclass
class Dependency:
    """
    A language-neutral dependency edge between two symbols or modules.

    ``source`` and ``target`` are qualified names (matching ``Symbol.qualified_name``).
    ``dep_type`` is a ``LanguageDependencyType``.
    ``confidence`` is in [0.0, 1.0]; 1.0 means statically certain.
    ``language`` identifies which adapter produced this dependency.
    ``metadata`` holds any language-specific extra information.
    """
    source:       str
    target:       str
    dep_type:     LanguageDependencyType
    source_file:  str
    source_line:  int = 0
    confidence:   float = 1.0
    language:     str = ""
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Dependency({self.source} "
            f"--[{self.dep_type.value}]--> "
            f"{self.target})"
        )


@dataclass
class ParsedSource:
    """
    Language-neutral parsed representation of a single source file.

    Produced by ``LanguageAdapter.parse()``.
    Consumed by the core dependency/graph pipeline.

    ``symbols``      – all symbols defined in this file
    ``dependencies`` – raw (unresolved) dependency edges from this file
    ``errors``       – any parse errors (non-empty means partial/failed parse)
    ``metadata``     – adapter-specific extra data (e.g. source_lines count)
    """
    file_path:    str
    language:     str
    symbols:      List[Symbol]      = field(default_factory=list)
    dependencies: List[Dependency]  = field(default_factory=list)
    errors:       List[str]         = field(default_factory=list)
    metadata:     Dict[str, Any]    = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def module_symbols(self) -> List[Symbol]:
        return [s for s in self.symbols if s.symbol_kind == SymbolKind.MODULE]

    @property
    def class_symbols(self) -> List[Symbol]:
        return [s for s in self.symbols if s.symbol_kind == SymbolKind.CLASS]

    @property
    def function_symbols(self) -> List[Symbol]:
        return [s for s in self.symbols if s.symbol_kind == SymbolKind.FUNCTION]

    @property
    def method_symbols(self) -> List[Symbol]:
        return [s for s in self.symbols if s.symbol_kind == SymbolKind.METHOD]

    def __repr__(self) -> str:
        return (
            f"ParsedSource(language={self.language!r}, "
            f"file={self.file_path!r}, "
            f"symbols={len(self.symbols)}, "
            f"deps={len(self.dependencies)}, "
            f"errors={len(self.errors)})"
        )


@dataclass
class ProjectContext:
    """
    Language-neutral context for a whole project.

    Passed to ``LanguageAdapter.resolve_symbols()`` so adapters can
    perform cross-file symbol resolution.

    ``root``            – absolute path to the project root
    ``all_sources``     – all ``ParsedSource`` objects in the project
    ``language_hints``  – optional per-file language overrides
    """
    root:           str
    all_sources:    List[ParsedSource]  = field(default_factory=list)
    language_hints: Dict[str, str]      = field(default_factory=dict)  # file_path → language


# ============================================================================
# Language Adapter Protocol
# ============================================================================

@runtime_checkable
class LanguageAdapter(Protocol):
    """
    Protocol that all language adapters must satisfy.

    Implementations should be stateless (or thread-safe) so they can be
    shared across calls.  The adapter is responsible for:

    1. Reporting which files it supports (``supports_file``).
    2. Parsing source text into a ``ParsedSource`` (``parse``).
    3. Extracting raw (unresolved) dependencies (``extract_dependencies``).
    4. Resolving symbols cross-file (``resolve_symbols``).

    The graph engine, change analyzer, and impact scorer are **not** aware
    of language-specific types; they operate on ``ParsedSource``,
    ``Symbol``, and ``Dependency`` objects produced by adapters.
    """

    @property
    def language(self) -> str:
        """
        Canonical lower-case language identifier.

        Examples: ``"python"``, ``"java"``, ``"typescript"``, ``"go"``.
        """
        ...

    def supports_file(self, path: str) -> bool:
        """
        Return True if this adapter can parse the given file.

        Adapters typically check the file extension but may also inspect
        shebang lines or magic bytes for ambiguous cases.
        """
        ...

    def parse(self, source: str, file_path: str) -> ParsedSource:
        """
        Parse ``source`` text and return a ``ParsedSource``.

        Args:
            source:    Complete source code as a string.
            file_path: Path used for error reporting and node IDs
                       (does not need to be readable from disk).

        Returns:
            ``ParsedSource`` with symbols and raw dependencies populated.
            On parse failure, ``ParsedSource.errors`` will be non-empty
            and ``symbols``/``dependencies`` may be empty or partial.
        """
        ...

    def extract_dependencies(self, parsed: ParsedSource) -> List[Dependency]:
        """
        Extract raw (unresolved) dependencies from a ``ParsedSource``.

        This is a separate step from ``parse`` so that callers can
        combine parsing across many files before starting resolution.

        Args:
            parsed: A ``ParsedSource`` produced by ``parse()``.

        Returns:
            List of ``Dependency`` objects.  These may already be
            populated on ``parsed.dependencies``; implementations may
            return the same list or augment it.
        """
        ...

    def resolve_symbols(
        self,
        parsed: ParsedSource,
        context: ProjectContext,
    ) -> ParsedSource:
        """
        Resolve symbols cross-file using the project context.

        Updates ``parsed.dependencies`` so that ``target`` fields point
        to fully-qualified names that exist in ``context.all_sources``.

        Args:
            parsed:  The ``ParsedSource`` to resolve.
            context: Project-wide context (all other parsed sources).

        Returns:
            A new or updated ``ParsedSource`` with resolved dependencies.
        """
        ...
