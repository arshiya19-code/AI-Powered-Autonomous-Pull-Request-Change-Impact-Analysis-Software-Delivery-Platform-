"""
Python Language Adapter — Language Adapter #1 (Phase 7).

``PythonLanguageAdapter`` wraps the existing Phase 1–3 implementation
(``ASTParser``, ``DependencyExtractor``, ``NameResolver``) behind the
``LanguageAdapter`` Protocol defined in ``analyzer.languages.base``.

Architecture
------------
PythonLanguageAdapter
    │
    ├── .parse()               ← ASTParser.parse_source()
    │       └── converts ParsedModule → ParsedSource
    │
    ├── .extract_dependencies()  ← DependencyExtractor.extract()
    │       └── converts ModuleDependencies → List[Dependency]
    │
    └── .resolve_symbols()     ← NameResolver (project-level resolution)
            └── uses ProjectContext to build cross-file resolution
                updates ParsedSource.dependencies with resolved targets

This adapter is a *thin translation layer*.  It does not re-implement
any parsing, extraction, or resolution logic — it delegates entirely to
the proven Phase 1–3 code and converts the results into language-neutral
``ParsedSource``/``Symbol``/``Dependency`` objects.

Conversion mapping
------------------
Phase 1-3 type               → Phase 7 type
-------------------------------------------------
ParsedModule                 → ParsedSource
ClassInfo                    → Symbol (SymbolKind.CLASS)
FunctionInfo (standalone)    → Symbol (SymbolKind.FUNCTION)
FunctionInfo (method)        → Symbol (SymbolKind.METHOD)
DependencyType.IMPORT        → LanguageDependencyType.IMPORTS
DependencyType.FROM_IMPORT   → LanguageDependencyType.IMPORTS
DependencyType.FUNCTION_CALL → LanguageDependencyType.CALLS
DependencyType.METHOD_CALL   → LanguageDependencyType.CALLS
DependencyType.INHERITANCE   → LanguageDependencyType.INHERITS
DependencyType.DECORATOR     → LanguageDependencyType.ANNOTATES
DependencyType.MODULE_REFERENCE → LanguageDependencyType.REFERENCES

Backward compatibility
----------------------
The Phase 1–3 modules (``ast_parser``, ``dependency_extractor``, ``resolver``)
are **not modified**.  All their public APIs remain available as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from analyzer.ast_parser import (
    ASTParser,
    ClassInfo,
    FunctionInfo,
    ParsedModule,
)
from analyzer.dependency_extractor import (
    DependencyExtractor,
    DependencyType,
    ModuleDependencies,
)
from analyzer.resolver import NameResolver, ProjectResolutionResult
from analyzer.languages.base import (
    Dependency,
    LanguageDependencyType,
    ParsedSource,
    ProjectContext,
    Symbol,
    SymbolKind,
)


# ============================================================================
# Dependency type mapping
# ============================================================================

_DEP_TYPE_MAP: Dict[DependencyType, LanguageDependencyType] = {
    DependencyType.IMPORT:            LanguageDependencyType.IMPORTS,
    DependencyType.FROM_IMPORT:       LanguageDependencyType.IMPORTS,
    DependencyType.FUNCTION_CALL:     LanguageDependencyType.CALLS,
    DependencyType.METHOD_CALL:       LanguageDependencyType.CALLS,
    DependencyType.INHERITANCE:       LanguageDependencyType.INHERITS,
    DependencyType.DECORATOR:         LanguageDependencyType.ANNOTATES,
    DependencyType.MODULE_REFERENCE:  LanguageDependencyType.REFERENCES,
}


# ============================================================================
# Python Language Adapter
# ============================================================================

class PythonLanguageAdapter:
    """
    Language Adapter #1: Python.

    Delegates to the existing Phase 1–3 pipeline and converts results
    into the language-neutral ``ParsedSource`` representation.

    This class satisfies the ``LanguageAdapter`` Protocol via structural
    subtyping — it does not inherit from any base class.
    """

    @property
    def language(self) -> str:
        """Canonical identifier for this adapter."""
        return "python"

    def supports_file(self, path: str) -> bool:
        """
        Return ``True`` for ``*.py`` and ``*.pyw`` files.

        Args:
            path: File path (extension is checked; file need not exist).
        """
        return Path(path).suffix.lower() in {".py", ".pyw"}

    # ------------------------------------------------------------------
    # Core protocol methods
    # ------------------------------------------------------------------

    def parse(self, source: str, file_path: str = "<string>") -> ParsedSource:
        """
        Parse Python source text using the Phase 1 ``ASTParser``.

        Args:
            source:    Python source code as a string.
            file_path: Path for error reporting (default ``"<string>"``).

        Returns:
            ``ParsedSource`` populated from the ``ParsedModule``.
        """
        parser = ASTParser(include_docstrings=True, include_calls=True)
        parsed_module = parser.parse_source(source, file_path)
        parsed = self._parsed_module_to_parsed_source(parsed_module)
        # Kept private and consumed only by _parsed_source_to_parsed_module,
        # the compatibility adapter used by the existing Python resolver.
        # Core graph/change/impact code never reads this value.
        parsed.metadata["_python_resolver_source"] = source
        return parsed

    def extract_dependencies(self, parsed: ParsedSource) -> List[Dependency]:
        """
        Extract raw dependencies from a ``ParsedSource``.

        Internally re-parses the source to obtain a ``ParsedModule``, then
        runs ``DependencyExtractor``.  The result is also stored on
        ``parsed.dependencies`` for convenience.

        Note: If ``parsed.dependencies`` is already populated (e.g. from
        ``parse()``), this returns those same dependencies without re-parsing.

        Args:
            parsed: A ``ParsedSource`` produced by ``parse()``.

        Returns:
            List of ``Dependency`` objects.
        """
        if parsed.dependencies:
            return list(parsed.dependencies)

        # Re-parse via AST to obtain a ParsedModule for the extractor.
        # This is necessary because ParsedSource does not carry raw AST state.
        source = parsed.metadata.get("source", "")
        if not source:
            return []

        parser = ASTParser(include_docstrings=False, include_calls=True)
        pm = parser.parse_source(source, parsed.file_path)
        extractor = DependencyExtractor()
        mod_deps = extractor.extract(pm)

        deps = _module_deps_to_dependencies(mod_deps)
        parsed.dependencies.extend(deps)
        return deps

    def resolve_symbols(
        self,
        parsed: ParsedSource,
        context: ProjectContext,
    ) -> ParsedSource:
        """
        Resolve cross-file symbols using the Phase 3 ``NameResolver``.

        This performs a project-wide resolution pass for the single file
        ``parsed``.  In practice the Phase 3 resolver works best when given
        all project files at once; here we use the project context to gather
        all Python ``ParsedModule`` objects.

        Args:
            parsed:  The ``ParsedSource`` to resolve.
            context: Project-wide context.

        Returns:
            The same ``parsed`` object with ``dependencies`` updated so that
            targets of resolved internal deps use fully-qualified names.
        """
        # Build ParsedModule objects from context for the resolver
        # Only Python sources participate in Python resolution
        python_sources = [
            s for s in context.all_sources
            if s.language == "python"
        ]

        if not python_sources:
            return parsed

        # Convert ParsedSource objects back to ParsedModule for Phase 3
        parsed_modules = [
            _parsed_source_to_parsed_module(s) for s in python_sources
        ]

        project_root = Path(context.root)
        resolver = NameResolver(project_root)
        resolution: ProjectResolutionResult = resolver.resolve_project(parsed_modules)

        # Update dependency targets with resolved names where available
        _apply_resolution(parsed, resolution)

        return parsed

    # ------------------------------------------------------------------
    # Convenience: parse a file from disk
    # ------------------------------------------------------------------

    def parse_file(self, file_path: str) -> ParsedSource:
        """
        Parse a Python file from disk.

        Convenience method not required by the Protocol but useful for callers
        that have a file path rather than source text.

        Args:
            file_path: Absolute or relative path to a ``.py`` file.

        Returns:
            ``ParsedSource`` (may contain errors if the file cannot be read).
        """
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Preserve the established Phase 1 error representation for files
            # that cannot be read while keeping successful file parsing on the
            # source-preserving compatibility path below.
            parser = ASTParser(include_docstrings=True, include_calls=True)
            return self._parsed_module_to_parsed_source(parser.parse_file(path))
        return self.parse(source, str(path))

    # ------------------------------------------------------------------
    # Internal conversion helpers
    # ------------------------------------------------------------------

    def _parsed_module_to_parsed_source(self, pm: ParsedModule) -> ParsedSource:
        """Convert a Phase 1 ``ParsedModule`` to a language-neutral ``ParsedSource``."""
        return _parsed_module_to_parsed_source(pm)


# ============================================================================
# Module-level conversion functions
# ============================================================================

def _parsed_module_to_parsed_source(pm: ParsedModule) -> ParsedSource:
    """
    Convert a Phase 1 ``ParsedModule`` to a ``ParsedSource``.

    Symbols extracted:
      - Module-level symbol (one per file)
      - Class definitions
      - Standalone functions
      - Methods (linked to their parent class)

    Dependencies are extracted in full from the parsed module and
    converted via ``_module_deps_to_dependencies``.
    """
    symbols: List[Symbol] = []

    # 1. Module symbol
    module_qname = pm.module_name
    symbols.append(Symbol(
        name=pm.module_name.split(".")[-1],
        qualified_name=module_qname,
        symbol_kind=SymbolKind.MODULE,
        file_path=pm.file_path,
        start_line=1,
        end_line=pm.source_lines,
        language="python",
        metadata={
            "docstring": pm.docstring,
            "source_lines": pm.source_lines,
        },
    ))

    # 2. Classes
    for cls in pm.classes:
        cls_qname = f"{module_qname}.{cls.name}"
        symbols.append(_class_to_symbol(cls, cls_qname, pm.file_path))

        # 3. Methods
        for method in cls.methods:
            method_qname = f"{cls_qname}.{method.name}"
            symbols.append(_function_to_symbol(
                method, method_qname, pm.file_path,
                kind=SymbolKind.METHOD,
                parent=cls_qname,
            ))

    # 4. Standalone functions
    for func in pm.functions:
        func_qname = f"{module_qname}.{func.name}"
        symbols.append(_function_to_symbol(
            func, func_qname, pm.file_path,
            kind=SymbolKind.FUNCTION,
        ))

    # 5. Dependencies — extract from the ParsedModule
    extractor = DependencyExtractor()
    mod_deps = extractor.extract(pm)
    dependencies = _module_deps_to_dependencies(mod_deps)

    metadata: Dict[str, Any] = {
        "source_lines": pm.source_lines,
        "parse_time_ms": pm.parse_time_ms,
        "module_name": pm.module_name,
        "has_syntax_errors": bool(pm.syntax_errors),
    }

    return ParsedSource(
        file_path=pm.file_path,
        language="python",
        symbols=symbols,
        dependencies=dependencies,
        errors=list(pm.syntax_errors),
        metadata=metadata,
    )


def _class_to_symbol(
    cls: ClassInfo,
    qualified_name: str,
    file_path: str,
) -> Symbol:
    """Convert a Phase 1 ``ClassInfo`` to a ``Symbol``."""
    loc = cls.location
    metadata: Dict[str, Any] = {
        "bases": cls.bases,
        "decorators": cls.decorators,
        "docstring": cls.docstring,
    }
    return Symbol(
        name=cls.name,
        qualified_name=qualified_name,
        symbol_kind=SymbolKind.CLASS,
        file_path=file_path,
        start_line=loc.start_line if loc else 0,
        end_line=loc.end_line if loc else 0,
        language="python",
        is_public=not cls.name.startswith("_"),
        metadata=metadata,
    )


def _function_to_symbol(
    func: FunctionInfo,
    qualified_name: str,
    file_path: str,
    kind: SymbolKind = SymbolKind.FUNCTION,
    parent: Optional[str] = None,
) -> Symbol:
    """Convert a Phase 1 ``FunctionInfo`` to a ``Symbol``."""
    loc = func.location
    metadata: Dict[str, Any] = {
        "args": func.args,
        "returns": func.returns,
        "decorators": func.decorators,
        "is_async": func.is_async,
        "is_static": func.is_static,
        "is_classmethod": func.is_classmethod,
        "docstring": func.docstring,
    }
    return Symbol(
        name=func.name,
        qualified_name=qualified_name,
        symbol_kind=kind,
        file_path=file_path,
        start_line=loc.start_line if loc else 0,
        end_line=loc.end_line if loc else 0,
        language="python",
        parent=parent,
        is_public=not func.name.startswith("_"),
        metadata=metadata,
    )


def _module_deps_to_dependencies(
    mod_deps: ModuleDependencies,
) -> List[Dependency]:
    """
    Convert Phase 2 ``ModuleDependencies`` to a list of ``Dependency`` objects.
    """
    result: List[Dependency] = []
    for dep in mod_deps.dependencies:
        lang_dep_type = _DEP_TYPE_MAP.get(dep.dep_type, LanguageDependencyType.REFERENCES)
        metadata = dict(dep.metadata)
        # The neutral relationship is IMPORTS; this optional semantic detail
        # preserves the existing graph distinction between direct and named
        # imports without exposing a Python AST type to the core.
        if dep.dep_type == DependencyType.IMPORT:
            metadata["import_form"] = "module"
        elif dep.dep_type == DependencyType.FROM_IMPORT:
            metadata["import_form"] = "named"
        result.append(Dependency(
            source=dep.source,
            target=dep.target,
            dep_type=lang_dep_type,
            source_file=dep.source_file,
            source_line=dep.source_line,
            confidence=dep.confidence,
            language="python",
            metadata=metadata,
        ))
    return result


def _parsed_source_to_parsed_module(ps: ParsedSource) -> ParsedModule:
    """
    Create a minimal ``ParsedModule`` from a ``ParsedSource``.

    This is used when the Phase 3 resolver needs ``ParsedModule`` objects.
    We reconstruct the module metadata from the ParsedSource metadata dict.
    """
    # ``ParsedSource`` intentionally contains no Python AST objects. The
    # Phase 3 resolver still needs the richer Phase 1 representation, so this
    # compatibility conversion is deliberately confined to this Python-only
    # boundary.
    source = ps.metadata.get("_python_resolver_source")
    if isinstance(source, str):
        return ASTParser(include_docstrings=True, include_calls=True).parse_source(
            source, ps.file_path
        )

    path = Path(ps.file_path)
    if path.is_file():
        return ASTParser(include_docstrings=True, include_calls=True).parse_file(path)

    # A source-less ParsedSource can still feed the language-neutral graph,
    # but Phase 3 cannot recover imports or calls. Preserve that missingness
    # explicitly instead of fabricating resolved facts.
    return ParsedModule(
        file_path=ps.file_path,
        module_name=str(ps.metadata.get("module_name", ps.file_path)),
        source_lines=int(ps.metadata.get("source_lines", 0)),
        syntax_errors=list(ps.errors) + [
            "Python resolver compatibility source unavailable"
        ],
    )


def _apply_resolution(
    parsed: ParsedSource,
    resolution: ProjectResolutionResult,
) -> None:
    """
    Update ``parsed.dependencies`` targets using the Phase 3 resolution result.

    For each dependency whose ``source`` matches the parsed module name, look
    up the resolved target from the resolution result and update the target
    field to its fully-qualified resolved name.
    """
    module_name = parsed.metadata.get("module_name", "")
    if not module_name or module_name not in resolution.module_results:
        return

    res_result = resolution.module_results[module_name]
    # Build a lookup: original_target → resolved_symbol
    target_map: Dict[str, str] = {}
    for rd in res_result.resolved_dependencies:
        if rd.resolved_symbol:
            target_map[rd.original_target] = rd.resolved_symbol
        elif rd.resolved_module:
            target_map[rd.original_target] = rd.resolved_module

    # Update dependency targets
    for dep in parsed.dependencies:
        resolved = target_map.get(dep.target)
        if resolved and dep.target != resolved:
            dep.target = resolved
            dep.metadata["resolved"] = True
