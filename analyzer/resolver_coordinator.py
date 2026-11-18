"""
Resolver Coordinator — Language-neutral resolution orchestration (Phase 7B).

Provides a clean boundary around language-specific resolvers so that the
core pipeline (graph builder, change analyzer, impact scorer) never
depends on Python-specific resolution types.

Architecture:
    LanguageAdapter.resolve_symbols()
            │
            ▼
    ResolverCoordinator (this module)
            │
            ├── PythonResolver → ProjectResolutionResult
            ├── JavaResolver   → ProjectResolutionResult (future)
            └── ...

The coordinator:
1. Accepts language-neutral ParsedSource + ProjectContext
2. Delegates to the appropriate language-specific resolver
3. Returns updated ParsedSource with resolved dependencies
4. Maintains a registry of language resolvers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, TYPE_CHECKING

from analyzer.languages.base import (
    Dependency,
    LanguageAdapter,
    LanguageDependencyType,
    ParsedSource,
    ProjectContext,
    Symbol,
    SymbolKind,
)

if TYPE_CHECKING:
    from analyzer.resolver import ProjectResolutionResult, NameResolver
    from pathlib import Path


# ============================================================================
# Language-specific Resolver Protocol
# ============================================================================

class LanguageResolver(Protocol):
    """
    Protocol for language-specific resolvers.

    Each language implements this to provide cross-file symbol resolution.
    The coordinator calls resolve() with the language's ParsedSource objects
    and expects updated ParsedSource objects with resolved dependency targets.
    """

    @property
    def language(self) -> str:
        """Canonical language identifier, e.g. 'python'."""
        ...

    def prepare(self, all_sources: List[ParsedSource]) -> None:
        """
        Prepare the resolver with all project sources.

        Called once before resolving individual files. Implementations
        should build any necessary symbol tables, module maps, etc.

        Args:
            all_sources: All ParsedSource objects in the project.
        """
        ...

    def resolve(
        self,
        parsed: ParsedSource,
        context: ProjectContext,
    ) -> ParsedSource:
        """
        Resolve cross-file symbols for a single ParsedSource.

        Updates parsed.dependencies so that targets point to fully-qualified
        names that exist in context.all_sources.

        Args:
            parsed: The ParsedSource to resolve.
            context: Project-wide context (all sources).

        Returns:
            The same ParsedSource with resolved dependencies.
        """
        ...


# ============================================================================
# Python Resolver Adapter
# ============================================================================

class PythonResolverAdapter:
    """
    Adapter that wraps the existing Phase 3 NameResolver behind the
    LanguageResolver protocol.

    This keeps the existing Python resolution logic intact while exposing
    it through the language-neutral interface.
    """

    def __init__(self, project_root: "Path") -> None:
        self._project_root = project_root
        self._resolver: Optional["NameResolver"] = None
        self._resolution_result: Optional["ProjectResolutionResult"] = None

    @property
    def language(self) -> str:
        return "python"

    def prepare(self, all_sources: List[ParsedSource]) -> None:
        """Convert ParsedSource → ParsedModule and run project resolution."""
        # Import here to avoid circular dependency
        from analyzer.languages.python.adapter import _parsed_source_to_parsed_module
        from analyzer.resolver import NameResolver

        python_sources = [s for s in all_sources if s.language == "python"]
        if not python_sources:
            return

        # Convert ParsedSource objects back to ParsedModule for Phase 3
        parsed_modules = [
            _parsed_source_to_parsed_module(s) for s in python_sources
        ]

        self._resolver = NameResolver(self._project_root)
        self._resolution_result = self._resolver.resolve_project(
            parsed_modules,
            source_paths=[Path(s.file_path) for s in python_sources],
        )

    def resolve(
        self,
        parsed: ParsedSource,
        context: ProjectContext,
    ) -> ParsedSource:
        """Apply pre-computed resolution to a single ParsedSource."""
        if parsed.language != "python" or self._resolution_result is None:
            return parsed

        # Import here to avoid circular dependency
        from analyzer.languages.python.adapter import _apply_resolution

        # _apply_resolution expects the full ProjectResolutionResult
        # and updates the parsed object in place
        _apply_resolution(parsed, self._resolution_result)
        return parsed


class AdapterResolver:
    """Resolver bridge for non-Python adapters using the neutral IR only."""

    def __init__(self, adapter: LanguageAdapter) -> None:
        self._adapter = adapter

    @property
    def language(self) -> str:
        return self._adapter.language

    def prepare(self, all_sources: List[ParsedSource]) -> None:
        # The adapters resolve against ProjectContext, so there is no
        # language-specific mutable state to prepare here.
        del all_sources

    def resolve(self, parsed: ParsedSource, context: ProjectContext) -> ParsedSource:
        return self._adapter.resolve_symbols(parsed, context)


# ============================================================================
# Resolver Coordinator
# ============================================================================

@dataclass
class ResolverCoordinator:
    """
    Coordinates language-specific resolvers for the core pipeline.

    The coordinator is the single point of contact for symbol resolution.
    Graph builder, change analyzer, and impact scorer only interact
    with this coordinator, never with language-specific resolvers directly.

    Usage:
        coordinator = ResolverCoordinator(project_root)
        coordinator.register_resolver(PythonResolverAdapter(project_root))
        # ... future: coordinator.register_resolver(JavaResolverAdapter(...))

        # Prepare with all sources
        coordinator.prepare(all_parsed_sources)

        # Resolve individual files
        resolved_source = coordinator.resolve(source, context)
    """

    project_root: "Path"
    _resolvers: Dict[str, LanguageResolver] = field(default_factory=dict)
    _prepared: bool = False

    def register_resolver(self, resolver: LanguageResolver) -> None:
        """
        Register a language-specific resolver.

        Args:
            resolver: An object implementing the LanguageResolver protocol.

        Raises:
            ValueError: If a resolver for the same language is already registered.
        """
        if resolver.language in self._resolvers:
            raise ValueError(
                f"Resolver for language '{resolver.language}' already registered"
            )
        self._resolvers[resolver.language] = resolver

    def get_resolver(self, language: str) -> Optional[LanguageResolver]:
        """Get the resolver for a language, or None if not registered."""
        return self._resolvers.get(language)

    def supported_languages(self) -> List[str]:
        """Return list of languages with registered resolvers."""
        return sorted(self._resolvers.keys())

    def prepare(self, all_sources: List[ParsedSource]) -> None:
        """
        Prepare all registered resolvers with the full project context.

        Must be called before resolve() for any source.

        Args:
            all_sources: All ParsedSource objects in the project.
        """
        for resolver in self._resolvers.values():
            resolver.prepare(all_sources)
        self._prepared = True

    def resolve(
        self,
        parsed: ParsedSource,
        context: ProjectContext,
    ) -> ParsedSource:
        """
        Resolve symbols for a single ParsedSource using its language's resolver.

        Args:
            parsed: The ParsedSource to resolve.
            context: Project-wide context.

        Returns:
            The ParsedSource with updated (resolved) dependencies.
        """
        if not self._prepared:
            # Auto-prepare if not explicitly prepared
            self.prepare(context.all_sources)

        resolver = self._resolvers.get(parsed.language)
        if resolver is None:
            # No resolver for this language — return as-is (unresolved)
            return parsed

        return resolver.resolve(parsed, context)

    def resolve_all(
        self,
        all_sources: List[ParsedSource],
        context: ProjectContext,
    ) -> List[ParsedSource]:
        """
        Resolve all sources in the project.

        Args:
            all_sources: All ParsedSource objects to resolve.
            context: Project-wide context.

        Returns:
            List of resolved ParsedSource objects (same objects, mutated).
        """
        if not self._prepared:
            self.prepare(all_sources)

        for source in all_sources:
            self.resolve(source, context)
        return all_sources


# ============================================================================
# Convenience functions
# ============================================================================

def create_default_coordinator(project_root: "Path") -> ResolverCoordinator:
    """
    Create a ResolverCoordinator with the Python resolver pre-registered.

    This is the standard entry point for Python-only projects.
    """
    coordinator = ResolverCoordinator(project_root)
    coordinator.register_resolver(PythonResolverAdapter(project_root))
    from analyzer.languages.registry import default_registry
    for language in ("typescript", "javascript", "csharp", "rust", "cpp"):
        adapter = default_registry.get(language)
        if adapter is not None:
            coordinator.register_resolver(AdapterResolver(adapter))
    return coordinator


def resolve_project(
    project_root: "Path",
    all_sources: List[ParsedSource],
) -> List[ParsedSource]:
    """
    Convenience function: prepare and resolve all sources for a project.

    Creates a default coordinator (Python-only), prepares it, and resolves
    all sources.

    Args:
        project_root: Absolute path to project root.
        all_sources: All ParsedSource objects in the project.

    Returns:
        The same ParsedSource objects with resolved dependencies.
    """
    coordinator = create_default_coordinator(project_root)
    return coordinator.resolve_all(all_sources, ProjectContext(
        root=str(project_root),
        all_sources=all_sources,
    ))
