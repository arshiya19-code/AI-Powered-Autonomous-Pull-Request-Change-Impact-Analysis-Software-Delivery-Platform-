"""
Language Registry (Phase 7).

A central registry that maps language identifiers to ``LanguageAdapter``
implementations.  Adapters are registered once at import-time and retrieved
on demand.

The module-level ``default_registry`` singleton is pre-registered with
``PythonLanguageAdapter`` so existing Python workflows require no changes.

Usage::

    from analyzer.languages.registry import default_registry

    # Check support
    default_registry.is_supported("python")   # True
    default_registry.is_supported("java")     # False

    # Retrieve adapter
    adapter = default_registry.get("python")
    if adapter is None:
        raise RuntimeError("No adapter for 'python'")

    # Register a new adapter (future)
    from my_package import JavaAdapter
    default_registry.register(JavaAdapter())

Design notes
------------
* ``register()`` is idempotent if the same adapter instance is registered
  twice, but raises ``DuplicateAdapterError`` if a *different* adapter tries
  to claim the same language identifier.
* Thread safety: the registry uses a plain dict.  It is designed to be
  populated at module load time (single-threaded), not mutated concurrently.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from analyzer.languages.base import LanguageAdapter

if TYPE_CHECKING:
    pass  # no circular imports needed


# ============================================================================
# Exceptions
# ============================================================================

class DuplicateAdapterError(Exception):
    """
    Raised when an adapter registration conflicts with an existing one.

    Two adapters for the same language cannot coexist in a registry unless
    they are the identical object.
    """
    def __init__(self, language: str) -> None:
        super().__init__(
            f"An adapter for language '{language}' is already registered. "
            f"Call registry.unregister('{language}') first if you intend to replace it."
        )
        self.language = language


class UnsupportedLanguageError(Exception):
    """
    Raised when a language is detected but no adapter is registered for it.

    Callers should catch this and respond gracefully (e.g. skip the file,
    log a warning) rather than letting it propagate as an unhandled error.
    """
    def __init__(self, language: str) -> None:
        super().__init__(
            f"No language adapter is registered for '{language}'. "
            f"The language was detected but cannot be processed."
        )
        self.language = language


# ============================================================================
# Registry
# ============================================================================

class LanguageRegistry:
    """
    Central registry mapping language identifiers to ``LanguageAdapter`` objects.

    Typical lifecycle::

        registry = LanguageRegistry()
        registry.register(PythonLanguageAdapter())
        # ... future: registry.register(JavaLanguageAdapter())

        adapter = registry.get("python")
        supported = registry.supported_languages()
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, LanguageAdapter] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, adapter: LanguageAdapter) -> None:
        """
        Register a language adapter.

        Args:
            adapter: An object that satisfies the ``LanguageAdapter`` Protocol.

        Raises:
            DuplicateAdapterError: If a different adapter for the same language
                is already registered.
        """
        language = adapter.language
        existing = self._adapters.get(language)
        if existing is not None and existing is not adapter:
            raise DuplicateAdapterError(language)
        self._adapters[language] = adapter

    def unregister(self, language: str) -> bool:
        """
        Remove the adapter for ``language`` (if any).

        Args:
            language: Lower-case language identifier.

        Returns:
            ``True`` if an adapter was removed, ``False`` if none was registered.
        """
        if language in self._adapters:
            del self._adapters[language]
            return True
        return False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, language: str) -> Optional[LanguageAdapter]:
        """
        Return the adapter for ``language``, or ``None`` if not registered.

        This is the preferred method for graceful unsupported-language
        handling: callers check for ``None`` rather than catching exceptions.
        """
        return self._adapters.get(language)

    def require(self, language: str) -> LanguageAdapter:
        """
        Return the adapter for ``language`` or raise ``UnsupportedLanguageError``.

        Use when the absence of an adapter is a hard error in the calling
        context.  For graceful handling use ``get()`` instead.

        Raises:
            UnsupportedLanguageError: If no adapter is registered.
        """
        adapter = self._adapters.get(language)
        if adapter is None:
            raise UnsupportedLanguageError(language)
        return adapter

    def is_supported(self, language: str) -> bool:
        """Return ``True`` if an adapter is registered for ``language``."""
        return language in self._adapters

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def supported_languages(self) -> List[str]:
        """Return a sorted list of all registered language identifiers."""
        return sorted(self._adapters.keys())

    def __len__(self) -> int:
        return len(self._adapters)

    def __repr__(self) -> str:
        languages = self.supported_languages()
        return f"LanguageRegistry(supported={languages!r})"


# ============================================================================
# Module-level default registry
# ============================================================================

# We use a factory function to avoid circular import issues: PythonLanguageAdapter
# imports from base.py which is safe, but we register lazily to keep import order clean.

def _build_default_registry() -> LanguageRegistry:
    """Build the registry for every language in the fixed cohort."""
    from analyzer.languages.python.adapter import PythonLanguageAdapter  # noqa: PLC0415
    from analyzer.languages.structured.adapter import (  # noqa: PLC0415
        CppLanguageAdapter, CSharpLanguageAdapter, JavaScriptLanguageAdapter,
        RustLanguageAdapter, TypeScriptLanguageAdapter,
    )
    reg = LanguageRegistry()
    reg.register(PythonLanguageAdapter())
    reg.register(TypeScriptLanguageAdapter())
    reg.register(JavaScriptLanguageAdapter())
    reg.register(CSharpLanguageAdapter())
    reg.register(RustLanguageAdapter())
    reg.register(CppLanguageAdapter())
    return reg


# Singleton — populated on first access via module-level eager init.
# Importing this module guarantees Python is available.
default_registry: LanguageRegistry = _build_default_registry()
