"""
Tests for Phase 7 — Multi-Language Architecture Foundation.

Test coverage:
  1.  Language detection (.py, .java, .js, .ts, .go, .cpp, unknown)
  2.  Detector: detect_many, detect_with_support, known_extensions/languages
  3.  Registry: register, get, require, is_supported, supported_languages
  4.  Registry: duplicate registration raises DuplicateAdapterError
  5.  Registry: unknown language returns None / raises UnsupportedLanguageError
  6.  Registry: unregister
  7.  default_registry: Python pre-registered
  8.  PythonLanguageAdapter: language property, supports_file
  9.  PythonLanguageAdapter: parse() → ParsedSource structure
  10. PythonLanguageAdapter: parse() with syntax error
  11. PythonLanguageAdapter: symbols extracted from source
  12. PythonLanguageAdapter: extract_dependencies() from source
  13. PythonLanguageAdapter: parse_file() convenience method
  14. Unified representations: SourceLocation immutable
  15. Unified representations: Symbol fields
  16. Unified representations: ParsedSource convenience properties
  17. Unified representations: Dependency fields
  18. Unsupported language: Java detected, no adapter
  19. Unsupported language: graceful handling via registry.get()
  20. Protocol compliance: PythonLanguageAdapter satisfies LanguageAdapter
  21. LanguageAdapter isinstance check (runtime_checkable)
  22. Multi-language repository model: detect many extensions
  23. Version check: __version__ == "0.7.0"
  24. Backward compatibility: existing imports still work
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Phase 7 imports under test
# ---------------------------------------------------------------------------
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
from analyzer.languages.detector import DetectedLanguage, LanguageDetector
from analyzer.languages.registry import (
    DuplicateAdapterError,
    LanguageRegistry,
    UnsupportedLanguageError,
    default_registry,
)
from analyzer.languages.python.adapter import PythonLanguageAdapter

# Phase 7 also re-exported from analyzer root
import analyzer


# ===========================================================================
# 1. Language Detection
# ===========================================================================

class TestLanguageDetector:
    def setup_method(self) -> None:
        self.detector = LanguageDetector()

    def test_python_extension(self) -> None:
        result = self.detector.detect("foo.py")
        assert result.language == "python"
        assert result.extension == ".py"

    def test_python_pyw(self) -> None:
        result = self.detector.detect("script.pyw")
        assert result.language == "python"

    def test_java_extension(self) -> None:
        result = self.detector.detect("Main.java")
        assert result.language == "java"

    def test_javascript_js(self) -> None:
        result = self.detector.detect("app.js")
        assert result.language == "javascript"

    def test_javascript_jsx(self) -> None:
        result = self.detector.detect("Component.jsx")
        assert result.language == "javascript"

    def test_typescript_ts(self) -> None:
        result = self.detector.detect("service.ts")
        assert result.language == "typescript"

    def test_typescript_tsx(self) -> None:
        result = self.detector.detect("App.tsx")
        assert result.language == "typescript"

    def test_go_extension(self) -> None:
        result = self.detector.detect("main.go")
        assert result.language == "go"

    def test_c_extension(self) -> None:
        result = self.detector.detect("main.c")
        assert result.language == "c"

    def test_c_header(self) -> None:
        result = self.detector.detect("header.h")
        assert result.language == "c"

    def test_cpp_extension(self) -> None:
        result = self.detector.detect("algo.cpp")
        assert result.language == "cpp"

    def test_cpp_hpp(self) -> None:
        result = self.detector.detect("mylib.hpp")
        assert result.language == "cpp"

    def test_unknown_extension(self) -> None:
        result = self.detector.detect("file.xyz")
        assert result.language == "unknown"
        assert result.extension == ".xyz"

    def test_no_extension(self) -> None:
        result = self.detector.detect("Makefile")
        assert result.language == "unknown"
        assert result.extension == ""

    def test_is_supported_always_false_from_detector(self) -> None:
        # The detector alone never marks anything as supported
        result = self.detector.detect("foo.py")
        assert result.is_supported is False

    def test_detect_many(self) -> None:
        paths = ["a.py", "b.java", "c.go", "d.xyz"]
        results = self.detector.detect_many(paths)
        assert results["a.py"].language == "python"
        assert results["b.java"].language == "java"
        assert results["c.go"].language == "go"
        assert results["d.xyz"].language == "unknown"
        assert len(results) == 4

    def test_detect_with_support_known(self) -> None:
        result = self.detector.detect_with_support("foo.py", ["python"])
        assert result.is_supported is True

    def test_detect_with_support_unknown(self) -> None:
        result = self.detector.detect_with_support("foo.java", ["python"])
        assert result.is_supported is False

    def test_known_extensions_not_empty(self) -> None:
        exts = self.detector.known_extensions()
        assert ".py" in exts
        assert ".java" in exts
        assert ".ts" in exts
        assert len(exts) > 5

    def test_known_languages_not_empty(self) -> None:
        langs = self.detector.known_languages()
        assert "python" in langs
        assert "java" in langs
        assert "go" in langs

    def test_detected_language_str(self) -> None:
        result = self.detector.detect("foo.py")
        s = str(result)
        assert "python" in s
        assert "not supported" in s  # is_supported=False


# ===========================================================================
# 2. Language Registry
# ===========================================================================

class TestLanguageRegistry:
    def setup_method(self) -> None:
        self.registry = LanguageRegistry()
        self.adapter = PythonLanguageAdapter()

    def test_register_and_get(self) -> None:
        self.registry.register(self.adapter)
        result = self.registry.get("python")
        assert result is self.adapter

    def test_get_unknown_returns_none(self) -> None:
        result = self.registry.get("java")
        assert result is None

    def test_require_raises_unsupported(self) -> None:
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            self.registry.require("java")
        assert exc_info.value.language == "java"
        assert "java" in str(exc_info.value)

    def test_require_returns_adapter(self) -> None:
        self.registry.register(self.adapter)
        result = self.registry.require("python")
        assert result is self.adapter

    def test_is_supported_true(self) -> None:
        self.registry.register(self.adapter)
        assert self.registry.is_supported("python") is True

    def test_is_supported_false(self) -> None:
        assert self.registry.is_supported("java") is False

    def test_supported_languages(self) -> None:
        self.registry.register(self.adapter)
        langs = self.registry.supported_languages()
        assert langs == ["python"]

    def test_duplicate_adapter_same_instance_idempotent(self) -> None:
        self.registry.register(self.adapter)
        # Registering the same object again is fine
        self.registry.register(self.adapter)
        assert len(self.registry) == 1

    def test_duplicate_adapter_different_instance_raises(self) -> None:
        self.registry.register(self.adapter)
        other = PythonLanguageAdapter()
        with pytest.raises(DuplicateAdapterError) as exc_info:
            self.registry.register(other)
        assert exc_info.value.language == "python"

    def test_unregister_existing(self) -> None:
        self.registry.register(self.adapter)
        removed = self.registry.unregister("python")
        assert removed is True
        assert self.registry.get("python") is None

    def test_unregister_nonexistent(self) -> None:
        removed = self.registry.unregister("java")
        assert removed is False

    def test_len(self) -> None:
        assert len(self.registry) == 0
        self.registry.register(self.adapter)
        assert len(self.registry) == 1

    def test_repr(self) -> None:
        self.registry.register(self.adapter)
        r = repr(self.registry)
        assert "python" in r

    def test_multiple_adapters(self) -> None:
        # Simulate future adapters by creating a stub that has a different language
        class StubAdapter:
            @property
            def language(self) -> str:
                return "java"

            def supports_file(self, path: str) -> bool:
                return path.endswith(".java")

            def parse(self, source: str, file_path: str) -> ParsedSource:
                return ParsedSource(file_path=file_path, language="java")

            def extract_dependencies(self, parsed: ParsedSource) -> List[Dependency]:
                return []

            def resolve_symbols(self, parsed: ParsedSource, context: ProjectContext) -> ParsedSource:
                return parsed

        self.registry.register(self.adapter)
        self.registry.register(StubAdapter())  # type: ignore[arg-type]
        assert self.registry.is_supported("python")
        assert self.registry.is_supported("java")
        assert sorted(self.registry.supported_languages()) == ["java", "python"]


# ===========================================================================
# 3. Default Registry
# ===========================================================================

class TestDefaultRegistry:
    def test_python_pre_registered(self) -> None:
        assert default_registry.is_supported("python")

    def test_java_not_registered(self) -> None:
        assert not default_registry.is_supported("java")

    def test_javascript_registered(self) -> None:
        assert default_registry.is_supported("javascript")

    def test_get_python_adapter(self) -> None:
        adapter = default_registry.get("python")
        assert adapter is not None
        assert adapter.language == "python"

    def test_get_java_returns_none(self) -> None:
        adapter = default_registry.get("java")
        assert adapter is None

    def test_supported_languages_contains_python(self) -> None:
        langs = default_registry.supported_languages()
        assert "python" in langs


# ===========================================================================
# 4. Python Language Adapter — Core Protocol
# ===========================================================================

class TestPythonAdapterProtocol:
    def setup_method(self) -> None:
        self.adapter = PythonLanguageAdapter()

    def test_language_property(self) -> None:
        assert self.adapter.language == "python"

    def test_supports_py_files(self) -> None:
        assert self.adapter.supports_file("foo.py") is True
        assert self.adapter.supports_file("path/to/module.py") is True

    def test_supports_pyw_files(self) -> None:
        assert self.adapter.supports_file("script.pyw") is True

    def test_does_not_support_java(self) -> None:
        assert self.adapter.supports_file("Main.java") is False

    def test_does_not_support_ts(self) -> None:
        assert self.adapter.supports_file("app.ts") is False

    def test_does_not_support_no_extension(self) -> None:
        assert self.adapter.supports_file("Makefile") is False

    def test_isinstance_language_adapter_protocol(self) -> None:
        # LanguageAdapter is runtime_checkable
        assert isinstance(self.adapter, LanguageAdapter)


# ===========================================================================
# 5. Python Adapter — parse()
# ===========================================================================

_SIMPLE_MODULE = """\
\"\"\"A simple module.\"\"\"

import os
from pathlib import Path


class MyClass:
    \"\"\"A class.\"\"\"

    def my_method(self, x: int) -> str:
        return str(x)


def standalone_func(a: int, b: int) -> int:
    return a + b
"""


class TestPythonAdapterParse:
    def setup_method(self) -> None:
        self.adapter = PythonLanguageAdapter()

    def test_parse_returns_parsed_source(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        assert isinstance(result, ParsedSource)

    def test_parse_language_field(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        assert result.language == "python"

    def test_parse_file_path(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        assert result.file_path == "mymodule.py"

    def test_parse_no_errors_on_valid_source(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        assert result.errors == []
        assert not result.has_errors

    def test_parse_extracts_class_symbol(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        class_names = [s.qualified_name for s in result.class_symbols]
        assert any("MyClass" in name for name in class_names)

    def test_parse_extracts_method_symbol(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        method_names = [s.qualified_name for s in result.method_symbols]
        assert any("my_method" in name for name in method_names)

    def test_parse_extracts_function_symbol(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        func_names = [s.qualified_name for s in result.function_symbols]
        assert any("standalone_func" in name for name in func_names)

    def test_parse_extracts_module_symbol(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        assert len(result.module_symbols) == 1

    def test_parse_symbols_have_language_python(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        for sym in result.symbols:
            assert sym.language == "python", f"Symbol {sym.qualified_name} has wrong language"

    def test_parse_extracts_import_dependencies(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        dep_types = {d.dep_type for d in result.dependencies}
        assert LanguageDependencyType.IMPORTS in dep_types

    def test_parse_dependencies_have_language(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        for dep in result.dependencies:
            assert dep.language == "python"

    def test_parse_metadata_source_lines(self) -> None:
        result = self.adapter.parse(_SIMPLE_MODULE, "mymodule.py")
        assert "source_lines" in result.metadata
        assert result.metadata["source_lines"] > 0

    def test_parse_syntax_error_source(self) -> None:
        bad_source = "def broken(\n    x: int\n# missing closing paren"
        result = self.adapter.parse(bad_source, "broken.py")
        assert result.has_errors
        assert len(result.errors) > 0

    def test_parse_empty_source(self) -> None:
        result = self.adapter.parse("", "empty.py")
        assert isinstance(result, ParsedSource)
        assert result.language == "python"


# ===========================================================================
# 6. Python Adapter — Specific Dependency Extraction
# ===========================================================================

_DEP_SOURCE = """\
import os
import sys
from pathlib import Path
from collections import defaultdict


class Base:
    pass


class Child(Base):
    @staticmethod
    def helper():
        pass

    def process(self):
        helper()
        os.path.join("a", "b")
"""


class TestPythonAdapterDependencies:
    def setup_method(self) -> None:
        self.adapter = PythonLanguageAdapter()

    def test_import_deps_present(self) -> None:
        parsed = self.adapter.parse(_DEP_SOURCE, "deps.py")
        import_deps = [d for d in parsed.dependencies
                       if d.dep_type == LanguageDependencyType.IMPORTS]
        assert len(import_deps) >= 2  # at least os, sys

    def test_inheritance_dep_present(self) -> None:
        parsed = self.adapter.parse(_DEP_SOURCE, "deps.py")
        inherits_deps = [d for d in parsed.dependencies
                         if d.dep_type == LanguageDependencyType.INHERITS]
        assert len(inherits_deps) >= 1

    def test_annotates_dep_present(self) -> None:
        parsed = self.adapter.parse(_DEP_SOURCE, "deps.py")
        ann_deps = [d for d in parsed.dependencies
                    if d.dep_type == LanguageDependencyType.ANNOTATES]
        assert len(ann_deps) >= 1  # @staticmethod

    def test_calls_dep_present(self) -> None:
        parsed = self.adapter.parse(_DEP_SOURCE, "deps.py")
        call_deps = [d for d in parsed.dependencies
                     if d.dep_type == LanguageDependencyType.CALLS]
        assert len(call_deps) >= 1

    def test_extract_dependencies_returns_list(self) -> None:
        parsed = ParsedSource(
            file_path="x.py",
            language="python",
            dependencies=[
                Dependency(
                    source="x",
                    target="y",
                    dep_type=LanguageDependencyType.IMPORTS,
                    source_file="x.py",
                )
            ],
        )
        deps = self.adapter.extract_dependencies(parsed)
        assert isinstance(deps, list)
        assert len(deps) == 1


# ===========================================================================
# 7. Python Adapter — parse_file()
# ===========================================================================

class TestPythonAdapterParseFile:
    def setup_method(self) -> None:
        self.adapter = PythonLanguageAdapter()

    def test_parse_file_returns_parsed_source(self, tmp_path: Path) -> None:
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n", encoding="utf-8")
        result = self.adapter.parse_file(str(py_file))
        assert isinstance(result, ParsedSource)
        assert result.language == "python"

    def test_parse_file_nonexistent_has_errors(self) -> None:
        result = self.adapter.parse_file("/nonexistent/path/to/file.py")
        assert result.has_errors


# ===========================================================================
# 8. Unified Representations
# ===========================================================================

class TestSourceLocation:
    def test_immutable(self) -> None:
        loc = SourceLocation(file_path="foo.py", start_line=1, end_line=5)
        with pytest.raises((AttributeError, TypeError)):
            loc.start_line = 10  # type: ignore[misc]

    def test_str(self) -> None:
        loc = SourceLocation(file_path="foo.py", start_line=1, end_line=5)
        assert "foo.py" in str(loc)
        assert "1" in str(loc)

    def test_defaults(self) -> None:
        loc = SourceLocation(file_path="x.py", start_line=1, end_line=1)
        assert loc.start_col == 0
        assert loc.end_col == 0


class TestSymbol:
    def test_creation(self) -> None:
        sym = Symbol(
            name="MyClass",
            qualified_name="mymodule.MyClass",
            symbol_kind=SymbolKind.CLASS,
            file_path="mymodule.py",
        )
        assert sym.name == "MyClass"
        assert sym.symbol_kind == SymbolKind.CLASS
        assert sym.language == ""  # default

    def test_str(self) -> None:
        sym = Symbol(
            name="func",
            qualified_name="mod.func",
            symbol_kind=SymbolKind.FUNCTION,
            file_path="mod.py",
        )
        s = str(sym)
        assert "function" in s
        assert "mod.func" in s

    def test_metadata_default_empty(self) -> None:
        sym = Symbol(
            name="x", qualified_name="m.x",
            symbol_kind=SymbolKind.VARIABLE, file_path="m.py"
        )
        assert sym.metadata == {}

    def test_is_public_default_true(self) -> None:
        sym = Symbol(
            name="x", qualified_name="m.x",
            symbol_kind=SymbolKind.FUNCTION, file_path="m.py"
        )
        assert sym.is_public is True


class TestDependency:
    def test_creation(self) -> None:
        dep = Dependency(
            source="mod_a",
            target="mod_b",
            dep_type=LanguageDependencyType.IMPORTS,
            source_file="mod_a.py",
            source_line=3,
        )
        assert dep.source == "mod_a"
        assert dep.dep_type == LanguageDependencyType.IMPORTS
        assert dep.confidence == 1.0

    def test_str(self) -> None:
        dep = Dependency(
            source="a", target="b",
            dep_type=LanguageDependencyType.CALLS,
            source_file="a.py"
        )
        s = str(dep)
        assert "calls" in s
        assert "a" in s
        assert "b" in s


class TestParsedSource:
    def test_has_errors_false(self) -> None:
        ps = ParsedSource(file_path="x.py", language="python")
        assert not ps.has_errors

    def test_has_errors_true(self) -> None:
        ps = ParsedSource(file_path="x.py", language="python", errors=["SyntaxError"])
        assert ps.has_errors

    def test_convenience_properties_empty(self) -> None:
        ps = ParsedSource(file_path="x.py", language="python")
        assert ps.module_symbols == []
        assert ps.class_symbols == []
        assert ps.function_symbols == []
        assert ps.method_symbols == []

    def test_convenience_properties_populated(self) -> None:
        syms = [
            Symbol(name="m", qualified_name="m", symbol_kind=SymbolKind.MODULE, file_path="x.py"),
            Symbol(name="C", qualified_name="m.C", symbol_kind=SymbolKind.CLASS, file_path="x.py"),
            Symbol(name="f", qualified_name="m.f", symbol_kind=SymbolKind.FUNCTION, file_path="x.py"),
            Symbol(name="meth", qualified_name="m.C.meth", symbol_kind=SymbolKind.METHOD, file_path="x.py"),
        ]
        ps = ParsedSource(file_path="x.py", language="python", symbols=syms)
        assert len(ps.module_symbols) == 1
        assert len(ps.class_symbols) == 1
        assert len(ps.function_symbols) == 1
        assert len(ps.method_symbols) == 1

    def test_repr(self) -> None:
        ps = ParsedSource(file_path="x.py", language="python")
        r = repr(ps)
        assert "python" in r
        assert "x.py" in r


# ===========================================================================
# 9. Unsupported Languages — Graceful Handling
# ===========================================================================

class TestUnsupportedLanguages:
    def test_java_detected_no_adapter(self) -> None:
        detector = LanguageDetector()
        detected = detector.detect("Main.java")
        assert detected.language == "java"
        # No adapter in default registry
        adapter = default_registry.get("java")
        assert adapter is None

    def test_unsupported_language_error(self) -> None:
        registry = LanguageRegistry()
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            registry.require("java")
        assert "java" in str(exc_info.value)

    def test_supported_adapter_is_available(self) -> None:
        assert default_registry.get("typescript") is not None

    def test_detect_many_unsupported(self) -> None:
        detector = LanguageDetector()
        paths = ["app.ts", "App.tsx", "main.go", "service.java"]
        results = detector.detect_many(paths)
        for path, detected in results.items():
            assert detected.language != "unknown", f"{path} should be detected"
        assert default_registry.get("typescript") is not None
        assert default_registry.get("go") is None
        assert default_registry.get("java") is None

    def test_unsupported_language_error_language_attr(self) -> None:
        err = UnsupportedLanguageError("go")
        assert err.language == "go"

    def test_duplicate_adapter_error_language_attr(self) -> None:
        err = DuplicateAdapterError("python")
        assert err.language == "python"


# ===========================================================================
# 10. Multi-Language Repository Model
# ===========================================================================

class TestMultiLanguageRepository:
    """
    Verifies that the system can model a polyglot repository:
    detect languages per file, identify supported vs unsupported,
    and use the correct adapter for supported files.
    """

    def test_polyglot_repo_detection(self) -> None:
        detector = LanguageDetector()
        files = [
            "src/main.py",
            "src/service.py",
            "frontend/App.tsx",
            "backend/Handler.java",
            "internal/main.go",
            "lib/util.cpp",
        ]
        detected = detector.detect_many(files)

        assert detected["src/main.py"].language == "python"
        assert detected["src/service.py"].language == "python"
        assert detected["frontend/App.tsx"].language == "typescript"
        assert detected["backend/Handler.java"].language == "java"
        assert detected["internal/main.go"].language == "go"
        assert detected["lib/util.cpp"].language == "cpp"

    def test_python_files_are_supported(self) -> None:
        detector = LanguageDetector()
        py_files = ["a.py", "b.py"]
        for path in py_files:
            detected = detector.detect(path)
            assert default_registry.is_supported(detected.language)

    def test_only_fixed_cohort_languages_are_supported(self) -> None:
        detector = LanguageDetector()
        assert default_registry.is_supported(detector.detect("app.ts").language)
        assert not default_registry.is_supported(detector.detect("Main.java").language)
        assert not default_registry.is_supported(detector.detect("main.go").language)

    def test_file_to_adapter_routing(self) -> None:
        """
        Given a list of files, route each to an adapter (or skip).
        This mirrors how a multi-language pipeline would work.
        """
        detector = LanguageDetector()
        files = ["a.py", "b.java", "c.go"]
        processed: list[str] = []
        skipped: list[str] = []

        for path in files:
            detected = detector.detect(path)
            adapter = default_registry.get(detected.language)
            if adapter is not None:
                processed.append(path)
            else:
                skipped.append(path)

        assert "a.py" in processed
        assert "b.java" in skipped
        assert "c.go" in skipped


# ===========================================================================
# 11. Version and Backward Compatibility
# ===========================================================================

class TestVersionAndBackwardCompat:
    def test_version_is_0_7_0(self) -> None:
        assert analyzer.__version__ == "0.7.0"

    def test_phase1_imports_still_work(self) -> None:
        from analyzer.ast_parser import ASTParser, ParsedModule, FunctionInfo, ClassInfo
        assert ASTParser is not None
        assert ParsedModule is not None

    def test_phase2_imports_still_work(self) -> None:
        from analyzer.dependency_extractor import DependencyExtractor, Dependency as D2, DependencyType
        assert DependencyExtractor is not None
        assert DependencyType is not None

    def test_phase3_imports_still_work(self) -> None:
        from analyzer.resolver import NameResolver, ModuleDiscovery, SymbolTableBuilder
        assert NameResolver is not None

    def test_phase4_imports_still_work(self) -> None:
        from analyzer.graph_engine import CustomDependencyGraph, GraphBuilder, GraphNode
        assert CustomDependencyGraph is not None

    def test_phase5_imports_still_work(self) -> None:
        from analyzer.change_analyzer import ChangeAnalyzer, ChangeSet, ImpactAnalyzer
        assert ChangeAnalyzer is not None

    def test_phase6_imports_still_work(self) -> None:
        from analyzer.impact_scorer import ImpactScorer, ImpactScoringResult
        assert ImpactScorer is not None

    def test_phase7_imports_from_analyzer_root(self) -> None:
        from analyzer import (
            LanguageAdapter,
            ParsedSource,
            PythonLanguageAdapter,
            LanguageDetector,
            LanguageRegistry,
            default_registry,
        )
        assert LanguageAdapter is not None
        assert PythonLanguageAdapter is not None
        assert default_registry is not None

    def test_source_location_from_ast_parser_unaffected(self) -> None:
        # The existing SourceLocation in ast_parser must still work
        from analyzer.ast_parser import SourceLocation as OldSL
        loc = OldSL(file_path="x.py", start_line=1, end_line=5)
        assert loc.file_path == "x.py"

    def test_new_source_location_in_languages(self) -> None:
        # The new LanguageSourceLocation in languages.base coexists
        from analyzer.languages.base import SourceLocation as NewSL
        loc = NewSL(file_path="x.py", start_line=1, end_line=5)
        assert loc.file_path == "x.py"

    def test_existing_parse_file_function(self) -> None:
        # The top-level parse_file convenience function still works
        from analyzer.ast_parser import parse_file
        assert callable(parse_file)


# ===========================================================================
# 12. LanguageDependencyType Completeness
# ===========================================================================

class TestLanguageDependencyType:
    def test_all_required_types_present(self) -> None:
        expected = {"imports", "calls", "inherits", "annotates", "references"}
        actual = {t.value for t in LanguageDependencyType}
        assert expected == actual, f"Missing types: {expected - actual}"


# ===========================================================================
# 13. SymbolKind Completeness
# ===========================================================================

class TestSymbolKind:
    def test_all_required_kinds_present(self) -> None:
        expected = {"module", "class", "function", "method", "variable"}
        actual = {k.value for k in SymbolKind}
        assert expected == actual, f"Missing kinds: {expected - actual}"


# ===========================================================================
# 14. ProjectContext
# ===========================================================================

class TestProjectContext:
    def test_creation_minimal(self) -> None:
        ctx = ProjectContext(root="/project")
        assert ctx.root == "/project"
        assert ctx.all_sources == []
        assert ctx.language_hints == {}

    def test_creation_with_sources(self) -> None:
        ps = ParsedSource(file_path="a.py", language="python")
        ctx = ProjectContext(root="/proj", all_sources=[ps])
        assert len(ctx.all_sources) == 1
