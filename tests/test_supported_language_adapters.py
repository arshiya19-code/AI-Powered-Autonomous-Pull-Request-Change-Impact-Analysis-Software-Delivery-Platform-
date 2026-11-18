"""Validation of every real adapter in the shared ParsedSource pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Type

import pytest

from analyzer.change_analyzer import ChangeAnalyzer
from analyzer.graph_engine import GraphBuilder
from analyzer.languages.base import LanguageAdapter, LanguageDependencyType, ProjectContext
from analyzer.languages.registry import default_registry
from analyzer.resolver_coordinator import create_default_coordinator
from analyzer.languages.structured import (
    CppLanguageAdapter, CSharpLanguageAdapter, JavaScriptLanguageAdapter,
    RustLanguageAdapter, TypeScriptLanguageAdapter,
)


@pytest.mark.parametrize(("adapter_type", "path", "source"), [
    (TypeScriptLanguageAdapter, "src/service.ts", '''import { helper } from "./helper";
export class Service { run() { return helper(); } }
export function entry() { return new Service().run(); }'''),
    (JavaScriptLanguageAdapter, "web/app.js", '''const helper = require("./helper");
class App { run() { return helper(); } }
function entry() { return new App().run(); }'''),
    (CSharpLanguageAdapter, "src/Service.cs", '''using System.Text;
public class Service { public string Run() { return Format(); } }
public static string Format() { return "ok"; }'''),
    (RustLanguageAdapter, "src/lib.rs", '''use crate::helper;
struct Service { value: i32 }
impl Service { fn run(&self) { helper(); } }
fn entry() { helper(); }'''),
    (CppLanguageAdapter, "src/service.cpp", '''#include "helper.hpp"
class Service { public: void run() { helper(); } };
void entry() { helper(); }'''),
])
def test_adapter_parses_ir_dependencies_calls_and_graph(
    tmp_path: Path, adapter_type: Type[LanguageAdapter], path: str, source: str,
) -> None:
    adapter = adapter_type()
    parsed = adapter.parse(source, path)
    assert parsed.language == adapter.language
    assert parsed.module_symbols and parsed.class_symbols
    assert parsed.function_symbols or parsed.method_symbols
    assert any(dep.dep_type == LanguageDependencyType.IMPORTS for dep in parsed.dependencies)
    assert any(dep.dep_type == LanguageDependencyType.CALLS for dep in parsed.dependencies)
    assert adapter.extract_dependencies(parsed) == parsed.dependencies
    context = ProjectContext(root=str(tmp_path), all_sources=[parsed])
    resolved = adapter.resolve_symbols(parsed, context)
    graph = GraphBuilder(tmp_path).build_from_ir([resolved])
    assert len(graph.nodes) >= len(parsed.symbols)


@pytest.mark.parametrize("adapter_type,path", [
    (TypeScriptLanguageAdapter, "broken.ts"), (JavaScriptLanguageAdapter, "broken.js"),
    (CSharpLanguageAdapter, "Broken.cs"), (RustLanguageAdapter, "broken.rs"),
    (CppLanguageAdapter, "broken.cpp"),
])
def test_malformed_source_is_explicit(adapter_type: Type[LanguageAdapter], path: str) -> None:
    parsed = adapter_type().parse("class Broken { function f() {", path)
    assert parsed.has_errors
    assert parsed.language == adapter_type().language


def test_mixed_language_snapshot_resolves_and_builds(tmp_path: Path) -> None:
    sources = [
        default_registry.require("typescript").parse('import { helper } from "./helper"; export function run() { return helper(); }', "ui/app.ts"),
        default_registry.require("javascript").parse('function helper() { return 1; }', "ui/helper.js"),
        default_registry.require("rust").parse('fn native() {}', "native/lib.rs"),
    ]
    coordinator = create_default_coordinator(tmp_path)
    resolved = coordinator.resolve_all(sources, ProjectContext(str(tmp_path), sources))
    graph = GraphBuilder(tmp_path).build_from_ir(resolved)
    assert graph.nodes


def test_mixed_diff_never_uses_python_fallback() -> None:
    diff = """diff --git a/app.ts b/app.ts
new file mode 100644
--- /dev/null
+++ b/app.ts
@@ -0,0 +1 @@
+export function run() { return 1; }
diff --git a/readme.txt b/readme.txt
new file mode 100644
--- /dev/null
+++ b/readme.txt
@@ -0,0 +1 @@
+notes
"""
    source = {("app.ts", "new"): "export function run() { return 1; }", ("readme.txt", "new"): "notes"}
    change_set = ChangeAnalyzer(
        lambda path, version: source.get((path, version)),
        language_registry=default_registry,
    ).analyze_diff(diff)
    assert any(symbol.qualified_name.endswith("run") for symbol in change_set.changed_symbols)
    assert ("readme.txt", "unknown") in change_set.unsupported_files
    assert not any(symbol.file_path == "readme.txt" for symbol in change_set.changed_symbols)


@pytest.mark.parametrize(("language", "path", "source"), [
    ("typescript", "app.ts", "export function run() { return 1; }"),
    ("javascript", "app.js", "function run() { return 1; }"),
    ("csharp", "App.cs", "public class App { public void Run() {} }"),
    ("rust", "app.rs", "fn run() {}"),
    ("cpp", "app.cpp", "void run() {}"),
])
def test_each_adapter_enters_change_analyzer(language: str, path: str, source: str) -> None:
    diff = f"""diff --git a/{path} b/{path}
new file mode 100644
--- /dev/null
+++ b/{path}
@@ -0,0 +1 @@
+{source}
"""
    result = ChangeAnalyzer(
        lambda requested, version: source if (requested, version) == (path, "new") else None,
        language_registry=default_registry,
    ).analyze_diff(diff)
    assert not result.unsupported_files
    assert result.changed_symbols
    assert default_registry.require(language).language == language
