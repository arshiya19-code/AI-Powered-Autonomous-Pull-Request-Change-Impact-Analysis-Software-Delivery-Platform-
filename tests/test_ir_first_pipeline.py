"""Parity coverage for the Phase 7B Python IR-first pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from analyzer.ast_parser import ASTParser
from analyzer.change_analyzer import ChangeAnalyzer
from analyzer.graph_engine import CustomDependencyGraph, GraphBuilder
from analyzer.impact_scorer import analyze_and_score
from analyzer.languages.python.adapter import PythonLanguageAdapter
from analyzer.languages.registry import default_registry
from analyzer.resolver_coordinator import resolve_project


_OLD_A = """from b import helper\n\ndef run(value):\n    return helper(value)\n"""
_NEW_A = """from b import helper\n\ndef run(value):\n    return helper(value) + 1\n"""
_B = """def helper(value):\n    return value * 2\n"""
_DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -1,4 +1,4 @@
 from b import helper
 
 def run(value):
-    return helper(value)
+    return helper(value) + 1
"""


def _provider(file_path: str, version: str) -> Optional[str]:
    sources: Dict[tuple[str, str], str] = {
        ("a.py", "old"): _OLD_A,
        ("a.py", "new"): _NEW_A,
        ("b.py", "old"): _B,
        ("b.py", "new"): _B,
    }
    return sources.get((file_path, version))


def _edge_set(graph: CustomDependencyGraph) -> set[tuple[str, str, str]]:
    """Return comparable custom-graph edges without relying on internals."""
    return {
        (edge.source, edge.target, edge.edge_type.value)
        for edge in graph.edges
    }


def test_python_ir_first_pipeline_matches_legacy_pipeline(tmp_path: Path) -> None:
    """The adapter/coordinator path retains legacy Python graph semantics."""
    parser = ASTParser(include_docstrings=True, include_calls=True)
    legacy_modules = [
        parser.parse_source(_NEW_A, str(tmp_path / "a.py")),
        parser.parse_source(_B, str(tmp_path / "b.py")),
    ]
    legacy_graph = GraphBuilder(tmp_path).build_from_files(tmp_path, legacy_modules)

    adapter = PythonLanguageAdapter()
    ir_sources = [
        adapter.parse(_NEW_A, str(tmp_path / "a.py")),
        adapter.parse(_B, str(tmp_path / "b.py")),
    ]
    resolved_sources = resolve_project(tmp_path, ir_sources)
    ir_graph = GraphBuilder(tmp_path).build_from_ir(resolved_sources)

    assert set(legacy_graph.nodes) == set(ir_graph.nodes)
    assert _edge_set(legacy_graph) == _edge_set(ir_graph)
    assert {
        symbol.qualified_name for source in resolved_sources for symbol in source.symbols
    } == {"a", "a.run", "b", "b.helper"}

    legacy_changes = ChangeAnalyzer(source_provider=_provider).analyze_diff(_DIFF)
    ir_changes = ChangeAnalyzer(
        source_provider=_provider, language_registry=default_registry
    ).analyze_diff(_DIFF)
    assert [symbol.qualified_name for symbol in legacy_changes.changed_symbols] == [
        symbol.qualified_name for symbol in ir_changes.changed_symbols
    ]
    assert len(legacy_changes.dependency_changes) == len(ir_changes.dependency_changes)

    legacy_score = analyze_and_score(legacy_changes, legacy_graph)
    ir_score = analyze_and_score(ir_changes, ir_graph)
    assert [node.node_id for node in legacy_score.base_result.directly_changed_nodes] == [
        node.node_id for node in ir_score.base_result.directly_changed_nodes
    ]
    assert [node.node_id for node in legacy_score.base_result.impacted_nodes] == [
        node.node_id for node in ir_score.base_result.impacted_nodes
    ]
    assert legacy_score.impact_score == ir_score.impact_score
    assert legacy_score.features == ir_score.features
    assert len(ir_score.features) == 42


def test_python_file_adapter_preserves_resolution_source(tmp_path: Path) -> None:
    """File parsing retains the source needed by the Phase 3 resolver bridge."""
    a_path = tmp_path / "a.py"
    b_path = tmp_path / "b.py"
    a_path.write_text(_NEW_A, encoding="utf-8")
    b_path.write_text(_B, encoding="utf-8")

    adapter = PythonLanguageAdapter()
    resolved_sources = resolve_project(
        tmp_path,
        [adapter.parse_file(str(a_path)), adapter.parse_file(str(b_path))],
    )
    graph = GraphBuilder(tmp_path).build_from_ir(resolved_sources)

    assert ("a", "b.helper", "from_import") in _edge_set(graph)
