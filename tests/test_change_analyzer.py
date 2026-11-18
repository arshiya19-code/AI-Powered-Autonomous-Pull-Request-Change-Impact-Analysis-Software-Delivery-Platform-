"""
Tests for Phase 5 — Custom Change Analyzer.

Covers:
 1.  Empty diff
 2.  Single-line modification
 3.  Multiple-line modification
 4.  Added file
 5.  Deleted file
 6.  Multiple changed files
 7.  Function modification
 8.  Class modification
 9.  Method modification
10.  Import addition
11.  Import removal
12.  Dependency modification (swap import names)
13.  Direct graph impact
14.  Transitive graph impact
15.  Multiple impacted paths
16.  Cyclic dependency graph
17.  No impacted nodes
18.  Large change (many hunks)
19.  Multiple changed functions in one file
20.  Mixed file/function/class/dependency changes
21+. Additional edge-case tests
"""

from __future__ import annotations

import textwrap
from typing import Dict, Optional

import pytest

from analyzer.change_analyzer import (
    # data structures
    ChangeSet,
    ChangeType,
    ChangedSymbol,
    DependencyChange,
    DependencyChangeType,
    FileDiff,
    DiffHunk,
    ImpactAnalysisResult,
    ImpactedNode,
    ImpactLevel,
    SymbolType,
    # classes
    ChangeAnalyzer,
    DependencyChangeDetector,
    ImpactAnalyzer,
    SymbolMapper,
    UnifiedDiffParser,
    # convenience
    analyze_diff,
    analyze_diff_and_impact,
    analyze_impact,
)
from analyzer.graph_engine import (
    CustomDependencyGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_graph(*node_specs: tuple) -> CustomDependencyGraph:
    """
    Build a tiny graph from (node_id, NodeType) tuples.
    No edges are added by this helper — add them manually.
    """
    g = CustomDependencyGraph()
    for nid, ntype in node_specs:
        g.add_node(GraphNode(
            node_id=nid,
            node_type=ntype,
            name=nid.split(".")[-1],
            qualified_name=nid,
            file_path=nid.replace(".", "/") + ".py",
        ))
    return g


def _add_edge(g: CustomDependencyGraph, src: str, tgt: str) -> None:
    """Add a CALLS edge between src and tgt."""
    g.add_edge(GraphEdge(source=src, target=tgt, edge_type=EdgeType.CALLS))


def _add_import_edge(g: CustomDependencyGraph, src: str, tgt: str) -> None:
    """Add a FROM_IMPORT edge between src and tgt."""
    g.add_edge(GraphEdge(source=src, target=tgt, edge_type=EdgeType.FROM_IMPORT))


def _source_provider(sources: Dict[str, str]):
    """Return a simple in-memory source provider."""
    def provider(file_path: str, version: str) -> Optional[str]:
        key = f"{version}:{file_path}"
        if key in sources:
            return sources[key]
        return sources.get(file_path)  # version-agnostic fallback
    return provider


# ===========================================================================
# Part 1 — UnifiedDiffParser unit tests
# ===========================================================================

class TestUnifiedDiffParser:
    """Tests for the low-level diff text parser."""

    def test_empty_string(self):
        """Test 1 — empty diff produces no FileDiffs."""
        parser = UnifiedDiffParser()
        result = parser.parse("")
        assert result == []

    def test_whitespace_only(self):
        parser = UnifiedDiffParser()
        result = parser.parse("   \n\n   ")
        assert result == []

    def test_single_file_modification(self):
        """Test 2 — single-line modification parsed correctly."""
        diff = textwrap.dedent("""\
            diff --git a/payment.py b/payment.py
            --- a/payment.py
            +++ b/payment.py
            @@ -10,3 +10,3 @@
             def old():
            -    return 1
            +    return 2
             pass
        """)
        parser = UnifiedDiffParser()
        result = parser.parse(diff)
        assert len(result) == 1
        fd = result[0]
        assert fd.file_path == "payment.py"
        assert fd.change_type == ChangeType.MODIFIED
        assert len(fd.hunks) == 1
        hunk = fd.hunks[0]
        assert hunk.old_start == 10
        assert hunk.new_start == 10
        assert fd.lines_added == 1
        assert fd.lines_deleted == 1

    def test_added_file(self):
        """Test 4 — added file detection."""
        diff = textwrap.dedent("""\
            diff --git a/new_module.py b/new_module.py
            new file mode 100644
            --- /dev/null
            +++ b/new_module.py
            @@ -0,0 +1,5 @@
            +def hello():
            +    pass
        """)
        parser = UnifiedDiffParser()
        result = parser.parse(diff)
        assert len(result) == 1
        fd = result[0]
        assert fd.change_type == ChangeType.ADDED
        assert fd.file_path == "new_module.py"
        assert fd.lines_added == 2

    def test_deleted_file(self):
        """Test 5 — deleted file detection."""
        diff = textwrap.dedent("""\
            diff --git a/old_module.py b/old_module.py
            deleted file mode 100644
            --- a/old_module.py
            +++ /dev/null
            @@ -1,3 +0,0 @@
            -def gone():
            -    pass
        """)
        parser = UnifiedDiffParser()
        result = parser.parse(diff)
        assert len(result) == 1
        fd = result[0]
        assert fd.change_type == ChangeType.DELETED
        assert fd.lines_deleted == 2
        assert fd.lines_added == 0

    def test_multiple_files(self):
        """Test 6 — multiple files in one diff."""
        diff = textwrap.dedent("""\
            diff --git a/a.py b/a.py
            --- a/a.py
            +++ b/a.py
            @@ -1,2 +1,2 @@
            -x = 1
            +x = 2
             pass
            diff --git a/b.py b/b.py
            --- a/b.py
            +++ b/b.py
            @@ -5,2 +5,3 @@
             y = 1
            +z = 3
             pass
        """)
        parser = UnifiedDiffParser()
        result = parser.parse(diff)
        assert len(result) == 2
        paths = {fd.file_path for fd in result}
        assert "a.py" in paths
        assert "b.py" in paths

    def test_multiple_hunks_same_file(self):
        """Test 3 — multiple hunks in one file."""
        diff = textwrap.dedent("""\
            diff --git a/utils.py b/utils.py
            --- a/utils.py
            +++ b/utils.py
            @@ -5,3 +5,3 @@
            -a = 1
            +a = 2
             pass
            @@ -20,3 +20,4 @@
             b = 1
            +c = 3
             pass
        """)
        parser = UnifiedDiffParser()
        result = parser.parse(diff)
        assert len(result) == 1
        fd = result[0]
        assert len(fd.hunks) == 2

    def test_hunk_line_numbers(self):
        """Hunk added/deleted line numbers are accurate."""
        diff = textwrap.dedent("""\
            diff --git a/calc.py b/calc.py
            --- a/calc.py
            +++ b/calc.py
            @@ -3,4 +3,4 @@
             line3
            -line4_old
            +line4_new
             line5
        """)
        parser = UnifiedDiffParser()
        result = parser.parse(diff)
        hunk = result[0].hunks[0]
        assert 4 in hunk.added_line_numbers
        assert 4 in hunk.deleted_line_numbers

    def test_renamed_file(self):
        """Renamed file detected with correct old and new paths."""
        diff = textwrap.dedent("""\
            diff --git a/old_name.py b/new_name.py
            rename from old_name.py
            rename to new_name.py
            --- a/old_name.py
            +++ b/new_name.py
            @@ -1,2 +1,2 @@
            -x = 1
            +x = 2
        """)
        parser = UnifiedDiffParser()
        result = parser.parse(diff)
        assert len(result) == 1
        fd = result[0]
        assert fd.change_type == ChangeType.RENAMED
        assert fd.file_path == "new_name.py"
        assert fd.old_path == "old_name.py"

    def test_hunk_affected_ranges(self):
        """affected_new_ranges returns correct (start, end) tuples."""
        diff = textwrap.dedent("""\
            diff --git a/svc.py b/svc.py
            --- a/svc.py
            +++ b/svc.py
            @@ -10,5 +10,6 @@
             a
            +b
             c
             d
             e
        """)
        parser = UnifiedDiffParser()
        fd = parser.parse(diff)[0]
        ranges = fd.affected_new_ranges
        assert len(ranges) == 1
        start, end = ranges[0]
        assert start == 10
        assert end == 15  # 10 + 6 - 1


# ===========================================================================
# Part 2 — SymbolMapper unit tests
# ===========================================================================

class TestSymbolMapper:
    """Tests for AST-based line→symbol mapping."""

    _SOURCE = textwrap.dedent("""\
        class PaymentService:
            def process(self):
                return 1

            def refund(self):
                return 2

        def helper():
            pass
    """)

    def test_method_detected(self):
        """Test 9 — method modification detected by line range."""
        mapper = SymbolMapper()
        symbols = mapper.map_changed_lines(
            "payment.py",
            [(2, 3)],        # process() is on lines 2–3
            self._SOURCE,
            "payment",
            ChangeType.MODIFIED,
        )
        names = {s.qualified_name for s in symbols}
        assert any("process" in n for n in names)

    def test_class_detected(self):
        """Test 8 — class detected when line falls in class body."""
        mapper = SymbolMapper()
        symbols = mapper.map_changed_lines(
            "payment.py",
            [(1, 1)],        # class declaration line
            self._SOURCE,
            "payment",
            ChangeType.MODIFIED,
        )
        class_syms = [s for s in symbols if s.symbol_type == SymbolType.CLASS]
        assert len(class_syms) >= 1
        assert any("PaymentService" in s.qualified_name for s in class_syms)

    def test_function_detected(self):
        """Test 7 — standalone function detected."""
        mapper = SymbolMapper()
        symbols = mapper.map_changed_lines(
            "payment.py",
            [(8, 9)],        # helper() is on lines 8–9
            self._SOURCE,
            "payment",
            ChangeType.MODIFIED,
        )
        func_syms = [s for s in symbols if s.symbol_type == SymbolType.FUNCTION]
        assert any("helper" in s.qualified_name for s in func_syms)

    def test_empty_line_ranges(self):
        """Empty line ranges produce no symbols."""
        mapper = SymbolMapper()
        result = mapper.map_changed_lines("x.py", [], self._SOURCE, "x", ChangeType.MODIFIED)
        assert result == []

    def test_empty_source(self):
        """Empty source produces no symbols."""
        mapper = SymbolMapper()
        result = mapper.map_changed_lines("x.py", [(1, 5)], "", "x", ChangeType.MODIFIED)
        assert result == []

    def test_multiple_functions_in_file(self):
        """Test 19 — multiple changed functions in one file detected."""
        source = textwrap.dedent("""\
            def foo():
                return 1

            def bar():
                return 2

            def baz():
                return 3
        """)
        mapper = SymbolMapper()
        # Lines 1–8 cover all three functions
        symbols = mapper.map_changed_lines(
            "funcs.py",
            [(1, 8)],
            source,
            "funcs",
            ChangeType.MODIFIED,
        )
        names = {s.qualified_name for s in symbols}
        assert "funcs.foo" in names
        assert "funcs.bar" in names
        assert "funcs.baz" in names

    def test_no_overlap_gives_no_symbols(self):
        """Line range that doesn't overlap any symbol gives empty result."""
        mapper = SymbolMapper()
        symbols = mapper.map_changed_lines(
            "payment.py",
            [(100, 110)],
            self._SOURCE,
            "payment",
            ChangeType.MODIFIED,
        )
        assert symbols == []

    def test_method_qualified_name_format(self):
        """Method qualified name is module.Class.method."""
        mapper = SymbolMapper()
        symbols = mapper.map_changed_lines(
            "payment.py",
            [(5, 6)],   # refund() lines
            self._SOURCE,
            "services.payment",
            ChangeType.MODIFIED,
        )
        method_syms = [s for s in symbols if s.symbol_type == SymbolType.METHOD]
        assert any(s.qualified_name == "services.payment.PaymentService.refund"
                   for s in method_syms)


# ===========================================================================
# Part 3 — DependencyChangeDetector unit tests
# ===========================================================================

class TestDependencyChangeDetector:
    """Tests for import-level change detection."""

    def test_import_addition(self):
        """Test 10 — new import detected as IMPORT_ADDED."""
        old = "from services.auth import validate_token\n"
        new  = "from services.auth import validate_token\nfrom utils import log\n"
        detector = DependencyChangeDetector()
        changes = detector.detect("payment.py", old, new)
        added = [c for c in changes if c.change_type == DependencyChangeType.IMPORT_ADDED]
        assert any(c.module == "utils" for c in added)

    def test_import_removal(self):
        """Test 11 — removed import detected as IMPORT_REMOVED."""
        old = "from services.auth import validate_token\nfrom utils import log\n"
        new  = "from services.auth import validate_token\n"
        detector = DependencyChangeDetector()
        changes = detector.detect("payment.py", old, new)
        removed = [c for c in changes if c.change_type == DependencyChangeType.IMPORT_REMOVED]
        assert any(c.module == "utils" for c in removed)

    def test_dependency_modification_swap(self):
        """
        Test 12 — swapping an imported name (old removed + new added).
        from services.auth import validate_token → from services.auth import validate_user
        """
        old = "from services.auth import validate_token\n"
        new  = "from services.auth import validate_user\n"
        detector = DependencyChangeDetector()
        changes = detector.detect("payment.py", old, new)
        types = {c.change_type for c in changes}
        # One removal + one addition
        assert DependencyChangeType.IMPORT_REMOVED in types
        assert DependencyChangeType.IMPORT_ADDED in types

    def test_no_change(self):
        """Identical imports produce no dependency changes."""
        source = "from services.auth import validate_token\n"
        detector = DependencyChangeDetector()
        changes = detector.detect("payment.py", source, source)
        assert changes == []

    def test_added_file_has_no_old(self):
        """Added file: only IMPORT_ADDED records, no removals."""
        detector = DependencyChangeDetector()
        changes = detector.detect(
            "new.py",
            None,
            "import os\nfrom pathlib import Path\n",
        )
        assert all(c.change_type == DependencyChangeType.IMPORT_ADDED for c in changes)

    def test_deleted_file_has_no_new(self):
        """Deleted file: only IMPORT_REMOVED records, no additions."""
        detector = DependencyChangeDetector()
        changes = detector.detect(
            "old.py",
            "import os\nfrom pathlib import Path\n",
            None,
        )
        assert all(c.change_type == DependencyChangeType.IMPORT_REMOVED for c in changes)


# ===========================================================================
# Part 4 — ChangeAnalyzer integration tests
# ===========================================================================

class TestChangeAnalyzer:
    """Integration tests for the main ChangeAnalyzer."""

    def test_empty_diff(self):
        """Test 1 — empty diff returns empty ChangeSet."""
        cs = analyze_diff("")
        assert cs.files_changed_count == 0
        assert cs.lines_added == 0
        assert cs.lines_deleted == 0
        assert cs.changed_symbols == []

    def test_single_line_modification(self):
        """Test 2 — single-line modification tracked correctly."""
        diff = textwrap.dedent("""\
            diff --git a/a.py b/a.py
            --- a/a.py
            +++ b/a.py
            @@ -5,3 +5,3 @@
             x = 1
            -y = 2
            +y = 3
             z = 4
        """)
        cs = analyze_diff(diff)
        assert len(cs.files_modified) == 1
        assert cs.lines_added == 1
        assert cs.lines_deleted == 1
        assert cs.total_lines_changed == 2

    def test_multiple_line_modification(self):
        """Test 3 — multi-line modification accumulates counts."""
        diff = textwrap.dedent("""\
            diff --git a/b.py b/b.py
            --- a/b.py
            +++ b/b.py
            @@ -1,5 +1,7 @@
            -line1
            -line2
            -line3
            +new1
            +new2
            +new3
            +new4
             stay
        """)
        cs = analyze_diff(diff)
        assert cs.lines_added == 4
        assert cs.lines_deleted == 3

    def test_added_file(self):
        """Test 4 — added file lands in files_added."""
        diff = textwrap.dedent("""\
            diff --git a/new_mod.py b/new_mod.py
            new file mode 100644
            --- /dev/null
            +++ b/new_mod.py
            @@ -0,0 +1,3 @@
            +def greet():
            +    return 'hi'
        """)
        cs = analyze_diff(diff)
        assert "new_mod.py" in cs.files_added
        assert cs.files_modified == []

    def test_deleted_file(self):
        """Test 5 — deleted file lands in files_deleted."""
        diff = textwrap.dedent("""\
            diff --git a/old_mod.py b/old_mod.py
            deleted file mode 100644
            --- a/old_mod.py
            +++ /dev/null
            @@ -1,2 +0,0 @@
            -def bye():
            -    pass
        """)
        cs = analyze_diff(diff)
        assert "old_mod.py" in cs.files_deleted
        assert cs.files_modified == []
        assert cs.lines_deleted == 2

    def test_multiple_changed_files(self):
        """Test 6 — multiple files all accounted for."""
        diff = textwrap.dedent("""\
            diff --git a/a.py b/a.py
            --- a/a.py
            +++ b/a.py
            @@ -1,2 +1,2 @@
            -x=1
            +x=2
             pass
            diff --git a/b.py b/b.py
            new file mode 100644
            --- /dev/null
            +++ b/b.py
            @@ -0,0 +1,2 @@
            +y=3
            +pass
            diff --git a/c.py b/c.py
            deleted file mode 100644
            --- a/c.py
            +++ /dev/null
            @@ -1,1 +0,0 @@
            -z=4
        """)
        cs = analyze_diff(diff)
        assert cs.files_changed_count == 3
        assert "a.py" in cs.files_modified
        assert "b.py" in cs.files_added
        assert "c.py" in cs.files_deleted

    def test_function_modification_with_source(self):
        """Test 7 — function modification detected via source provider."""
        source = textwrap.dedent("""\
            def compute(x):
                return x + 1
        """)
        diff = textwrap.dedent("""\
            diff --git a/calc.py b/calc.py
            --- a/calc.py
            +++ b/calc.py
            @@ -1,2 +1,2 @@
             def compute(x):
            -    return x + 1
            +    return x + 2
        """)
        sources = {"new:calc.py": source, "old:calc.py": source}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        funcs = [s for s in cs.changed_symbols if s.symbol_type == SymbolType.FUNCTION]
        assert any("compute" in s.qualified_name for s in funcs)

    def test_class_modification_with_source(self):
        """Test 8 — class modification detected via source provider."""
        source = textwrap.dedent("""\
            class MyClass:
                x = 1
        """)
        diff = textwrap.dedent("""\
            diff --git a/mymod.py b/mymod.py
            --- a/mymod.py
            +++ b/mymod.py
            @@ -1,2 +1,2 @@
             class MyClass:
            -    x = 1
            +    x = 2
        """)
        sources = {"new:mymod.py": source, "old:mymod.py": source}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        class_syms = [s for s in cs.changed_symbols if s.symbol_type == SymbolType.CLASS]
        assert any("MyClass" in s.qualified_name for s in class_syms)

    def test_method_modification_with_source(self):
        """Test 9 — method modification detected via source provider."""
        source = textwrap.dedent("""\
            class Service:
                def run(self):
                    pass
        """)
        diff = textwrap.dedent("""\
            diff --git a/svc.py b/svc.py
            --- a/svc.py
            +++ b/svc.py
            @@ -2,3 +2,3 @@
                 def run(self):
            -        pass
            +        return True
        """)
        sources = {"new:svc.py": source, "old:svc.py": source}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        method_syms = [s for s in cs.changed_symbols if s.symbol_type == SymbolType.METHOD]
        assert any("run" in s.qualified_name for s in method_syms)

    def test_import_addition_detected(self):
        """Test 10 — new import in diff creates dependency change."""
        old_src = "from utils import log\n"
        new_src = "from utils import log\nfrom pathlib import Path\n"
        diff = textwrap.dedent("""\
            diff --git a/handler.py b/handler.py
            --- a/handler.py
            +++ b/handler.py
            @@ -1,1 +1,2 @@
             from utils import log
            +from pathlib import Path
        """)
        sources = {"old:handler.py": old_src, "new:handler.py": new_src}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        added = [d for d in cs.dependency_changes
                 if d.change_type == DependencyChangeType.IMPORT_ADDED]
        assert any("pathlib" in d.module for d in added)

    def test_import_removal_detected(self):
        """Test 11 — removed import creates IMPORT_REMOVED dependency change."""
        old_src = "from utils import log\nfrom pathlib import Path\n"
        new_src = "from utils import log\n"
        diff = textwrap.dedent("""\
            diff --git a/handler.py b/handler.py
            --- a/handler.py
            +++ b/handler.py
            @@ -1,2 +1,1 @@
             from utils import log
            -from pathlib import Path
        """)
        sources = {"old:handler.py": old_src, "new:handler.py": new_src}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        removed = [d for d in cs.dependency_changes
                   if d.change_type == DependencyChangeType.IMPORT_REMOVED]
        assert any("pathlib" in d.module for d in removed)

    def test_dependency_modification(self):
        """Test 12 — swap imported name → one removal + one addition."""
        old_src = "from services.auth import validate_token\n"
        new_src = "from services.auth import validate_user\n"
        diff = textwrap.dedent("""\
            diff --git a/pay.py b/pay.py
            --- a/pay.py
            +++ b/pay.py
            @@ -1,1 +1,1 @@
            -from services.auth import validate_token
            +from services.auth import validate_user
        """)
        sources = {"old:pay.py": old_src, "new:pay.py": new_src}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        types = {d.change_type for d in cs.dependency_changes}
        assert DependencyChangeType.IMPORT_REMOVED in types
        assert DependencyChangeType.IMPORT_ADDED in types

    def test_mixed_changes(self):
        """Test 20 — multiple files with functions, classes, and dependency changes."""
        src_a_old = "import os\n\ndef func_a():\n    return 1\n"
        src_a_new = "import os\nimport sys\n\ndef func_a():\n    return 2\n"
        src_b_new = "class Widget:\n    def render(self):\n        pass\n"

        diff = textwrap.dedent("""\
            diff --git a/a.py b/a.py
            --- a/a.py
            +++ b/a.py
            @@ -1,4 +1,5 @@
             import os
            +import sys
             
             def func_a():
            -    return 1
            +    return 2
            diff --git a/b.py b/b.py
            new file mode 100644
            --- /dev/null
            +++ b/b.py
            @@ -0,0 +1,3 @@
            +class Widget:
            +    def render(self):
            +        pass
        """)
        sources = {
            "old:a.py": src_a_old,
            "new:a.py": src_a_new,
            "new:b.py": src_b_new,
        }
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        assert "a.py" in cs.files_modified
        assert "b.py" in cs.files_added
        # Dependency change: sys import added
        added = [d for d in cs.dependency_changes
                 if d.change_type == DependencyChangeType.IMPORT_ADDED]
        assert any("sys" in d.module for d in added)

    def test_changeset_computed_properties(self):
        """ChangeSet computed properties stay consistent."""
        diff = textwrap.dedent("""\
            diff --git a/x.py b/x.py
            --- a/x.py
            +++ b/x.py
            @@ -1,2 +1,2 @@
            -a=1
            +a=2
        """)
        cs = analyze_diff(diff)
        assert cs.total_lines_changed == cs.lines_added + cs.lines_deleted
        assert cs.files_changed_count == 1
        assert "x.py" in cs.changed_files

    def test_multiple_changed_functions_in_one_file(self):
        """Test 19 — multiple functions in one file all detected."""
        source = textwrap.dedent("""\
            def alpha():
                return 1


            def beta():
                return 2


            def gamma():
                return 3
        """)
        diff = textwrap.dedent("""\
            diff --git a/funcs.py b/funcs.py
            --- a/funcs.py
            +++ b/funcs.py
            @@ -1,10 +1,10 @@
            -def alpha():
            -    return 1
            +def alpha():
            +    return 10
             
             
            -def beta():
            -    return 2
            +def beta():
            +    return 20
             
             
             def gamma():
        """)
        sources = {"new:funcs.py": source, "old:funcs.py": source}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        func_names = {s.qualified_name for s in cs.changed_functions}
        assert "funcs.alpha" in func_names
        assert "funcs.beta" in func_names


# ===========================================================================
# Part 5 — ImpactAnalyzer tests
# ===========================================================================

class TestImpactAnalyzer:
    """Tests for graph-based impact propagation."""

    def _build_linear_graph(self) -> CustomDependencyGraph:
        """A → B → C → D (linear chain)."""
        g = _make_graph(
            ("A", NodeType.FUNCTION),
            ("B", NodeType.FUNCTION),
            ("C", NodeType.FUNCTION),
            ("D", NodeType.FUNCTION),
        )
        _add_edge(g, "B", "A")  # B depends on A
        _add_edge(g, "C", "B")  # C depends on B
        _add_edge(g, "D", "C")  # D depends on C
        return g

    def test_direct_graph_impact(self):
        """Test 13 — directly changed node identified."""
        g = self._build_linear_graph()

        diff = textwrap.dedent("""\
            diff --git a/A.py b/A.py
            --- a/A.py
            +++ b/A.py
            @@ -1,2 +1,2 @@
            -x=1
            +x=2
        """)
        cs = analyze_diff(diff)
        # Manually mark A as changed
        cs.changed_node_ids = {"A"}
        # Add a minimal symbol so impact analyzer finds the node
        cs.changed_symbols.append(ChangedSymbol(
            qualified_name="A",
            symbol_type=SymbolType.FUNCTION,
            file_path="A.py",
            change_type=ChangeType.MODIFIED,
            module_name="A",
        ))

        result = analyze_impact(cs, g)
        direct_ids = {n.node_id for n in result.directly_changed_nodes}
        assert "A" in direct_ids

    def test_transitive_graph_impact(self):
        """Test 14 — transitive dependents propagated correctly."""
        g = self._build_linear_graph()
        cs = ChangeSet()
        cs.changed_symbols.append(ChangedSymbol(
            qualified_name="A",
            symbol_type=SymbolType.FUNCTION,
            file_path="A.py",
            change_type=ChangeType.MODIFIED,
            module_name="A",
        ))

        result = analyze_impact(cs, g)
        impacted_ids = {n.node_id for n in result.impacted_nodes}
        # B is direct dependent (depth 1)
        # C is transitive (depth 2)
        # D is transitive (depth 3)
        assert "B" in impacted_ids
        assert "C" in impacted_ids
        assert "D" in impacted_ids
        assert result.maximum_impact_depth >= 3

    def test_multiple_impacted_paths(self):
        """Test 15 — diamond dependency: C depends on both A and B, both changed."""
        g = _make_graph(
            ("A", NodeType.FUNCTION),
            ("B", NodeType.FUNCTION),
            ("C", NodeType.FUNCTION),
        )
        _add_edge(g, "C", "A")
        _add_edge(g, "C", "B")

        cs = ChangeSet()
        cs.changed_symbols.extend([
            ChangedSymbol("A", SymbolType.FUNCTION, "A.py", ChangeType.MODIFIED, "A"),
            ChangedSymbol("B", SymbolType.FUNCTION, "B.py", ChangeType.MODIFIED, "B"),
        ])

        result = analyze_impact(cs, g)
        impacted_ids = {n.node_id for n in result.impacted_nodes}
        assert "C" in impacted_ids

    def test_cyclic_dependency_graph(self):
        """Test 16 — cycle handled safely without infinite loop."""
        g = _make_graph(
            ("X", NodeType.FUNCTION),
            ("Y", NodeType.FUNCTION),
            ("Z", NodeType.FUNCTION),
        )
        _add_edge(g, "Y", "X")
        _add_edge(g, "Z", "Y")
        _add_edge(g, "X", "Z")  # cycle: Z→Y→X→Z

        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol("X", SymbolType.FUNCTION, "X.py", ChangeType.MODIFIED, "X")
        )
        # Should terminate (not infinite loop)
        result = analyze_impact(cs, g)
        assert isinstance(result, ImpactAnalysisResult)

    def test_no_impacted_nodes(self):
        """Test 17 — isolated node has no dependents."""
        g = _make_graph(("solo", NodeType.FUNCTION))
        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol("solo", SymbolType.FUNCTION, "solo.py", ChangeType.MODIFIED, "solo")
        )
        result = analyze_impact(cs, g)
        assert len(result.impacted_nodes) == 0
        assert result.maximum_impact_depth == 0

    def test_impact_depth_classification(self):
        """Depth-1 nodes are DIRECT_DEPENDENT; depth-2+ are TRANSITIVE."""
        g = self._build_linear_graph()
        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol("A", SymbolType.FUNCTION, "A.py", ChangeType.MODIFIED, "A")
        )
        result = analyze_impact(cs, g)

        depth_map = result.impact_depth_map
        b_node = next((n for n in result.impacted_nodes if n.node_id == "B"), None)
        c_node = next((n for n in result.impacted_nodes if n.node_id == "C"), None)

        assert b_node is not None
        assert b_node.impact_level == ImpactLevel.DIRECT_DEPENDENT
        assert b_node.impact_depth == 1

        assert c_node is not None
        assert c_node.impact_level == ImpactLevel.TRANSITIVE_DEPENDENT
        assert c_node.impact_depth == 2

    def test_impact_type_breakdown(self):
        """impacted_modules / functions / classes / methods populated correctly."""
        g = _make_graph(
            ("mod", NodeType.MODULE),
            ("mod.MyClass", NodeType.CLASS),
            ("mod.MyClass.method", NodeType.METHOD),
            ("lib", NodeType.MODULE),
        )
        _add_edge(g, "lib", "mod")

        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol("mod", SymbolType.MODULE, "mod.py", ChangeType.MODIFIED, "mod")
        )
        result = analyze_impact(cs, g)
        impacted_types = {n.node_id: n.node_type for n in result.impacted_nodes}
        # lib imports mod so lib should be impacted
        assert "lib" in impacted_types
        assert impacted_types["lib"] == NodeType.MODULE
        assert len(result.impacted_modules) >= 1

    def test_features_computed(self):
        """Feature dict contains all expected keys with sensible values."""
        g = self._build_linear_graph()
        cs = ChangeSet()
        cs.files_modified.append("A.py")
        cs.lines_added = 5
        cs.lines_deleted = 3
        cs.changed_symbols.append(
            ChangedSymbol("A", SymbolType.FUNCTION, "A.py", ChangeType.MODIFIED, "A")
        )
        result = analyze_impact(cs, g)
        f = result.features

        expected_keys = {
            "files_changed", "lines_added", "lines_deleted", "total_lines_changed",
            "functions_changed", "classes_changed", "methods_changed",
            "dependencies_added", "dependencies_removed", "dependencies_changed",
            "direct_nodes_changed", "impacted_nodes",
            "impacted_modules", "impacted_functions", "impacted_classes", "impacted_methods",
            "maximum_impact_depth", "impact_ratio", "dependency_change_ratio",
            "average_impact_depth",
        }
        assert expected_keys.issubset(set(f.keys()))

        # Basic sanity checks
        assert f["files_changed"] == 1.0
        assert f["lines_added"] == 5.0
        assert f["lines_deleted"] == 3.0
        assert f["total_lines_changed"] == 8.0
        assert f["impacted_nodes"] >= 3.0  # B, C, D
        assert 0.0 <= f["impact_ratio"] <= 1.0

    def test_large_change(self):
        """Test 18 — large change: many symbols and dependencies handled efficiently."""
        # Build a star graph: hub with 50 dependents
        g = _make_graph(("hub", NodeType.FUNCTION))
        for i in range(50):
            nid = f"spoke_{i}"
            g.add_node(GraphNode(
                node_id=nid, node_type=NodeType.FUNCTION,
                name=nid, qualified_name=nid, file_path=f"{nid}.py"
            ))
            _add_edge(g, nid, "hub")

        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol("hub", SymbolType.FUNCTION, "hub.py", ChangeType.MODIFIED, "hub")
        )
        result = analyze_impact(cs, g)
        assert len(result.impacted_nodes) == 50

    def test_empty_changeset_on_graph(self):
        """Empty ChangeSet on populated graph gives empty impact."""
        g = self._build_linear_graph()
        cs = ChangeSet()
        result = analyze_impact(cs, g)
        assert len(result.directly_changed_nodes) == 0
        assert len(result.impacted_nodes) == 0

    def test_feature_impact_ratio_bounded(self):
        """impact_ratio is always in [0, 1]."""
        g = self._build_linear_graph()
        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol("A", SymbolType.FUNCTION, "A.py", ChangeType.MODIFIED, "A")
        )
        result = analyze_impact(cs, g)
        assert 0.0 <= result.features["impact_ratio"] <= 1.0


# ===========================================================================
# Part 6 — End-to-end convenience function tests
# ===========================================================================

class TestConvenienceFunctions:
    """Tests for the module-level convenience functions."""

    def test_analyze_diff_returns_changeset(self):
        diff = textwrap.dedent("""\
            diff --git a/x.py b/x.py
            --- a/x.py
            +++ b/x.py
            @@ -1,1 +1,1 @@
            -a=1
            +a=2
        """)
        cs = analyze_diff(diff)
        assert isinstance(cs, ChangeSet)

    def test_analyze_impact_returns_result(self):
        g = _make_graph(("f", NodeType.FUNCTION))
        cs = ChangeSet()
        result = analyze_impact(cs, g)
        assert isinstance(result, ImpactAnalysisResult)

    def test_analyze_diff_and_impact_tuple(self):
        g = _make_graph(("f", NodeType.FUNCTION))
        diff = textwrap.dedent("""\
            diff --git a/f.py b/f.py
            --- a/f.py
            +++ b/f.py
            @@ -1,1 +1,1 @@
            -a=1
            +a=2
        """)
        cs, result = analyze_diff_and_impact(diff, g)
        assert isinstance(cs, ChangeSet)
        assert isinstance(result, ImpactAnalysisResult)


# ===========================================================================
# Part 7 — Additional edge-case tests
# ===========================================================================

class TestEdgeCases:
    """Additional robustness and edge-case tests."""

    def test_diff_with_no_python_files(self):
        """Non-Python file diffs are parsed without crashing."""
        diff = textwrap.dedent("""\
            diff --git a/README.md b/README.md
            --- a/README.md
            +++ b/README.md
            @@ -1,1 +1,2 @@
             # Title
            +New line
        """)
        cs = analyze_diff(diff)
        assert cs.files_changed_count >= 0  # Should not crash

    def test_changeset_repr(self):
        """ChangeSet has a useful repr."""
        cs = ChangeSet()
        r = repr(cs)
        assert "ChangeSet" in r

    def test_impact_result_repr(self):
        """ImpactAnalysisResult has a useful repr."""
        result = ImpactAnalysisResult()
        r = repr(result)
        assert "ImpactAnalysisResult" in r

    def test_changed_symbol_repr(self):
        """ChangedSymbol has a useful repr."""
        sym = ChangedSymbol("m.f", SymbolType.FUNCTION, "m.py", ChangeType.MODIFIED, "m")
        r = repr(sym)
        assert "ChangedSymbol" in r

    def test_file_diff_line_counts_consistency(self):
        """FileDiff.lines_added and lines_deleted are consistent with hunks."""
        diff = textwrap.dedent("""\
            diff --git a/t.py b/t.py
            --- a/t.py
            +++ b/t.py
            @@ -1,3 +1,4 @@
             a
            -b
            +c
            +d
             e
        """)
        parser = UnifiedDiffParser()
        fd = parser.parse(diff)[0]
        assert fd.lines_added == 2
        assert fd.lines_deleted == 1

    def test_diff_hunk_affected_range_zero_count(self):
        """DiffHunk with count=0 returns sensible range."""
        hunk = DiffHunk(old_start=5, old_count=0, new_start=5, new_count=0)
        assert hunk.affected_new_range == (5, 5)
        assert hunk.affected_old_range == (5, 5)

    def test_module_name_resolver_with_path_separator(self):
        """Default module name resolver handles nested paths."""
        name = ChangeAnalyzer._default_module_name("services/payment/processor.py")
        assert name == "services.payment.processor"

    def test_module_name_resolver_init_py(self):
        """Default module name resolver strips __init__.py."""
        name = ChangeAnalyzer._default_module_name("services/payment/__init__.py")
        assert name == "services.payment"

    def test_no_source_provider_no_crash(self):
        """Analyzer without source provider still works for file/line counts."""
        diff = textwrap.dedent("""\
            diff --git a/a.py b/a.py
            --- a/a.py
            +++ b/a.py
            @@ -1,2 +1,2 @@
            -x=1
            +x=2
        """)
        cs = ChangeAnalyzer().analyze_diff(diff)
        assert cs.lines_added == 1
        assert cs.lines_deleted == 1

    def test_graph_node_not_in_graph_skipped(self):
        """Changed symbol with no matching graph node is handled gracefully."""
        g = _make_graph(("existing", NodeType.FUNCTION))
        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol(
                qualified_name="nonexistent.function",
                symbol_type=SymbolType.FUNCTION,
                file_path="nonexistent.py",
                change_type=ChangeType.MODIFIED,
                module_name="nonexistent",
            )
        )
        result = analyze_impact(cs, g)
        # Should not crash; no impacted nodes since "existing" is unrelated
        assert isinstance(result, ImpactAnalysisResult)

    def test_dependency_change_repr(self):
        """DependencyChange has a useful repr."""
        from analyzer.ast_parser import ImportType as _ImportType
        dc = DependencyChange(
            file_path="p.py",
            change_type=DependencyChangeType.IMPORT_ADDED,
            dep_type=_ImportType.FROM_IMPORT,
            module="services",
            names=["run"],
        )
        assert "DependencyChange" in repr(dc)

    def test_added_file_all_symbols_marked_added(self):
        """All symbols in an added file are marked ADDED."""
        src = textwrap.dedent("""\
            class A:
                def m(self):
                    pass

            def f():
                pass
        """)
        diff = textwrap.dedent("""\
            diff --git a/brand_new.py b/brand_new.py
            new file mode 100644
            --- /dev/null
            +++ b/brand_new.py
            @@ -0,0 +1,7 @@
            +class A:
            +    def m(self):
            +        pass
            +
            +def f():
            +    pass
        """)
        sources = {"new:brand_new.py": src}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        for sym in cs.changed_symbols:
            assert sym.change_type == ChangeType.ADDED

    def test_deleted_file_all_symbols_marked_deleted(self):
        """All symbols in a deleted file are marked DELETED."""
        src = textwrap.dedent("""\
            def orphan():
                pass
        """)
        diff = textwrap.dedent("""\
            diff --git a/legacy.py b/legacy.py
            deleted file mode 100644
            --- a/legacy.py
            +++ /dev/null
            @@ -1,2 +0,0 @@
            -def orphan():
            -    pass
        """)
        sources = {"old:legacy.py": src}
        cs = analyze_diff(diff, source_provider=_source_provider(sources))
        for sym in cs.changed_symbols:
            assert sym.change_type == ChangeType.DELETED

    def test_average_impact_depth_zero_with_no_impact(self):
        """average_impact_depth is 0 when there are no impacted nodes."""
        result = ImpactAnalysisResult()
        assert result.average_impact_depth == 0.0

    def test_maximum_impact_depth_zero_with_no_depth_map(self):
        """maximum_impact_depth is 0 when depth map is empty."""
        result = ImpactAnalysisResult()
        assert result.maximum_impact_depth == 0

    def test_impact_analyzer_affected_count_properties(self):
        """ImpactAnalysisResult.affected_*_count reflects direct + impacted."""
        g = _make_graph(
            ("fn_a", NodeType.FUNCTION),
            ("fn_b", NodeType.FUNCTION),
        )
        _add_edge(g, "fn_b", "fn_a")
        cs = ChangeSet()
        cs.changed_symbols.append(
            ChangedSymbol("fn_a", SymbolType.FUNCTION, "fn_a.py", ChangeType.MODIFIED, "fn_a")
        )
        result = analyze_impact(cs, g)
        # fn_a is direct, fn_b is impacted → total 2 functions
        assert result.affected_function_count == 2
        assert result.affected_node_count == 2
