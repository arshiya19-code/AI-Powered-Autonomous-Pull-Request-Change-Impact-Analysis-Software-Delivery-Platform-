"""
Tests for Phase 4 — Custom Software Dependency Graph Engine.

Covers all 25 required test scenarios:
  1.  Node creation
  2.  Duplicate node handling
  3.  Edge creation
  4.  Duplicate edge handling
  5.  Incoming edges
  6.  Outgoing edges
  7.  Direct dependencies
  8.  Direct dependents
  9.  Transitive dependencies
  10. Transitive dependents
  11. Path finding
  12. Module nodes
  13. Class nodes
  14. Function nodes
  15. Method nodes
  16. Import edges
  17. Function call edges
  18. Inheritance edges
  19. Nested class/method relationships
  20. Multiple modules
  21. Circular dependencies
  22. Isolated nodes
  23. Missing / unresolved dependencies
  24. Confidence propagation
  25. Graph statistics

All tests use temporary directories and hand-crafted sources so they are
completely independent of the real project structure.
"""

import pytest
from pathlib import Path
from typing import Set

from analyzer.ast_parser import ASTParser, parse_source
from analyzer.dependency_extractor import DependencyExtractor, DependencyType
from analyzer.graph_engine import (
    CustomDependencyGraph,
    GraphBuilder,
    GraphEdge,
    GraphNode,
    EdgeType,
    NodeType,
    build_graph,
    build_graph_with_resolution,
)
from analyzer.resolver import (
    NameResolver,
    ResolutionStatus,
    ProjectResolutionResult,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def write_file(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def parse_file(tmp_path: Path, rel: str) -> "ParsedModule":
    parser = ASTParser()
    return parser.parse_file(tmp_path / rel)


def make_graph(tmp_path: Path, files: dict) -> CustomDependencyGraph:
    """Write files, parse all of them, build and return the graph."""
    for rel, content in files.items():
        write_file(tmp_path, rel, content)
    parser = ASTParser()
    parsed = [parser.parse_file(tmp_path / rel) for rel in files]
    return build_graph(tmp_path, parsed)


def make_node(nid: str, node_type: NodeType = NodeType.FUNCTION) -> GraphNode:
    return GraphNode(
        node_id=nid,
        node_type=node_type,
        name=nid.split(".")[-1],
        qualified_name=nid,
    )


def make_edge(src: str, tgt: str, etype: EdgeType = EdgeType.CALLS,
              confidence: float = 1.0) -> GraphEdge:
    return GraphEdge(source=src, target=tgt, edge_type=etype,
                     confidence=confidence)


# ===========================================================================
# 1. Node creation
# ===========================================================================

class TestNodeCreation:

    def test_add_node_returns_node(self):
        g = CustomDependencyGraph()
        node = make_node("pkg.func")
        returned = g.add_node(node)
        assert returned is g.nodes["pkg.func"]

    def test_node_stored_in_nodes_dict(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a.b"))
        assert "a.b" in g.nodes
        assert g.nodes["a.b"].qualified_name == "a.b"

    def test_adjacency_sets_created_on_add(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("n1"))
        assert "n1" in g.outgoing
        assert "n1" in g.incoming
        assert g.outgoing["n1"] == set()
        assert g.incoming["n1"] == set()

    def test_get_node_returns_none_for_missing(self):
        g = CustomDependencyGraph()
        assert g.get_node("nonexistent") is None

    def test_has_node(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("x"))
        assert g.has_node("x")
        assert not g.has_node("y")

    def test_all_node_types_accepted(self):
        g = CustomDependencyGraph()
        for nt in NodeType:
            g.add_node(GraphNode(
                node_id=f"node_{nt.value}",
                node_type=nt,
                name=nt.value,
                qualified_name=f"node_{nt.value}",
            ))
        assert len(g.nodes) == len(NodeType)


# ===========================================================================
# 2. Duplicate node handling
# ===========================================================================

class TestDuplicateNodeHandling:

    def test_add_duplicate_returns_existing(self):
        g = CustomDependencyGraph()
        n1 = make_node("dup")
        n2 = make_node("dup")
        r1 = g.add_node(n1)
        r2 = g.add_node(n2)
        assert r1 is r2  # same object
        assert len(g.nodes) == 1

    def test_duplicate_node_does_not_reset_adjacency(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b"))
        g.add_node(make_node("a"))  # duplicate
        # adjacency must survive
        assert "b" in g.outgoing["a"]


# ===========================================================================
# 3. Edge creation
# ===========================================================================

class TestEdgeCreation:

    def test_add_edge_requires_existing_nodes(self):
        g = CustomDependencyGraph()
        # No nodes yet — edge should be silently dropped
        e = g.add_edge(make_edge("ghost", "phantom"))
        assert len(g.edges) == 0

    def test_add_valid_edge(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b"))
        assert len(g.edges) == 1

    def test_edge_updates_adjacency(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b"))
        assert "b" in g.outgoing["a"]
        assert "a" in g.incoming["b"]

    def test_has_edge_with_type(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        assert g.has_edge("a", "b", EdgeType.CALLS)
        assert not g.has_edge("a", "b", EdgeType.INHERITS)

    def test_has_edge_without_type(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        assert g.has_edge("a", "b")

    def test_all_edge_types_accepted(self):
        g = CustomDependencyGraph()
        nodes = ["n0", "n1", "n2", "n3", "n4", "n5"]
        for n in nodes:
            g.add_node(make_node(n))
        for i, et in enumerate(EdgeType):
            g.add_edge(make_edge(f"n{i}", f"n{(i+1) % len(nodes)}", et))
        assert len(g.edges) == len(EdgeType)


# ===========================================================================
# 4. Duplicate edge handling
# ===========================================================================

class TestDuplicateEdgeHandling:

    def test_duplicate_edge_not_added_twice(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))  # duplicate
        assert len(g.edges) == 1

    def test_different_types_are_distinct_edges(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        g.add_edge(make_edge("a", "b", EdgeType.IMPORT))
        assert len(g.edges) == 2


# ===========================================================================
# 5 & 6. Incoming / outgoing edges
# ===========================================================================

class TestIncomingOutgoingEdges:

    def _setup(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        g.add_edge(make_edge("a", "c", EdgeType.IMPORT))
        g.add_edge(make_edge("b", "c", EdgeType.CALLS))
        return g

    def test_outgoing_edges(self):
        g = self._setup()
        out_a = g.get_outgoing_edges("a")
        targets = {e.target for e in out_a}
        assert targets == {"b", "c"}

    def test_incoming_edges(self):
        g = self._setup()
        inc_c = g.get_incoming_edges("c")
        sources = {e.source for e in inc_c}
        assert sources == {"a", "b"}

    def test_no_outgoing_for_leaf(self):
        g = self._setup()
        assert g.get_outgoing_edges("c") == []

    def test_no_incoming_for_root(self):
        g = self._setup()
        assert g.get_incoming_edges("a") == []


# ===========================================================================
# 7 & 8. Direct dependencies / dependents
# ===========================================================================

class TestDirectDependenciesDependents:

    def _setup(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("a", "c"))
        g.add_edge(make_edge("b", "d"))
        return g

    def test_direct_dependencies_of_a(self):
        g = self._setup()
        deps = {n.node_id for n in g.get_dependencies("a")}
        assert deps == {"b", "c"}

    def test_direct_dependents_of_d(self):
        g = self._setup()
        dents = {n.node_id for n in g.get_dependents("d")}
        assert dents == {"b"}

    def test_no_dependencies_for_leaf(self):
        g = self._setup()
        assert g.get_dependencies("d") == []

    def test_no_dependents_for_root(self):
        g = self._setup()
        assert g.get_dependents("a") == []

    def test_get_neighbors_includes_both_directions(self):
        g = self._setup()
        neighbors = {n.node_id for n in g.get_neighbors("b")}
        # b → d (out), a → b (in)
        assert "a" in neighbors
        assert "d" in neighbors


# ===========================================================================
# 9. Transitive dependencies
# ===========================================================================

class TestTransitiveDependencies:

    def test_transitive_deps_chain(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "d"))

        deps = g.get_transitive_dependencies("a")
        assert deps == {"b", "c", "d"}

    def test_transitive_deps_does_not_include_start(self):
        g = CustomDependencyGraph()
        for n in ["x", "y"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("x", "y"))
        deps = g.get_transitive_dependencies("x")
        assert "x" not in deps

    def test_transitive_deps_empty_for_isolated(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("alone"))
        assert g.get_transitive_dependencies("alone") == set()

    def test_transitive_deps_max_depth(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "d"))

        deps_1 = g.get_transitive_dependencies("a", max_depth=1)
        assert deps_1 == {"b"}

        deps_2 = g.get_transitive_dependencies("a", max_depth=2)
        assert deps_2 == {"b", "c"}

    def test_transitive_deps_edge_type_filter(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        g.add_edge(make_edge("b", "c", EdgeType.IMPORT))

        # Filter to only CALLS edges
        deps = g.get_transitive_dependencies("a", edge_types={EdgeType.CALLS})
        assert "b" in deps
        assert "c" not in deps  # not reachable via CALLS only


# ===========================================================================
# 10. Transitive dependents (reverse impact)
# ===========================================================================

class TestTransitiveDependents:

    def test_transitive_dependents_chain(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "d"))

        # If d changes, who is affected?
        affected = g.get_transitive_dependents("d")
        assert affected == {"a", "b", "c"}

    def test_transitive_dependents_empty_for_root(self):
        g = CustomDependencyGraph()
        for n in ["a", "b"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        assert g.get_transitive_dependents("a") == set()

    def test_transitive_dependents_fan_in(self):
        """Multiple sources depend on the same target."""
        g = CustomDependencyGraph()
        for n in ["core", "svc1", "svc2", "svc3"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("svc1", "core"))
        g.add_edge(make_edge("svc2", "core"))
        g.add_edge(make_edge("svc3", "core"))

        affected = g.get_transitive_dependents("core")
        assert affected == {"svc1", "svc2", "svc3"}

    def test_transitive_dependents_max_depth(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "d"))

        # d changes → only c at depth 1
        affected = g.get_transitive_dependents("d", max_depth=1)
        assert affected == {"c"}


# ===========================================================================
# 11. Path finding
# ===========================================================================

class TestPathFinding:

    def _chain_graph(self) -> CustomDependencyGraph:
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "d"))
        return g

    def test_path_found_in_chain(self):
        g = self._chain_graph()
        path = g.find_path("a", "d")
        assert path == ["a", "b", "c", "d"]

    def test_path_to_self(self):
        g = self._chain_graph()
        assert g.find_path("a", "a") == ["a"]

    def test_no_path_returns_none(self):
        g = self._chain_graph()
        # d→a doesn't exist (directed)
        assert g.find_path("d", "a") is None

    def test_path_missing_node_returns_none(self):
        g = self._chain_graph()
        assert g.find_path("a", "z") is None

    def test_find_all_paths(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        # Two paths: a→b→d and a→c→d
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("a", "c"))
        g.add_edge(make_edge("b", "d"))
        g.add_edge(make_edge("c", "d"))

        paths = g.find_all_paths("a", "d")
        assert len(paths) == 2
        path_sets = [frozenset(p) for p in paths]
        assert frozenset(["a", "b", "d"]) in path_sets
        assert frozenset(["a", "c", "d"]) in path_sets

    def test_path_edge_type_filter(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        g.add_edge(make_edge("b", "c", EdgeType.IMPORT))

        # Path exists without filter
        assert g.find_path("a", "c") == ["a", "b", "c"]
        # No path if only CALLS allowed (b→c is IMPORT)
        assert g.find_path("a", "c", edge_types={EdgeType.CALLS}) is None


# ===========================================================================
# 12–15. Node type tests (MODULE, CLASS, FUNCTION, METHOD)
# ===========================================================================

class TestNodeTypes:

    def test_module_nodes_from_build(self, tmp_path):
        files = {"app.py": "import os\n", "utils.py": "def helper(): pass\n"}
        g = make_graph(tmp_path, files)
        modules = g.get_module_nodes()
        module_names = {n.qualified_name for n in modules}
        assert "app" in module_names
        assert "utils" in module_names

    def test_class_nodes_from_build(self, tmp_path):
        files = {"models.py": "class User:\n    pass\nclass Admin(User):\n    pass\n"}
        g = make_graph(tmp_path, files)
        classes = g.get_class_nodes()
        names = {n.name for n in classes}
        assert "User" in names
        assert "Admin" in names

    def test_function_nodes_from_build(self, tmp_path):
        files = {"utils.py": "def helper(): pass\ndef process(): pass\n"}
        g = make_graph(tmp_path, files)
        funcs = g.get_function_nodes()
        names = {n.name for n in funcs}
        assert "helper" in names
        assert "process" in names

    def test_method_nodes_from_build(self, tmp_path):
        files = {
            "svc.py": (
                "class Service:\n"
                "    def execute(self): pass\n"
                "    def validate(self): pass\n"
            )
        }
        g = make_graph(tmp_path, files)
        methods = g.get_method_nodes()
        names = {n.name for n in methods}
        assert "execute" in names
        assert "validate" in names

    def test_project_node_exists(self, tmp_path):
        files = {"app.py": "x = 1\n"}
        g = make_graph(tmp_path, files)
        assert g.has_node("<project>")
        proj = g.get_node("<project>")
        assert proj.node_type == NodeType.PROJECT

    def test_nodes_by_type_filter(self):
        g = CustomDependencyGraph()
        g.add_node(GraphNode("mod", NodeType.MODULE, "mod", "mod"))
        g.add_node(GraphNode("cls", NodeType.CLASS, "cls", "cls"))
        g.add_node(GraphNode("fn", NodeType.FUNCTION, "fn", "fn"))

        assert len(g.nodes_by_type(NodeType.MODULE)) == 1
        assert len(g.nodes_by_type(NodeType.CLASS)) == 1
        assert len(g.nodes_by_type(NodeType.FUNCTION)) == 1


# ===========================================================================
# 16. Import edges
# ===========================================================================

class TestImportEdges:

    def test_import_edge_created(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py":   "import utils\n",
        }
        g = make_graph(tmp_path, files)
        import_edges = g.edges_by_type(EdgeType.IMPORT)
        targets = {e.target for e in import_edges}
        assert "utils" in targets

    def test_from_import_edge_created(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py":   "from utils import helper\n",
        }
        g = make_graph(tmp_path, files)
        fi_edges = g.edges_by_type(EdgeType.FROM_IMPORT)
        # The edge should point to utils or utils.helper
        targets = {e.target for e in fi_edges}
        assert any("utils" in t for t in targets)


# ===========================================================================
# 17. Function call edges
# ===========================================================================

class TestFunctionCallEdges:

    def test_call_edge_created_for_local_call(self, tmp_path):
        source = "def helper(): pass\ndef process():\n    helper()\n"
        files = {"app.py": source}
        g = make_graph(tmp_path, files)
        call_edges = g.edges_by_type(EdgeType.CALLS)
        targets = {e.target for e in call_edges}
        assert any("helper" in t for t in targets)

    def test_cross_module_call_edge(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py":   "from utils import helper\ndef run():\n    helper()\n",
        }
        g = make_graph(tmp_path, files)
        call_edges = g.edges_by_type(EdgeType.CALLS)
        targets = {e.target for e in call_edges}
        assert any("helper" in t for t in targets)


# ===========================================================================
# 18. Inheritance edges
# ===========================================================================

class TestInheritanceEdges:

    def test_inherits_edge_same_module(self, tmp_path):
        source = "class Base:\n    pass\nclass Child(Base):\n    pass\n"
        files = {"models.py": source}
        g = make_graph(tmp_path, files)
        inh_edges = g.edges_by_type(EdgeType.INHERITS)
        assert len(inh_edges) >= 1
        targets = {e.target for e in inh_edges}
        assert any("Base" in t for t in targets)

    def test_inherits_edge_cross_module(self, tmp_path):
        files = {
            "base.py":  "class Base:\n    pass\n",
            "child.py": "from base import Base\nclass Child(Base):\n    pass\n",
        }
        g = make_graph(tmp_path, files)
        inh_edges = g.edges_by_type(EdgeType.INHERITS)
        targets = {e.target for e in inh_edges}
        assert any("Base" in t for t in targets)


# ===========================================================================
# 19. Nested class/method relationships
# ===========================================================================

class TestNestedRelationships:

    def test_module_contains_class(self, tmp_path):
        files = {"svc.py": "class MyService:\n    def run(self): pass\n"}
        g = make_graph(tmp_path, files)
        contains_edges = g.edges_by_type(EdgeType.CONTAINS)
        targets = {e.target for e in contains_edges}
        assert any("MyService" in t for t in targets)

    def test_class_contains_method(self, tmp_path):
        files = {"svc.py": "class MyService:\n    def run(self): pass\n"}
        g = make_graph(tmp_path, files)
        contains_edges = g.edges_by_type(EdgeType.CONTAINS)
        # cls → method edge: source contains "MyService", target contains "run"
        cls_to_method = [
            e for e in contains_edges
            if "MyService" in e.source and "run" in e.target
        ]
        assert len(cls_to_method) >= 1

    def test_method_node_has_parent_class_in_id(self, tmp_path):
        files = {"svc.py": "class MyService:\n    def run(self): pass\n"}
        g = make_graph(tmp_path, files)
        method_nodes = g.get_method_nodes()
        assert any("MyService" in n.node_id and "run" in n.node_id
                   for n in method_nodes)

    def test_module_contains_function(self, tmp_path):
        files = {"utils.py": "def helper(): pass\n"}
        g = make_graph(tmp_path, files)
        contains_edges = g.edges_by_type(EdgeType.CONTAINS)
        fn_targets = [e.target for e in contains_edges if "helper" in e.target]
        assert fn_targets


# ===========================================================================
# 20. Multiple modules
# ===========================================================================

class TestMultipleModules:

    def test_all_modules_present(self, tmp_path):
        files = {
            "utils.py":    "def helper(): pass\n",
            "models.py":   "class User:\n    pass\n",
            "services.py": "from models import User\ndef create(): pass\n",
            "app.py":      "from services import create\ndef main():\n    create()\n",
        }
        g = make_graph(tmp_path, files)
        module_names = {n.qualified_name for n in g.get_module_nodes()}
        for mod in ("utils", "models", "services", "app"):
            assert mod in module_names

    def test_cross_module_dependency_tracked(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py":   "from utils import helper\ndef run():\n    helper()\n",
        }
        g = make_graph(tmp_path, files)
        # app depends on utils (transitively or directly)
        deps = g.get_transitive_dependencies("app")
        assert any("utils" in d for d in deps)

    def test_project_node_connects_all_modules(self, tmp_path):
        files = {
            "a.py": "x = 1\n",
            "b.py": "y = 2\n",
        }
        g = make_graph(tmp_path, files)
        project_deps = {n.node_id for n in g.get_dependencies("<project>")}
        assert "a" in project_deps
        assert "b" in project_deps


# ===========================================================================
# 21. Circular dependencies
# ===========================================================================

class TestCircularDependencies:

    def test_cycle_does_not_hang_transitive_deps(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "a"))  # cycle

        # Must terminate
        deps = g.get_transitive_dependencies("a")
        assert deps == {"b", "c"}

    def test_cycle_does_not_hang_transitive_dependents(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "a"))  # cycle

        affected = g.get_transitive_dependents("c")
        assert affected == {"a", "b"}

    def test_self_loop_handled(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("self_ref"))
        # Self-loop: source == target
        g.add_edge(make_edge("self_ref", "self_ref"))
        # Transitive should be empty (start excluded, and cycle visited)
        deps = g.get_transitive_dependencies("self_ref")
        assert "self_ref" not in deps

    def test_path_in_cyclic_graph(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "a"))

        path = g.find_path("a", "c")
        assert path is not None
        assert path[0] == "a" and path[-1] == "c"


# ===========================================================================
# 22. Isolated nodes
# ===========================================================================

class TestIsolatedNodes:

    def test_isolated_node_has_empty_adjacency(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("island"))
        assert g.outgoing["island"] == set()
        assert g.incoming["island"] == set()

    def test_isolated_node_has_no_dependencies(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("island"))
        assert g.get_dependencies("island") == []
        assert g.get_dependents("island") == []

    def test_isolated_node_counted_in_statistics(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("island"))
        stats = g.statistics()
        assert stats["isolated_nodes"] == 1

    def test_transitive_ops_on_isolated_node(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("island"))
        assert g.get_transitive_dependencies("island") == set()
        assert g.get_transitive_dependents("island") == set()
        assert g.find_path("island", "island") == ["island"]


# ===========================================================================
# 23. Missing / unresolved dependencies
# ===========================================================================

class TestMissingUnresolvedDependencies:

    def test_unresolved_symbol_call_no_crash(self, tmp_path):
        source = "def run():\n    completely_unknown_func()\n"
        files = {"app.py": source}
        # Should not raise
        g = make_graph(tmp_path, files)
        assert g is not None

    def test_external_module_not_added_as_node(self, tmp_path):
        files = {"app.py": "import os\nimport sys\n"}
        g = make_graph(tmp_path, files)
        # os / sys should NOT be nodes in our project graph
        assert not g.has_node("os")
        assert not g.has_node("sys")

    def test_dangling_edge_silently_dropped(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        # target "ghost" doesn't exist
        g.add_edge(make_edge("a", "ghost"))
        assert len(g.edges) == 0

    def test_syntax_error_module_no_crash(self, tmp_path):
        broken = "def broken(\n    pass\n"
        write_file(tmp_path, "broken.py", broken)
        parser = ASTParser()
        pm = parser.parse_source(broken, str(tmp_path / "broken.py"))
        g = build_graph(tmp_path, [pm])
        assert isinstance(g, CustomDependencyGraph)


# ===========================================================================
# 24. Confidence propagation
# ===========================================================================

class TestConfidencePropagation:

    def test_edge_confidence_stored(self):
        g = CustomDependencyGraph()
        g.add_node(make_node("a"))
        g.add_node(make_node("b"))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS, confidence=0.75))
        edge = g.edges[0]
        assert edge.confidence == 0.75

    def test_confidence_from_resolver_propagated(self, tmp_path):
        """Exact symbol match → confidence 1.0 on the graph edge."""
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py":   "from utils import helper\ndef run():\n    helper()\n",
        }
        g = make_graph(tmp_path, files)
        call_edges = [e for e in g.edges_by_type(EdgeType.CALLS)
                      if "helper" in e.target]
        if call_edges:
            assert call_edges[0].confidence == 1.0

    def test_all_edge_confidences_in_range(self, tmp_path):
        files = {
            "utils.py":    "def helper(): pass\n",
            "models.py":   "class User:\n    pass\n",
            "services.py": "from models import User\ndef create(): pass\n",
            "app.py":      "from services import create\ndef main():\n    create()\n",
        }
        g = make_graph(tmp_path, files)
        for edge in g.edges:
            assert 0.0 <= edge.confidence <= 1.0, (
                f"confidence out of range: {edge}"
            )


# ===========================================================================
# 25. Graph statistics
# ===========================================================================

class TestGraphStatistics:

    def test_statistics_keys_present(self):
        g = CustomDependencyGraph()
        stats = g.statistics()
        for key in ("total_nodes", "total_edges", "node_type_counts",
                    "edge_type_counts", "isolated_nodes",
                    "max_out_degree", "max_in_degree"):
            assert key in stats

    def test_statistics_empty_graph(self):
        g = CustomDependencyGraph()
        stats = g.statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_statistics_after_build(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py":   "from utils import helper\ndef run():\n    helper()\n",
        }
        g = make_graph(tmp_path, files)
        stats = g.statistics()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        # We have MODULE nodes at minimum
        assert stats["node_type_counts"].get("module", 0) >= 2

    def test_max_out_degree(self):
        g = CustomDependencyGraph()
        for n in ["hub", "a", "b", "c"]:
            g.add_node(make_node(n))
        for target in ["a", "b", "c"]:
            g.add_edge(make_edge("hub", target))
        stats = g.statistics()
        assert stats["max_out_degree"] == 3

    def test_max_in_degree(self):
        g = CustomDependencyGraph()
        for n in ["sink", "a", "b", "c"]:
            g.add_node(make_node(n))
        for src in ["a", "b", "c"]:
            g.add_edge(make_edge(src, "sink"))
        stats = g.statistics()
        assert stats["max_in_degree"] == 3


# ===========================================================================
# Additional: node removal
# ===========================================================================

class TestNodeRemoval:

    def test_remove_node_removes_edges(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.remove_node("b")
        assert not g.has_node("b")
        assert len(g.edges) == 0

    def test_remove_missing_node_returns_false(self):
        g = CustomDependencyGraph()
        assert g.remove_node("nonexistent") is False

    def test_remove_edge(self):
        g = CustomDependencyGraph()
        for n in ["a", "b"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b", EdgeType.CALLS))
        assert g.remove_edge("a", "b", EdgeType.CALLS)
        assert len(g.edges) == 0
        assert "b" not in g.outgoing["a"]


# ===========================================================================
# Additional: subgraph extraction
# ===========================================================================

class TestSubgraphExtraction:

    def test_subgraph_contains_only_requested_nodes(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "d"))

        sub = g.get_subgraph({"a", "b", "c"})
        assert "d" not in sub.nodes
        assert "a" in sub.nodes and "b" in sub.nodes and "c" in sub.nodes

    def test_subgraph_edges_only_within_subset(self):
        g = CustomDependencyGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(make_node(n))
        g.add_edge(make_edge("a", "b"))
        g.add_edge(make_edge("b", "c"))
        g.add_edge(make_edge("c", "d"))

        sub = g.get_subgraph({"a", "b", "c"})
        for e in sub.edges:
            assert e.source in sub.nodes and e.target in sub.nodes
        # c→d edge should not be in subgraph
        assert not sub.has_edge("c", "d")


# ===========================================================================
# Additional: repr / convenience functions
# ===========================================================================

class TestReprAndConvenience:

    def test_graph_repr(self):
        g = CustomDependencyGraph()
        r = repr(g)
        assert "CustomDependencyGraph" in r
        assert "nodes=0" in r

    def test_node_repr(self):
        n = make_node("pkg.func")
        r = repr(n)
        assert "GraphNode" in r

    def test_edge_repr(self):
        e = make_edge("a", "b", EdgeType.CALLS)
        r = repr(e)
        assert "calls" in r

    def test_build_graph_with_resolution_convenience(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py":   "from utils import helper\n",
        }
        for rel, content in files.items():
            write_file(tmp_path, rel, content)

        parser = ASTParser()
        parsed = [parser.parse_file(tmp_path / r) for r in files]

        resolver = NameResolver(tmp_path)
        resolution = resolver.resolve_project(parsed)

        g = build_graph_with_resolution(parsed, resolution, project_root=tmp_path)
        assert isinstance(g, CustomDependencyGraph)
        assert g.has_node("<project>")

    def test_nodes_in_file(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        parser = ASTParser()
        pm = parser.parse_file(tmp_path / "utils.py")
        g = build_graph(tmp_path, [pm])
        nodes = g.nodes_in_file(str(tmp_path / "utils.py"))
        # At least the module node and the function node
        assert len(nodes) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
