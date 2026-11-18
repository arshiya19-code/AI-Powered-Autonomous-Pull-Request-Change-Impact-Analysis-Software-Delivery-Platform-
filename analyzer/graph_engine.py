"""
Custom Software Dependency Graph Engine (Phase 4).

Builds and queries an in-memory dependency graph from the output of the
Phase 1-3 pipeline (AST Parser → Dependency Extractor → Name/Module Resolver).

Design constraints:
  - NO NetworkX, NO Neo4j, NO GAT, NO external graph libraries.
  - Pure Python: dict, set, list, dataclass.
  - Bidirectional adjacency: outgoing AND incoming edges maintained at all
    times so that both forward-dependency queries and reverse-impact queries
    are O(degree) after construction.

High-level architecture:
  ┌─────────────────────────────────────────────┐
  │            CustomDependencyGraph             │
  │  nodes:    Dict[node_id → GraphNode]         │
  │  edges:    List[GraphEdge]                   │
  │  outgoing: Dict[node_id → Set[node_id]]      │
  │  incoming: Dict[node_id → Set[node_id]]      │
  │  _edge_index: Dict[(src,tgt,type) → edge]    │
  └───────────────┬─────────────────────────────┘
                  │ populated by
  ┌───────────────▼─────────────────────────────┐
  │              GraphBuilder                    │
  │  Consumes: ParsedModule + ModuleDependencies │
  │            + ProjectResolutionResult         │
  │  Produces: CustomDependencyGraph             │
  └─────────────────────────────────────────────┘

Node types:
  PROJECT   – synthetic root representing the whole project
  MODULE    – a Python .py file / package
  CLASS     – a class definition
  FUNCTION  – a standalone function
  METHOD    – a method inside a class

Edge types:
  CONTAINS    – structural containment  (MODULE→CLASS, CLASS→METHOD, …)
  IMPORT      – ``import module``
  FROM_IMPORT – ``from module import name``
  CALLS       – function/method call
  INHERITS    – class inheritance
  DECORATES   – decorator usage

Graph algorithms implemented (all custom, no external libraries):
  get_dependencies()         – direct outgoing neighbours
  get_dependents()           – direct incoming neighbours
  get_outgoing_edges()       – all edges leaving a node
  get_incoming_edges()       – all edges entering a node
  find_path()                – BFS path between two nodes
  get_transitive_dependencies()   – DFS/BFS reachable set (forward)
  get_transitive_dependents()     – DFS/BFS reachable set (reverse)
  nodes_by_type()            – filter nodes by NodeType
  edges_by_type()            – filter edges by EdgeType
  get_subgraph()             – induced subgraph over a node subset
  statistics()               – summary dict

Limitations:
  - Only Python projects (AST parser is Python-only).
  - Dynamic imports are not represented.
  - Wildcard imports are not expanded.
  - Cycles are handled safely (visited-set guard in all traversals).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple
)

from analyzer.ast_parser import ParsedModule
from analyzer.dependency_extractor import (
    DependencyType,
    ModuleDependencies,
    extract_dependencies_multiple,
)
from analyzer.resolver import (
    ProjectResolutionResult,
    ResolutionResult,
    ResolutionStatus,
    ResolvedDependency,
    NameResolver,
)
from analyzer.languages.base import (
    Dependency as IRDependency,
    LanguageDependencyType,
    ParsedSource,
    Symbol,
    SymbolKind,
)


# ============================================================================
# Enums
# ============================================================================

class NodeType(Enum):
    """Types of nodes in the dependency graph."""
    PROJECT  = "project"
    MODULE   = "module"
    CLASS    = "class"
    FUNCTION = "function"
    METHOD   = "method"


class EdgeType(Enum):
    """Types of directed edges in the dependency graph."""
    CONTAINS    = "contains"     # structural: MODULE→CLASS, CLASS→METHOD …
    IMPORT      = "import"       # import module
    FROM_IMPORT = "from_import"  # from module import name
    CALLS       = "calls"        # function/method call
    INHERITS    = "inherits"     # class inheritance
    DECORATES   = "decorates"    # decorator usage


# ============================================================================
# Core data structures
# ============================================================================

@dataclass
class GraphNode:
    """
    A vertex in the dependency graph.

    ``node_id`` is a globally unique string identifier, typically the
    fully-qualified Python name (e.g. ``"services.payment.process_payment"``).
    For the synthetic project root the id is ``"<project>"``.
    """
    node_id:        str
    node_type:      NodeType
    name:           str             # Simple name (e.g. "process_payment")
    qualified_name: str             # Dotted qualified name
    file_path:      Optional[str] = None
    start_line:     int = 0
    end_line:       int = 0
    metadata:       Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __repr__(self) -> str:
        return f"GraphNode({self.node_type.value}:{self.node_id!r})"


@dataclass
class GraphEdge:
    """
    A directed edge in the dependency graph.

    ``source`` and ``target`` are ``node_id`` strings.
    ``confidence`` is inherited from the resolver (0.0 – 1.0).
    """
    source:     str
    target:     str
    edge_type:  EdgeType
    confidence: float = 1.0
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"GraphEdge({self.source!r} "
            f"--[{self.edge_type.value}]--> "
            f"{self.target!r}  conf={self.confidence:.2f})"
        )


# ============================================================================
# Custom Dependency Graph
# ============================================================================

class CustomDependencyGraph:
    """
    In-memory, bidirectional directed dependency graph.

    Storage
    -------
    nodes     : Dict[node_id, GraphNode]
    edges     : List[GraphEdge]
    outgoing  : Dict[node_id, Set[node_id]]   – forward adjacency
    incoming  : Dict[node_id, Set[node_id]]   – reverse adjacency
    _edge_idx : Dict[(src, tgt, EdgeType), GraphEdge] – dedup index

    All mutating operations keep outgoing/incoming/edge_idx in sync.
    """

    def __init__(self) -> None:
        self.nodes:     Dict[str, GraphNode]  = {}
        self.edges:     List[GraphEdge]       = []
        self.outgoing:  Dict[str, Set[str]]   = {}
        self.incoming:  Dict[str, Set[str]]   = {}
        self._edge_idx: Dict[Tuple[str, str, EdgeType], GraphEdge] = {}

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> GraphNode:
        """
        Add a node to the graph.

        If a node with the same ``node_id`` already exists the existing node
        is returned unchanged (idempotent).
        """
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            self.outgoing[node.node_id] = set()
            self.incoming[node.node_id] = set()
        return self.nodes[node.node_id]

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """
        Add a directed edge between two existing nodes.

        If either endpoint does not exist in the graph it is silently ignored
        (dangling-reference safety). Duplicate edges (same src, tgt, type)
        are deduplicated; the existing edge is returned.
        """
        key = (edge.source, edge.target, edge.edge_type)
        if key in self._edge_idx:
            return self._edge_idx[key]

        # Only add if both endpoints exist
        if edge.source not in self.nodes or edge.target not in self.nodes:
            return edge  # silently drop dangling edges

        self.edges.append(edge)
        self._edge_idx[key] = edge
        self.outgoing[edge.source].add(edge.target)
        self.incoming[edge.target].add(edge.source)
        return edge

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its incident edges from the graph.

        Returns True if the node existed, False otherwise.
        """
        if node_id not in self.nodes:
            return False

        # Remove all edges touching this node
        incident = list(self.get_outgoing_edges(node_id)) + \
                   list(self.get_incoming_edges(node_id))
        for edge in incident:
            self.remove_edge(edge.source, edge.target, edge.edge_type)

        del self.nodes[node_id]
        self.outgoing.pop(node_id, None)
        self.incoming.pop(node_id, None)
        return True

    def remove_edge(self, source: str, target: str, edge_type: EdgeType) -> bool:
        """
        Remove a specific directed edge.

        Returns True if the edge existed, False otherwise.
        """
        key = (source, target, edge_type)
        if key not in self._edge_idx:
            return False

        edge = self._edge_idx.pop(key)
        self.edges.remove(edge)
        self.outgoing[source].discard(target)

        # Only remove from incoming if no other edge of different type exists
        other_edges = [e for e in self.edges
                       if e.source == source and e.target == target]
        if not other_edges:
            self.incoming[target].discard(source)

        return True

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Return node by id, or None if not present."""
        return self.nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def has_edge(self, source: str, target: str,
                 edge_type: Optional[EdgeType] = None) -> bool:
        if edge_type is not None:
            return (source, target, edge_type) in self._edge_idx
        return any(
            (source, target, et) in self._edge_idx
            for et in EdgeType
        )

    def get_outgoing_edges(self, node_id: str) -> List[GraphEdge]:
        """Return all edges leaving ``node_id``."""
        return [e for e in self.edges if e.source == node_id]

    def get_incoming_edges(self, node_id: str) -> List[GraphEdge]:
        """Return all edges entering ``node_id``."""
        return [e for e in self.edges if e.target == node_id]

    def get_dependencies(self, node_id: str) -> List[GraphNode]:
        """
        Direct forward dependencies of ``node_id``.

        Returns all nodes that ``node_id`` depends on (out-neighbours).
        """
        return [
            self.nodes[nid]
            for nid in self.outgoing.get(node_id, set())
            if nid in self.nodes
        ]

    def get_dependents(self, node_id: str) -> List[GraphNode]:
        """
        Direct reverse dependents of ``node_id``.

        Returns all nodes that depend on ``node_id`` (in-neighbours).
        Critical for change-impact analysis.
        """
        return [
            self.nodes[nid]
            for nid in self.incoming.get(node_id, set())
            if nid in self.nodes
        ]

    def get_neighbors(self, node_id: str) -> List[GraphNode]:
        """All adjacent nodes (both directions), deduplicated."""
        seen: Set[str] = set()
        result: List[GraphNode] = []
        for nid in (self.outgoing.get(node_id, set()) |
                    self.incoming.get(node_id, set())):
            if nid not in seen and nid in self.nodes:
                seen.add(nid)
                result.append(self.nodes[nid])
        return result

    # ------------------------------------------------------------------
    # Graph traversal algorithms (all custom)
    # ------------------------------------------------------------------

    def get_transitive_dependencies(
            self,
            node_id: str,
            edge_types: Optional[Set[EdgeType]] = None,
            max_depth: Optional[int] = None,
    ) -> Set[str]:
        """
        BFS forward traversal: all nodes reachable from ``node_id``
        following outgoing edges.

        Args:
            node_id:    Starting node.
            edge_types: If specified, only traverse edges of these types.
            max_depth:  Maximum hop count (None = unlimited).

        Returns:
            Set of node_ids reachable (excluding ``node_id`` itself).
        """
        return self._bfs(node_id, direction="outgoing",
                         edge_types=edge_types, max_depth=max_depth)

    def get_transitive_dependents(
            self,
            node_id: str,
            edge_types: Optional[Set[EdgeType]] = None,
            max_depth: Optional[int] = None,
    ) -> Set[str]:
        """
        BFS reverse traversal: all nodes from which ``node_id`` is reachable.

        This is the **change-impact propagation** query:
        "If ``node_id`` changes, which other nodes are transitively affected?"

        Args:
            node_id:    The changed node.
            edge_types: Restrict traversal to specific edge types.
            max_depth:  Maximum hop count.

        Returns:
            Set of node_ids that transitively depend on ``node_id``.
        """
        return self._bfs(node_id, direction="incoming",
                         edge_types=edge_types, max_depth=max_depth)

    def find_path(
            self,
            source: str,
            target: str,
            edge_types: Optional[Set[EdgeType]] = None,
    ) -> Optional[List[str]]:
        """
        BFS shortest path from ``source`` to ``target``.

        Returns the list of node_ids forming the path (inclusive of both
        endpoints), or None if no path exists.

        Args:
            source:     Starting node_id.
            target:     Goal node_id.
            edge_types: If specified, only traverse edges of these types.
        """
        if source not in self.nodes or target not in self.nodes:
            return None
        if source == target:
            return [source]

        # BFS with predecessor tracking
        visited: Set[str] = {source}
        queue: deque[str] = deque([source])
        predecessor: Dict[str, Optional[str]] = {source: None}

        while queue:
            current = queue.popleft()
            for nid in self._neighbours_forward(current, edge_types):
                if nid not in visited:
                    visited.add(nid)
                    predecessor[nid] = current
                    if nid == target:
                        return self._reconstruct_path(predecessor, target)
                    queue.append(nid)

        return None  # no path

    def find_all_paths(
            self,
            source: str,
            target: str,
            edge_types: Optional[Set[EdgeType]] = None,
            max_paths: int = 10,
    ) -> List[List[str]]:
        """
        DFS enumeration of all simple paths from ``source`` to ``target``.

        Capped at ``max_paths`` to avoid combinatorial explosion on dense
        graphs. Each path is a list of node_ids.
        """
        if source not in self.nodes or target not in self.nodes:
            return []

        results: List[List[str]] = []
        stack: List[Tuple[str, List[str]]] = [(source, [source])]

        while stack and len(results) < max_paths:
            current, path = stack.pop()
            if current == target and len(path) > 1:
                results.append(path)
                continue
            path_set = set(path)
            for nid in self._neighbours_forward(current, edge_types):
                if nid not in path_set:
                    stack.append((nid, path + [nid]))

        return results

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Return all nodes of a given type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def edges_by_type(self, edge_type: EdgeType) -> List[GraphEdge]:
        """Return all edges of a given type."""
        return [e for e in self.edges if e.edge_type == edge_type]

    def nodes_in_file(self, file_path: str) -> List[GraphNode]:
        """Return all nodes whose source file matches ``file_path``."""
        return [n for n in self.nodes.values() if n.file_path == file_path]

    def get_module_nodes(self) -> List[GraphNode]:
        """Convenience wrapper."""
        return self.nodes_by_type(NodeType.MODULE)

    def get_function_nodes(self) -> List[GraphNode]:
        return self.nodes_by_type(NodeType.FUNCTION)

    def get_class_nodes(self) -> List[GraphNode]:
        return self.nodes_by_type(NodeType.CLASS)

    def get_method_nodes(self) -> List[GraphNode]:
        return self.nodes_by_type(NodeType.METHOD)

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def get_subgraph(self, node_ids: Set[str]) -> "CustomDependencyGraph":
        """
        Return a new graph containing only the nodes in ``node_ids``
        and the edges whose both endpoints are in ``node_ids``.
        """
        sub = CustomDependencyGraph()
        for nid in node_ids:
            if nid in self.nodes:
                sub.add_node(self.nodes[nid])
        for edge in self.edges:
            if edge.source in node_ids and edge.target in node_ids:
                sub.add_edge(edge)
        return sub

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """Return a summary dict of graph properties."""
        type_counts: Dict[str, int] = {}
        for n in self.nodes.values():
            type_counts[n.node_type.value] = \
                type_counts.get(n.node_type.value, 0) + 1

        edge_counts: Dict[str, int] = {}
        for e in self.edges:
            edge_counts[e.edge_type.value] = \
                edge_counts.get(e.edge_type.value, 0) + 1

        # Identify nodes with no edges at all (truly isolated)
        isolated = [
            nid for nid in self.nodes
            if not self.outgoing.get(nid) and not self.incoming.get(nid)
        ]

        # Max out-degree
        max_out = max(
            (len(s) for s in self.outgoing.values()), default=0
        )
        max_in = max(
            (len(s) for s in self.incoming.values()), default=0
        )

        return {
            "total_nodes":     len(self.nodes),
            "total_edges":     len(self.edges),
            "node_type_counts": type_counts,
            "edge_type_counts": edge_counts,
            "isolated_nodes":  len(isolated),
            "max_out_degree":  max_out,
            "max_in_degree":   max_in,
        }

    def __repr__(self) -> str:
        return (
            f"CustomDependencyGraph("
            f"nodes={len(self.nodes)}, edges={len(self.edges)})"
        )

    # ------------------------------------------------------------------
    # Internal traversal helpers
    # ------------------------------------------------------------------

    def _bfs(
            self,
            start: str,
            direction: str,              # "outgoing" | "incoming"
            edge_types: Optional[Set[EdgeType]],
            max_depth: Optional[int],
    ) -> Set[str]:
        """Generic BFS used by transitive_dependencies / transitive_dependents."""
        if start not in self.nodes:
            return set()

        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque([(start, 0)])
        visited.add(start)

        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            if direction == "outgoing":
                neighbours = self._neighbours_forward(current, edge_types)
            else:
                neighbours = self._neighbours_reverse(current, edge_types)
            for nid in neighbours:
                if nid not in visited:
                    visited.add(nid)
                    queue.append((nid, depth + 1))

        visited.discard(start)
        return visited

    def _neighbours_forward(
            self, node_id: str,
            edge_types: Optional[Set[EdgeType]]
    ) -> List[str]:
        """Outgoing neighbours, optionally filtered by edge type."""
        if edge_types is None:
            return list(self.outgoing.get(node_id, set()))
        return [
            e.target for e in self.edges
            if e.source == node_id and e.edge_type in edge_types
        ]

    def _neighbours_reverse(
            self, node_id: str,
            edge_types: Optional[Set[EdgeType]]
    ) -> List[str]:
        """Incoming neighbours, optionally filtered by edge type."""
        if edge_types is None:
            return list(self.incoming.get(node_id, set()))
        return [
            e.source for e in self.edges
            if e.target == node_id and e.edge_type in edge_types
        ]

    @staticmethod
    def _reconstruct_path(
            predecessor: Dict[str, Optional[str]], target: str
    ) -> List[str]:
        path: List[str] = []
        current: Optional[str] = target
        while current is not None:
            path.append(current)
            current = predecessor[current]
        path.reverse()
        return path


# ============================================================================
# Graph Builder
# ============================================================================

class GraphBuilder:
    """
    Constructs a CustomDependencyGraph from Phase 1-3 pipeline output.

    Accepts:
        parsed_modules   – list of ParsedModule  (Phase 1)
        resolution       – ProjectResolutionResult (Phase 3)

    Algorithm:
        1. Add a synthetic PROJECT root node.
        2. For each parsed module:
           a. Add MODULE node.
           b. Add PROJECT → MODULE  CONTAINS edge.
           c. Add CLASS / FUNCTION / METHOD nodes.
           d. Add MODULE→CLASS, MODULE→FUNCTION, CLASS→METHOD CONTAINS edges.
        3. For each resolved dependency (from Phase 3):
           a. Map DependencyType to EdgeType.
           b. Look up source & target node_ids.
           c. Add edge if both nodes exist.
    """

    # Map resolver DependencyType → graph EdgeType
    _DEP_TO_EDGE: Dict[DependencyType, EdgeType] = {
        DependencyType.IMPORT:       EdgeType.IMPORT,
        DependencyType.FROM_IMPORT:  EdgeType.FROM_IMPORT,
        DependencyType.FUNCTION_CALL: EdgeType.CALLS,
        DependencyType.METHOD_CALL:  EdgeType.CALLS,
        DependencyType.INHERITANCE:  EdgeType.INHERITS,
        DependencyType.DECORATOR:    EdgeType.DECORATES,
        DependencyType.MODULE_REFERENCE: EdgeType.IMPORT,
    }

    # Map IR LanguageDependencyType → graph EdgeType (for IR-first path)
    _IR_DEP_TO_EDGE: Dict[LanguageDependencyType, EdgeType] = {
        LanguageDependencyType.IMPORTS:     EdgeType.IMPORT,
        LanguageDependencyType.CALLS:       EdgeType.CALLS,
        LanguageDependencyType.INHERITS:    EdgeType.INHERITS,
        LanguageDependencyType.ANNOTATES:   EdgeType.DECORATES,
        LanguageDependencyType.REFERENCES:  EdgeType.IMPORT,
    }

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root
        self._graph = CustomDependencyGraph()
        # Maps file_path → dotted module name (populated during build)
        self._file_to_module: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
            self,
            parsed_modules: List[ParsedModule],
            resolution: Optional[ProjectResolutionResult] = None,
    ) -> CustomDependencyGraph:
        """
        Build and return the dependency graph.

        Args:
            parsed_modules: All parsed modules from Phase 1.
            resolution:     Full project resolution result from Phase 3.
                            If None, only structural (CONTAINS) edges are built.

        Returns:
            Populated CustomDependencyGraph.
        """
        self._graph = CustomDependencyGraph()

        # Build file → module mapping from resolution if available
        if resolution:
            for mod_name, res_result in resolution.module_results.items():
                self._file_to_module[res_result.file_path] = mod_name

        # Derive module name for any file not covered by resolution
        for pm in parsed_modules:
            if pm.file_path not in self._file_to_module:
                self._file_to_module[pm.file_path] = pm.module_name

        # 1. Add synthetic project root
        project_node = self._graph.add_node(GraphNode(
            node_id="<project>",
            node_type=NodeType.PROJECT,
            name="<project>",
            qualified_name="<project>",
            metadata={"project_root": str(self.project_root or "")},
        ))

        # 2. Build structural nodes + CONTAINS edges from parsed modules
        for pm in parsed_modules:
            self._add_module_structure(pm)

        # 3. Build dependency edges from resolver output
        if resolution:
            self._add_resolved_edges(resolution)

        return self._graph

    def build_from_files(
            self,
            project_root: Path,
            parsed_modules: List[ParsedModule],
            exclude_dirs: Optional[List[str]] = None,
    ) -> CustomDependencyGraph:
        """
        Convenience method: runs Phase 3 internally then builds the graph.

        Useful for tests and simple pipelines that don't need to inspect
        intermediate resolver output.
        """
        self.project_root = project_root
        resolver = NameResolver(project_root, exclude_dirs)
        resolution = resolver.resolve_project(parsed_modules)
        return self.build(parsed_modules, resolution)

    # ------------------------------------------------------------------
    # IR-first build method (Phase 7B)
    # ------------------------------------------------------------------

    def build_from_ir(
            self,
            parsed_sources: List[ParsedSource],
    ) -> CustomDependencyGraph:
        """
        Build a CustomDependencyGraph from language-neutral ParsedSource objects.

        This is the IR-first entry point that replaces the Python-specific
        `build(parsed_modules, resolution)` path.

        Args:
            parsed_sources: List of ParsedSource objects (one per file)
                            produced by a LanguageAdapter.

        Returns:
            Populated CustomDependencyGraph.

        This method:
            1. Adds a synthetic PROJECT root node.
            2. For each ParsedSource, adds structural nodes (MODULE, CLASS,
               FUNCTION, METHOD) and CONTAINS edges based on Symbol data.
            3. Adds dependency edges from ParsedSource.dependencies using
               the IR dependency type mapping.

        Note: This path does NOT require a Phase 3 ProjectResolutionResult.
        The ParsedSource objects are expected to have already-resolved
        dependency targets (qualified names) via the LanguageAdapter's
        resolve_symbols() step.
        """
        self._graph = CustomDependencyGraph()

        # Build file → module mapping from ParsedSource metadata
        file_to_module: Dict[str, str] = {}
        for ps in parsed_sources:
            if ps.language == "python":
                module_name = ps.metadata.get("module_name", ps.file_path)
            else:
                # For non-Python, derive from file path
                from pathlib import Path
                p = Path(ps.file_path)
                module_name = p.stem if p.suffix else ps.file_path
            file_to_module[ps.file_path] = module_name

        # 1. Add synthetic project root
        self._graph.add_node(GraphNode(
            node_id="<project>",
            node_type=NodeType.PROJECT,
            name="<project>",
            qualified_name="<project>",
            metadata={"project_root": str(self.project_root or "")},
        ))

        # 2. Build structural nodes + CONTAINS edges from ParsedSource symbols
        for ps in parsed_sources:
            self._add_module_structure_from_ir(ps, file_to_module)

        # 3. Build dependency edges from IR dependencies
        for ps in parsed_sources:
            self._add_ir_dependencies(ps, file_to_module)

        return self._graph

    # ------------------------------------------------------------------
    # Internal: structural nodes from IR
    # ------------------------------------------------------------------

    def _add_module_structure_from_ir(
            self,
            ps: ParsedSource,
            file_to_module: Dict[str, str],
    ) -> None:
        """Create MODULE, CLASS, FUNCTION, METHOD nodes and CONTAINS edges from IR symbols."""
        module_name = file_to_module.get(ps.file_path, ps.metadata.get("module_name", ps.file_path))
        module_id = module_name

        # MODULE node
        self._graph.add_node(GraphNode(
            node_id=module_id,
            node_type=NodeType.MODULE,
            name=module_name.split(".")[-1],
            qualified_name=module_name,
            file_path=ps.file_path,
            start_line=1,
            end_line=ps.metadata.get("source_lines", 0),
            metadata={"has_syntax_errors": ps.has_errors},
        ))

        # PROJECT → MODULE
        self._graph.add_edge(GraphEdge(
            source="<project>",
            target=module_id,
            edge_type=EdgeType.CONTAINS,
        ))

        # Group symbols by kind
        classes = [s for s in ps.symbols if s.symbol_kind == SymbolKind.CLASS]
        functions = [s for s in ps.symbols if s.symbol_kind == SymbolKind.FUNCTION]
        methods = [s for s in ps.symbols if s.symbol_kind == SymbolKind.METHOD]

        # Classes
        for cls in classes:
            cls_id = cls.qualified_name
            self._graph.add_node(GraphNode(
                node_id=cls_id,
                node_type=NodeType.CLASS,
                name=cls.name,
                qualified_name=cls_id,
                file_path=ps.file_path,
                start_line=cls.start_line,
                end_line=cls.end_line,
                metadata=dict(cls.metadata),
            ))
            # MODULE → CLASS
            self._graph.add_edge(GraphEdge(
                source=module_id,
                target=cls_id,
                edge_type=EdgeType.CONTAINS,
            ))

        # Standalone functions
        for func in functions:
            func_id = func.qualified_name
            self._graph.add_node(GraphNode(
                node_id=func_id,
                node_type=NodeType.FUNCTION,
                name=func.name,
                qualified_name=func_id,
                file_path=ps.file_path,
                start_line=func.start_line,
                end_line=func.end_line,
                metadata=dict(func.metadata),
            ))
            # MODULE → FUNCTION
            self._graph.add_edge(GraphEdge(
                source=module_id,
                target=func_id,
                edge_type=EdgeType.CONTAINS,
            ))

        # Methods (need to link to parent class)
        for method in methods:
            method_id = method.qualified_name
            parent_class = method.parent
            self._graph.add_node(GraphNode(
                node_id=method_id,
                node_type=NodeType.METHOD,
                name=method.name,
                qualified_name=method_id,
                file_path=ps.file_path,
                start_line=method.start_line,
                end_line=method.end_line,
                metadata=dict(method.metadata),
            ))
            # CLASS → METHOD (if parent exists)
            if parent_class and parent_class in self._graph.nodes:
                self._graph.add_edge(GraphEdge(
                    source=parent_class,
                    target=method_id,
                    edge_type=EdgeType.CONTAINS,
                ))
            elif parent_class:
                # Parent class might be in a different ParsedSource - check by qualified_name
                for cls in classes:
                    if cls.qualified_name == parent_class:
                        self._graph.add_edge(GraphEdge(
                            source=cls.qualified_name,
                            target=method_id,
                            edge_type=EdgeType.CONTAINS,
                        ))
                        break

    def _add_ir_dependencies(
            self,
            ps: ParsedSource,
            file_to_module: Dict[str, str],
    ) -> None:
        """Add dependency edges from IR Dependency objects."""
        module_name = file_to_module.get(ps.file_path, ps.file_path)
        for dep in ps.dependencies:
            edge_type = self._IR_DEP_TO_EDGE.get(dep.dep_type)
            if dep.dep_type == LanguageDependencyType.IMPORTS:
                # ``import_form`` is optional adapter-provided semantic IR.
                # It is intentionally language-neutral: adapters can express
                # whether an import names a module or an exported symbol.
                if dep.metadata.get("import_form") == "named":
                    edge_type = EdgeType.FROM_IMPORT
            if edge_type is None:
                continue

            source_id = self._ir_graph_node_id(dep.source, module_name)
            target_id = self._ir_graph_node_id(dep.target, module_name)

            # Only add edge if both endpoints exist in the graph
            if source_id in self._graph.nodes and target_id in self._graph.nodes:
                self._graph.add_edge(GraphEdge(
                    source=source_id,
                    target=target_id,
                    edge_type=edge_type,
                    confidence=dep.confidence,
                    metadata={
                        "dep_type": dep.dep_type.value,
                        "source_file": dep.source_file,
                        "source_line": dep.source_line,
                        "language": dep.language,
                    },
                ))

    def _ir_graph_node_id(self, candidate: str, module_name: str) -> str:
        """Resolve an IR dependency endpoint to an existing graph node.

        Adapters may report a local symbol as either ``name`` or its fully
        qualified form. Normalising that presentation here retains graph
        semantics without introducing Python-specific core types.
        """
        if candidate in self._graph.nodes:
            return candidate
        qualified = f"{module_name}.{candidate}"
        if qualified in self._graph.nodes:
            return qualified
        return candidate

    # ------------------------------------------------------------------
    # Internal: structural nodes (legacy)
    # ------------------------------------------------------------------

    def _add_module_structure(self, pm: ParsedModule) -> None:
        """Create MODULE, CLASS, FUNCTION, METHOD nodes and CONTAINS edges."""
        module_name = self._file_to_module.get(pm.file_path, pm.module_name)
        module_id   = module_name

        # MODULE node
        mod_node = self._graph.add_node(GraphNode(
            node_id=module_id,
            node_type=NodeType.MODULE,
            name=module_name.split(".")[-1],
            qualified_name=module_name,
            file_path=pm.file_path,
            start_line=1,
            end_line=pm.source_lines,
            metadata={"has_syntax_errors": bool(pm.syntax_errors)},
        ))

        # PROJECT → MODULE
        self._graph.add_edge(GraphEdge(
            source="<project>",
            target=module_id,
            edge_type=EdgeType.CONTAINS,
        ))

        # Classes
        for cls in pm.classes:
            cls_id = f"{module_name}.{cls.name}"
            self._graph.add_node(GraphNode(
                node_id=cls_id,
                node_type=NodeType.CLASS,
                name=cls.name,
                qualified_name=cls_id,
                file_path=pm.file_path,
                start_line=cls.location.start_line if cls.location else 0,
                end_line=cls.location.end_line if cls.location else 0,
                metadata={"bases": cls.bases, "decorators": cls.decorators},
            ))
            # MODULE → CLASS
            self._graph.add_edge(GraphEdge(
                source=module_id,
                target=cls_id,
                edge_type=EdgeType.CONTAINS,
            ))

            # Methods
            for method in cls.methods:
                method_id = f"{module_name}.{cls.name}.{method.name}"
                self._graph.add_node(GraphNode(
                    node_id=method_id,
                    node_type=NodeType.METHOD,
                    name=method.name,
                    qualified_name=method_id,
                    file_path=pm.file_path,
                    start_line=method.location.start_line if method.location else 0,
                    end_line=method.location.end_line if method.location else 0,
                    metadata={
                        "is_async": method.is_async,
                        "is_static": method.is_static,
                        "is_classmethod": method.is_classmethod,
                        "args": method.args,
                        "returns": method.returns,
                    },
                ))
                # CLASS → METHOD
                self._graph.add_edge(GraphEdge(
                    source=cls_id,
                    target=method_id,
                    edge_type=EdgeType.CONTAINS,
                ))

        # Standalone functions
        for func in pm.functions:
            func_id = f"{module_name}.{func.name}"
            self._graph.add_node(GraphNode(
                node_id=func_id,
                node_type=NodeType.FUNCTION,
                name=func.name,
                qualified_name=func_id,
                file_path=pm.file_path,
                start_line=func.location.start_line if func.location else 0,
                end_line=func.location.end_line if func.location else 0,
                metadata={
                    "is_async": func.is_async,
                    "args": func.args,
                    "returns": func.returns,
                    "decorators": func.decorators,
                },
            ))
            # MODULE → FUNCTION
            self._graph.add_edge(GraphEdge(
                source=module_id,
                target=func_id,
                edge_type=EdgeType.CONTAINS,
            ))

    # ------------------------------------------------------------------
    # Internal: dependency edges from resolver
    # ------------------------------------------------------------------

    def _add_resolved_edges(self, resolution: ProjectResolutionResult) -> None:
        """
        Walk all ResolvedDependency objects and create graph edges.

        Only RESOLVED internal dependencies produce edges.
        EXTERNAL dependencies are skipped (they're not nodes in our graph).
        """
        for mod_name, res_result in resolution.module_results.items():
            for rd in res_result.resolved_dependencies:
                if rd.status != ResolutionStatus.RESOLVED:
                    continue  # skip external / unresolved

                edge_type = self._DEP_TO_EDGE.get(rd.dep_type)
                if edge_type is None:
                    continue

                source_id = self._resolve_source_id(rd, mod_name)
                target_id = self._resolve_target_id(rd)

                if source_id and target_id and source_id != target_id:
                    self._graph.add_edge(GraphEdge(
                        source=source_id,
                        target=target_id,
                        edge_type=edge_type,
                        confidence=rd.confidence,
                        metadata={
                            "dep_type": rd.dep_type.value,
                            "original_target": rd.original_target,
                            "source_line": rd.source_line,
                        },
                    ))

    def _resolve_source_id(
            self, rd: ResolvedDependency, module_name: str
    ) -> Optional[str]:
        """
        Determine the source node_id for a resolved dependency.

        The source_symbol on the ResolvedDependency is the
        function/class/method *name within the module*, not a qualified id.
        We try to find the matching graph node.
        """
        src_sym = rd.source_symbol  # e.g. "process_payment" or "MyClass.method"

        if src_sym:
            # Try qualified: module.symbol
            candidate = f"{module_name}.{src_sym}"
            if candidate in self._graph.nodes:
                return candidate

            # src_sym might already be qualified (e.g. "MyClass.method")
            parts = src_sym.split(".")
            if len(parts) == 2:
                candidate2 = f"{module_name}.{parts[0]}.{parts[1]}"
                if candidate2 in self._graph.nodes:
                    return candidate2

        # Fall back to module node
        if module_name in self._graph.nodes:
            return module_name

        return None

    def _resolve_target_id(self, rd: ResolvedDependency) -> Optional[str]:
        """
        Determine the target node_id for a resolved dependency.

        Priority:
          1. resolved_symbol (most specific – e.g. "utils.helper")
          2. resolved_module  (e.g. "utils")
          3. None
        """
        if rd.resolved_symbol and rd.resolved_symbol in self._graph.nodes:
            return rd.resolved_symbol
        if rd.resolved_module and rd.resolved_module in self._graph.nodes:
            return rd.resolved_module
        return None


# ============================================================================
# Convenience functions
# ============================================================================

def build_graph(
        project_root: Path,
        parsed_modules: List[ParsedModule],
        exclude_dirs: Optional[List[str]] = None,
) -> CustomDependencyGraph:
    """
    End-to-end convenience: parse → extract → resolve → build graph.

    Only runs the graph-build step; parsing must have been done by the caller.
    """
    builder = GraphBuilder(project_root)
    return builder.build_from_files(project_root, parsed_modules, exclude_dirs)


def build_graph_with_resolution(
        parsed_modules: List[ParsedModule],
        resolution: ProjectResolutionResult,
        project_root: Optional[Path] = None,
) -> CustomDependencyGraph:
    """Build a graph when a pre-computed resolution result is available."""
    builder = GraphBuilder(project_root)
    return builder.build(parsed_modules, resolution)
