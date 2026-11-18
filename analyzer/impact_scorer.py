"""
Phase 6 — Custom Change Impact Analysis & Impact Scoring.

Extends Phase 5 impact analysis with:
  1. Nodes-by-depth breakdown
  2. Cross-module impact measurement
  3. Dependency-propagation edge-type counting (CALLS, IMPORT, INHERITS, DECORATES)
  4. Connectivity / structural metrics for changed nodes
     (degree, dependent count, dependency count)
  5. Critical-component detection via mean+2σ threshold on dependent count
  6. Impact Score  [0–100]  — deterministic analytical score
  7. Impact Level  LOW / MEDIUM / HIGH / CRITICAL  based on score thresholds
  8. Extended feature vector suitable for Phase 7 CatBoost model
  9. Human-readable textual explanation (deterministic, no LLM)

Architecture
------------
ChangeSet  (Phase 5)
    │
    ▼
ImpactAnalyzer.analyze_impact()  (Phase 5)
    │
    ▼
ImpactAnalysisResult  (Phase 5)
    │
    ▼
ImpactScorer.score(result, change_set, graph)   ◄──── Phase 6 entry point
    │
    ▼
ImpactScoringResult
    ├── base Phase 5 result (unchanged)
    ├── nodes_by_depth        Dict[depth → List[ImpactedNode]]
    ├── propagation_metrics   PropagationMetrics
    ├── connectivity_metrics  ConnectivityMetrics
    ├── critical_changed_nodes  List[str]  (node_ids)
    ├── impact_score          float  [0–100]
    ├── impact_level          ImpactLevel6  (LOW/MEDIUM/HIGH/CRITICAL)
    ├── features              Dict[str, float]  (extended feature vector)
    └── explanations          List[str]

Impact Score formula (documented below in ImpactScorer._compute_score).

Design constraints:
  - No NetworkX, Neo4j, or external graph libraries.
  - No LLM.
  - Uses CustomDependencyGraph from Phase 4.
  - Uses ImpactAnalysisResult from Phase 5.
  - Deterministic and reproducible.
  - Does NOT modify any Phase 1–5 files.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from analyzer.change_analyzer import (
    ChangeSet,
    ImpactAnalysisResult,
    ImpactedNode,
    ImpactLevel,
    ImpactAnalyzer,
    analyze_impact as _p5_analyze_impact,
)
from analyzer.graph_engine import (
    CustomDependencyGraph,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)


# ============================================================================
# Phase 6 Enumerations
# ============================================================================

class OverallImpactLevel(Enum):
    """
    Classification of overall change impact severity.

    Thresholds are applied to the Impact Score [0–100]:
        LOW       0  – 24   Localised change; few or no dependents affected.
        MEDIUM   25  – 49   Moderate spread; some cross-module reach.
        HIGH     50  – 74   Broad impact; many components affected.
        CRITICAL 75  – 100  Severe; highly connected components changed
                            with deep transitive propagation.

    These quarter-scale thresholds provide equal-width bins that are easy to
    reason about and map directly to common severity scales (LOW/MED/HIGH/CRIT).
    """
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "OverallImpactLevel":
        """Classify a score in [0, 100] into an impact level."""
        if score < 25.0:
            return cls.LOW
        elif score < 50.0:
            return cls.MEDIUM
        elif score < 75.0:
            return cls.HIGH
        else:
            return cls.CRITICAL


# ============================================================================
# Supplementary data structures
# ============================================================================

@dataclass
class PropagationMetrics:
    """
    Counts of each edge type traversed during impact propagation.

    These are collected by examining edges incident to impacted nodes.
    They reveal *how* the change propagates through the codebase:

    calls_count       — CALLS edges in the impact subgraph
    import_count      — IMPORT edges in the impact subgraph
    from_import_count — FROM_IMPORT edges in the impact subgraph
    inherits_count    — INHERITS edges (inheritance impact — high severity)
    decorates_count   — DECORATES edges (decorator impact)
    contains_count    — CONTAINS edges (structural containment)
    total_count       — sum of all above

    Note: CONTAINS edges are structural, not dependency edges, so they
    are tracked separately and excluded from the non-structural total.
    """
    calls_count:       int = 0
    import_count:      int = 0
    from_import_count: int = 0
    inherits_count:    int = 0
    decorates_count:   int = 0
    contains_count:    int = 0

    @property
    def dependency_edge_total(self) -> int:
        """Total dependency edges (excludes structural CONTAINS edges)."""
        return (
            self.calls_count
            + self.import_count
            + self.from_import_count
            + self.inherits_count
            + self.decorates_count
        )

    @property
    def total_count(self) -> int:
        """Total of all edge types including structural."""
        return self.dependency_edge_total + self.contains_count

    def __repr__(self) -> str:
        return (
            f"PropagationMetrics(calls={self.calls_count}, "
            f"import={self.import_count}, from_import={self.from_import_count}, "
            f"inherits={self.inherits_count}, decorates={self.decorates_count})"
        )


@dataclass
class NodeConnectivityMetrics:
    """
    Structural connectivity metrics for a single graph node.

    These are *not* formal graph-centrality measures (PageRank, betweenness, etc.).
    They are simple, transparent degree-based metrics:

    in_degree   — number of edges pointing INTO this node (nodes that depend on it)
    out_degree  — number of edges leaving this node (nodes this depends on)
    degree      — in_degree + out_degree  (total adjacency)
    dependent_count  — synonym for in_degree (semantically: dependents)
    dependency_count — synonym for out_degree (semantically: dependencies)
    """
    node_id:          str
    in_degree:        int = 0
    out_degree:       int = 0

    @property
    def degree(self) -> int:
        return self.in_degree + self.out_degree

    @property
    def dependent_count(self) -> int:
        """How many nodes depend on this node (in-degree)."""
        return self.in_degree

    @property
    def dependency_count(self) -> int:
        """How many nodes this node depends on (out-degree)."""
        return self.out_degree


@dataclass
class ConnectivityMetrics:
    """
    Aggregate connectivity metrics across all directly changed nodes.

    per_node    — NodeConnectivityMetrics for each directly changed node
    max_degree            — highest degree among changed nodes
    avg_degree            — mean degree across changed nodes
    max_dependent_count   — highest in_degree (most dependents)
    avg_dependent_count   — mean in_degree across changed nodes
    max_dependency_count  — highest out_degree
    avg_dependency_count  — mean out_degree across changed nodes

    These aggregates reveal whether the change touches highly connected
    ('hub') components, which typically have higher blast radius.
    """
    per_node: Dict[str, NodeConnectivityMetrics] = field(default_factory=dict)

    @property
    def max_degree(self) -> int:
        if not self.per_node:
            return 0
        return max(m.degree for m in self.per_node.values())

    @property
    def avg_degree(self) -> float:
        if not self.per_node:
            return 0.0
        return statistics.mean(m.degree for m in self.per_node.values())

    @property
    def max_dependent_count(self) -> int:
        if not self.per_node:
            return 0
        return max(m.dependent_count for m in self.per_node.values())

    @property
    def avg_dependent_count(self) -> float:
        if not self.per_node:
            return 0.0
        return statistics.mean(m.dependent_count for m in self.per_node.values())

    @property
    def max_dependency_count(self) -> int:
        if not self.per_node:
            return 0
        return max(m.dependency_count for m in self.per_node.values())

    @property
    def avg_dependency_count(self) -> float:
        if not self.per_node:
            return 0.0
        return statistics.mean(m.dependency_count for m in self.per_node.values())


# ============================================================================
# Core Phase 6 result
# ============================================================================

@dataclass
class ImpactScoringResult:
    """
    Extended impact analysis result produced by Phase 6.

    Wraps the Phase 5 ImpactAnalysisResult and adds scoring, metrics, and
    explanation data.

    Fields
    ------
    base_result             Phase 5 ImpactAnalysisResult (unchanged)
    nodes_by_depth          Dict mapping BFS depth → list of ImpactedNodes
    changed_modules         Set of module names that were directly changed
    impacted_module_names   Set of distinct module names in impacted_nodes
    cross_module_impact_count  Number of distinct modules affected beyond changed ones
    propagation_metrics     Edge-type breakdown of how impact propagates
    connectivity_metrics    Degree metrics for directly changed nodes
    critical_changed_nodes  node_ids flagged as structurally critical
    impact_score            Float [0–100] — deterministic analytical score
    impact_level            LOW / MEDIUM / HIGH / CRITICAL
    features                Extended numeric feature dict (for Phase 7)
    explanations            Human-readable impact explanation lines
    """
    base_result:               ImpactAnalysisResult
    nodes_by_depth:            Dict[int, List[ImpactedNode]] = field(default_factory=dict)
    changed_modules:           Set[str] = field(default_factory=set)
    impacted_module_names:     Set[str] = field(default_factory=set)
    cross_module_impact_count: int = 0
    propagation_metrics:       PropagationMetrics = field(default_factory=PropagationMetrics)
    connectivity_metrics:      ConnectivityMetrics = field(default_factory=ConnectivityMetrics)
    critical_changed_nodes:    List[str] = field(default_factory=list)
    impact_score:              float = 0.0
    impact_level:              OverallImpactLevel = OverallImpactLevel.LOW
    features:                  Dict[str, float] = field(default_factory=dict)
    explanations:              List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience pass-throughs to the base result
    # ------------------------------------------------------------------

    @property
    def directly_changed_nodes(self) -> List[ImpactedNode]:
        return self.base_result.directly_changed_nodes

    @property
    def impacted_nodes(self) -> List[ImpactedNode]:
        return self.base_result.impacted_nodes

    @property
    def impacted_modules(self) -> List[ImpactedNode]:
        return self.base_result.impacted_modules

    @property
    def impacted_classes(self) -> List[ImpactedNode]:
        return self.base_result.impacted_classes

    @property
    def impacted_functions(self) -> List[ImpactedNode]:
        return self.base_result.impacted_functions

    @property
    def impacted_methods(self) -> List[ImpactedNode]:
        return self.base_result.impacted_methods

    @property
    def maximum_impact_depth(self) -> int:
        return self.base_result.maximum_impact_depth

    @property
    def average_impact_depth(self) -> float:
        return self.base_result.average_impact_depth

    @property
    def critical_node_count(self) -> int:
        return len(self.critical_changed_nodes)

    @property
    def cross_module_ratio(self) -> float:
        """Fraction of all affected modules that are cross-module impact."""
        total = len(self.changed_modules) + self.cross_module_impact_count
        if total == 0:
            return 0.0
        return self.cross_module_impact_count / total

    def __repr__(self) -> str:
        return (
            f"ImpactScoringResult("
            f"score={self.impact_score:.1f}, "
            f"level={self.impact_level.value}, "
            f"direct={len(self.directly_changed_nodes)}, "
            f"impacted={len(self.impacted_nodes)}, "
            f"depth={self.maximum_impact_depth})"
        )


# ============================================================================
# Impact Scorer — Phase 6 engine
# ============================================================================

class ImpactScorer:
    """
    Phase 6 scoring engine.

    Accepts a Phase 5 ImpactAnalysisResult and the original ChangeSet and
    CustomDependencyGraph, and produces a full ImpactScoringResult.

    Public API
    ----------
    score(result, change_set, graph) → ImpactScoringResult
    """

    # Weights for each scoring component (must sum to 1.0)
    # These are documented proportions, not arbitrary:
    #   Breadth  (0.30) — how many components are potentially affected
    #   Depth    (0.20) — how deeply the change propagates
    #   Cross-mod(0.20) — how many separate modules are involved
    #   Connectivity (0.20) — whether a high-degree hub node was changed
    #   Dep-propagation (0.10) — edge-type diversity of propagation
    _W_BREADTH       = 0.30
    _W_DEPTH         = 0.20
    _W_CROSS_MODULE  = 0.20
    _W_CONNECTIVITY  = 0.20
    _W_PROPAGATION   = 0.10

    # Depth cap for normalisation (depth beyond this is treated as maximum)
    # 8 hops is a defensible upper bound: most real codebases rarely exceed
    # this before a natural architectural boundary absorbs the impact.
    _MAX_DEPTH_CAP = 8

    # Critical-node threshold: a changed node is considered structurally
    # critical when its dependent count exceeds mean + 2 * std_dev of
    # in-degree across ALL graph nodes.  2σ captures the top ~2.3% of nodes
    # under a normal distribution — the genuine "hubs" of the codebase.
    _CRITICAL_SIGMA_MULTIPLIER = 2.0

    def __init__(self, graph: CustomDependencyGraph) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def score(
        self,
        base_result: ImpactAnalysisResult,
        change_set: ChangeSet,
    ) -> ImpactScoringResult:
        """
        Run Phase 6 scoring on a Phase 5 ImpactAnalysisResult.

        Args:
            base_result : Output of ImpactAnalyzer.analyze_impact()
            change_set  : The ChangeSet that produced base_result

        Returns:
            ImpactScoringResult with all Phase 6 fields populated.
        """
        result = ImpactScoringResult(base_result=base_result)

        # 1. Depth breakdown
        result.nodes_by_depth = self._build_depth_map(base_result)

        # 2. Module-level cross-module analysis
        result.changed_modules = self._extract_changed_modules(base_result, change_set)
        result.impacted_module_names = self._extract_impacted_module_names(base_result)
        result.cross_module_impact_count = len(
            result.impacted_module_names - result.changed_modules
        )

        # 3. Propagation metrics (edge-type counts in impact subgraph)
        result.propagation_metrics = self._compute_propagation_metrics(base_result)

        # 4. Connectivity metrics for directly changed nodes
        result.connectivity_metrics = self._compute_connectivity_metrics(base_result)

        # 5. Critical-node detection
        result.critical_changed_nodes = self._detect_critical_nodes(base_result)

        # 6. Impact Score
        result.impact_score = self._compute_score(result, change_set)

        # 7. Impact Level
        result.impact_level = OverallImpactLevel.from_score(result.impact_score)

        # 8. Extended feature vector
        result.features = self._build_features(result, change_set)

        # 9. Human-readable explanations
        result.explanations = self._build_explanations(result, change_set)

        return result

    # ------------------------------------------------------------------
    # Step 1 — Depth breakdown
    # ------------------------------------------------------------------

    def _build_depth_map(
        self, base: ImpactAnalysisResult
    ) -> Dict[int, List[ImpactedNode]]:
        """
        Group all nodes (directly changed + impacted) by their BFS depth.

        Depth 0 = directly changed nodes.
        Depth 1 = direct dependents.
        Depth 2+ = transitive dependents.
        """
        by_depth: Dict[int, List[ImpactedNode]] = {}

        for node in base.directly_changed_nodes:
            by_depth.setdefault(0, []).append(node)

        for node in base.impacted_nodes:
            d = base.impact_depth_map.get(node.node_id, node.impact_depth)
            by_depth.setdefault(d, []).append(node)

        return by_depth

    # ------------------------------------------------------------------
    # Step 2 — Cross-module analysis
    # ------------------------------------------------------------------

    def _extract_changed_modules(
        self, base: ImpactAnalysisResult, cs: ChangeSet
    ) -> Set[str]:
        """
        Determine the set of module names directly touched by the change.

        Sources (in order of preference):
          1. file_path of directly changed nodes → module-qualified component
          2. module_name field of changed symbols in the ChangeSet
        """
        modules: Set[str] = set()

        for node in base.directly_changed_nodes:
            mod = self._module_from_node(node)
            if mod:
                modules.add(mod)

        if hasattr(cs, "changed_symbols"):
            for sym in cs.changed_symbols:
                if sym.module_name:
                    modules.add(sym.module_name)

        return modules

    def _extract_impacted_module_names(
        self, base: ImpactAnalysisResult
    ) -> Set[str]:
        """Collect distinct module names of all indirectly impacted nodes."""
        modules: Set[str] = set()
        for node in base.impacted_nodes:
            mod = self._module_from_node(node)
            if mod:
                modules.add(mod)
        return modules

    def _module_from_node(self, node: ImpactedNode) -> Optional[str]:
        """
        Derive a module name from an ImpactedNode.

        Strategy:
          - For MODULE nodes: the node_id IS the module name.
          - For CLASS/FUNCTION/METHOD: the module is the node_id up to the
            first component that isn't a class or method name.
            As a heuristic: take all but the last component for CLASS/FUNCTION,
            all but the last two for METHOD.
        """
        if node.node_type == NodeType.MODULE:
            return node.node_id
        if node.node_type in (NodeType.CLASS, NodeType.FUNCTION):
            parts = node.node_id.split(".")
            return ".".join(parts[:-1]) if len(parts) > 1 else node.node_id
        if node.node_type == NodeType.METHOD:
            parts = node.node_id.split(".")
            return ".".join(parts[:-2]) if len(parts) > 2 else node.node_id
        return None

    # ------------------------------------------------------------------
    # Step 3 — Propagation metrics
    # ------------------------------------------------------------------

    def _compute_propagation_metrics(
        self, base: ImpactAnalysisResult
    ) -> PropagationMetrics:
        """
        Count each edge type in the subgraph induced by all impacted nodes.

        We examine every edge in the graph whose source or target is in the
        impacted-or-changed set. This reveals the propagation mechanism:
        e.g., a change propagating mainly through CALLS is different from
        one propagating through INHERITS (inheritance cascades are typically
        harder to test exhaustively).
        """
        all_impacted_ids: Set[str] = (
            {n.node_id for n in base.directly_changed_nodes}
            | {n.node_id for n in base.impacted_nodes}
        )

        metrics = PropagationMetrics()

        for edge in self._graph.edges:
            # Only count edges that connect two nodes in the impact set
            if edge.source in all_impacted_ids and edge.target in all_impacted_ids:
                if edge.edge_type == EdgeType.CALLS:
                    metrics.calls_count += 1
                elif edge.edge_type == EdgeType.IMPORT:
                    metrics.import_count += 1
                elif edge.edge_type == EdgeType.FROM_IMPORT:
                    metrics.from_import_count += 1
                elif edge.edge_type == EdgeType.INHERITS:
                    metrics.inherits_count += 1
                elif edge.edge_type == EdgeType.DECORATES:
                    metrics.decorates_count += 1
                elif edge.edge_type == EdgeType.CONTAINS:
                    metrics.contains_count += 1

        return metrics

    # ------------------------------------------------------------------
    # Step 4 — Connectivity metrics
    # ------------------------------------------------------------------

    def _compute_connectivity_metrics(
        self, base: ImpactAnalysisResult
    ) -> ConnectivityMetrics:
        """
        Compute in/out-degree for every directly changed node.

        Uses the graph's outgoing and incoming adjacency dicts, which are
        already maintained by CustomDependencyGraph for O(1) access.
        """
        cm = ConnectivityMetrics()

        for node in base.directly_changed_nodes:
            nid = node.node_id
            in_deg  = len(self._graph.incoming.get(nid, set()))
            out_deg = len(self._graph.outgoing.get(nid, set()))
            cm.per_node[nid] = NodeConnectivityMetrics(
                node_id=nid,
                in_degree=in_deg,
                out_degree=out_deg,
            )

        return cm

    # ------------------------------------------------------------------
    # Step 5 — Critical-node detection
    # ------------------------------------------------------------------

    def _detect_critical_nodes(
        self, base: ImpactAnalysisResult
    ) -> List[str]:
        """
        Flag directly changed nodes that are structurally critical.

        Definition
        ----------
        A node is structurally critical when its dependent count (in-degree)
        exceeds:
            μ  +  2σ
        where μ and σ are the mean and standard deviation of in-degree
        across ALL nodes in the graph.

        Rationale
        ---------
        - Using project-relative statistics avoids hardcoded thresholds.
        - 2σ (≈97.7th percentile under normality) captures genuine hubs
          while avoiding false positives on medium-connectivity nodes.
        - Inheritance (INHERITS) edges contribute to in-degree, so nodes
          that are widely subclassed are correctly flagged as critical.

        Fallback
        ---------
        If the graph has fewer than 3 nodes (insufficient for σ to be
        meaningful), a node is flagged as critical when its in-degree
        is strictly greater than 0.
        """
        all_in_degrees = [
            len(self._graph.incoming.get(nid, set()))
            for nid in self._graph.nodes
        ]

        if len(all_in_degrees) >= 3:
            mean_deg = statistics.mean(all_in_degrees)
            std_deg  = statistics.pstdev(all_in_degrees)  # population stdev
            threshold = mean_deg + self._CRITICAL_SIGMA_MULTIPLIER * std_deg
        else:
            # Tiny graph — anything with at least one dependent is notable
            threshold = 0.0

        critical: List[str] = []
        for node in base.directly_changed_nodes:
            in_deg = len(self._graph.incoming.get(node.node_id, set()))
            if in_deg > threshold:
                critical.append(node.node_id)

        return critical

    # ------------------------------------------------------------------
    # Step 6 — Impact Score
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        result: ImpactScoringResult,
        cs: ChangeSet,
    ) -> float:
        """
        Compute the Impact Score in [0, 100].

        The score is a weighted sum of five normalised components, each
        capped to [0, 1] before weighting:

        Component          Weight  What it captures
        ---------          ------  ----------------
        breadth_component   0.30  How many nodes are impacted relative to the
                                  total graph size (impact_ratio from Phase 5).
        depth_component     0.20  How deeply the change propagates.
                                  Normalised against _MAX_DEPTH_CAP (8 hops).
        cross_module_comp   0.20  How many distinct modules beyond the changed
                                  ones are affected.  Normalised against the
                                  total module count in the graph.
        connectivity_comp   0.20  Whether the changed node is a high-degree hub.
                                  Uses max_dependent_count / (mean_all_in + ε).
        propagation_comp    0.10  Diversity and volume of propagation edge types.
                                  Normalised: dep_edges / (total_edges + ε).

        Final score = 100 × Σ(weight × component)

        Why these weights?
        - Breadth (0.30) is the most direct measure of blast radius.
        - Depth (0.20) and cross-module (0.20) are independent structural
          dimensions of equal importance.
        - Connectivity (0.20) captures whether a hub was touched.
        - Propagation diversity (0.10) is supporting evidence, not primary.
        """
        base = result.base_result
        total_nodes = max(len(self._graph.nodes), 1)

        # --- Breadth component (0–1) ---
        # Transitive impact fraction of the total graph.
        # Only count transitively impacted nodes (dependents), not the
        # directly changed nodes themselves — they are the source, not the blast radius.
        all_affected = len(base.impacted_nodes)
        breadth = min(all_affected / total_nodes, 1.0)

        # --- Depth component (0–1) ---
        # normalised against the cap; deeper chains = higher contribution
        max_depth = base.maximum_impact_depth
        depth = min(max_depth / self._MAX_DEPTH_CAP, 1.0)

        # --- Cross-module component (0–1) ---
        total_module_nodes = max(
            len(self._graph.nodes_by_type(NodeType.MODULE)), 1
        )
        cross_mod = min(result.cross_module_impact_count / total_module_nodes, 1.0)

        # --- Connectivity component (0–1) ---
        # Uses max in-degree of changed nodes vs project mean in-degree.
        # Rationale: if the most-connected changed node has k times the average
        # in-degree, the component scales with that ratio, capped at 1.
        all_in = [
            len(self._graph.incoming.get(nid, set()))
            for nid in self._graph.nodes
        ]
        mean_in = statistics.mean(all_in) if all_in else 0.0
        max_changed_in = result.connectivity_metrics.max_dependent_count
        if mean_in > 0:
            connectivity = min(max_changed_in / (mean_in * 4.0), 1.0)
        else:
            # graph has no edges — any changed node with in_deg > 0 is notable
            connectivity = min(max_changed_in / 4.0, 1.0) if max_changed_in > 0 else 0.0

        # --- Propagation diversity component (0–1) ---
        pm = result.propagation_metrics
        total_graph_edges = max(len(self._graph.edges), 1)
        propagation = min(pm.dependency_edge_total / total_graph_edges, 1.0)

        # --- Weighted sum → [0, 100] ---
        raw = (
            self._W_BREADTH      * breadth
            + self._W_DEPTH      * depth
            + self._W_CROSS_MODULE * cross_mod
            + self._W_CONNECTIVITY * connectivity
            + self._W_PROPAGATION  * propagation
        )
        return round(min(max(raw * 100.0, 0.0), 100.0), 2)

    # ------------------------------------------------------------------
    # Step 7 — Feature vector
    # ------------------------------------------------------------------

    def _build_features(
        self,
        result: ImpactScoringResult,
        cs: ChangeSet,
    ) -> Dict[str, float]:
        """
        Build the extended feature vector for Phase 7 (CatBoost).

        Every feature has a documented semantic meaning.
        All values are numeric (float) to be directly ingestible by the model.

        Phase 5 base features are preserved and extended with Phase 6 metrics.
        """
        base  = result.base_result
        pm    = result.propagation_metrics
        cm    = result.connectivity_metrics

        # Start with Phase 5 features as base
        features: Dict[str, float] = dict(base.features)

        # --- Cross-module features ---
        features["changed_modules"]           = float(len(result.changed_modules))
        features["cross_module_impact_count"] = float(result.cross_module_impact_count)
        features["cross_module_ratio"]        = result.cross_module_ratio

        # --- Connectivity features ---
        features["average_changed_node_degree"]      = cm.avg_degree
        features["maximum_changed_node_degree"]      = float(cm.max_degree)
        features["average_changed_node_dependents"]  = cm.avg_dependent_count
        features["maximum_changed_node_dependents"]  = float(cm.max_dependent_count)
        features["average_changed_node_dependencies"] = cm.avg_dependency_count
        features["maximum_changed_node_dependencies"] = float(cm.max_dependency_count)

        # --- Propagation edge-type features ---
        features["import_propagation_count"]      = float(pm.import_count)
        features["from_import_propagation_count"] = float(pm.from_import_count)
        features["call_propagation_count"]        = float(pm.calls_count)
        features["inheritance_propagation_count"] = float(pm.inherits_count)
        features["decoration_propagation_count"]  = float(pm.decorates_count)
        features["total_propagation_edges"]       = float(pm.dependency_edge_total)

        # --- Critical-node features ---
        features["critical_node_count"]           = float(result.critical_node_count)

        # --- Depth distribution features ---
        depths = list(base.impact_depth_map.values())
        features["nodes_at_depth_0"] = float(len(result.nodes_by_depth.get(0, [])))
        features["nodes_at_depth_1"] = float(len(result.nodes_by_depth.get(1, [])))
        features["nodes_at_depth_2"] = float(len(result.nodes_by_depth.get(2, [])))
        features["nodes_at_depth_3_plus"] = float(sum(
            len(v) for k, v in result.nodes_by_depth.items() if k >= 3
        ))

        # --- Impact score & level ---
        features["impact_score"]         = result.impact_score
        features["impact_level_numeric"] = float(
            ["low", "medium", "high", "critical"].index(result.impact_level.value)
        )

        return features

    # ------------------------------------------------------------------
    # Step 8 — Human-readable explanations
    # ------------------------------------------------------------------

    def _build_explanations(
        self,
        result: ImpactScoringResult,
        cs: ChangeSet,
    ) -> List[str]:
        """
        Generate bullet-point explanations suitable for dashboard display.

        These are produced deterministically from the computed metrics.
        No LLM or external service is used.

        The explanations answer: "Why does this change have this impact level?"
        """
        lines: List[str] = []
        base = result.base_result
        pm   = result.propagation_metrics
        cm   = result.connectivity_metrics

        direct_count  = len(base.directly_changed_nodes)
        impacted_count = len(base.impacted_nodes)
        max_depth     = base.maximum_impact_depth
        total_affected = direct_count + impacted_count

        # --- Change scope ---
        if hasattr(cs, "files_changed_count"):
            fc = cs.files_changed_count
            lines.append(f"{fc} file{'s' if fc != 1 else ''} changed in this diff.")

        if direct_count == 0:
            lines.append("No graph nodes were directly matched to changed code.")
            lines.append("Impact analysis is based on file-level changes only.")
            return lines

        lines.append(
            f"{direct_count} component{'s were' if direct_count != 1 else ' was'} "
            f"directly modified."
        )

        # --- Blast radius ---
        if impacted_count == 0:
            lines.append("No additional components are transitively affected.")
        else:
            lines.append(
                f"{impacted_count} additional component{'s' if impacted_count != 1 else ''} "
                f"may be affected through dependency propagation."
            )

        # --- Depth ---
        if max_depth > 0:
            lines.append(
                f"Impact propagates to a maximum depth of {max_depth} "
                f"hop{'s' if max_depth != 1 else ''} through the dependency graph."
            )

        # --- Cross-module ---
        if result.cross_module_impact_count > 0:
            lines.append(
                f"Impact spans {result.cross_module_impact_count} module"
                f"{'s' if result.cross_module_impact_count != 1 else ''} "
                f"beyond the directly changed code "
                f"({len(result.changed_modules)} module"
                f"{'s' if len(result.changed_modules) != 1 else ''} directly changed)."
            )
        else:
            lines.append(
                "Impact is contained within the directly changed module(s) "
                f"({len(result.changed_modules)} module"
                f"{'s' if len(result.changed_modules) != 1 else ''})."
            )

        # --- Connectivity ---
        max_dep = cm.max_dependent_count
        if max_dep > 0:
            lines.append(
                f"The most-connected changed component has {max_dep} "
                f"direct dependent{'s' if max_dep != 1 else ''}."
            )

        # --- Critical nodes ---
        if result.critical_node_count > 0:
            lines.append(
                f"{result.critical_node_count} changed component"
                f"{'s are' if result.critical_node_count != 1 else ' is'} "
                f"structurally critical (above mean+2σ dependent count)."
            )

        # --- Propagation breakdown ---
        if pm.calls_count > 0:
            lines.append(
                f"{pm.calls_count} CALLS edge"
                f"{'s' if pm.calls_count != 1 else ''} propagate the impact."
            )
        if pm.from_import_count > 0:
            lines.append(
                f"{pm.from_import_count} FROM_IMPORT edge"
                f"{'s' if pm.from_import_count != 1 else ''} propagate the impact."
            )
        if pm.import_count > 0:
            lines.append(
                f"{pm.import_count} IMPORT edge"
                f"{'s' if pm.import_count != 1 else ''} propagate the impact."
            )
        if pm.inherits_count > 0:
            lines.append(
                f"{pm.inherits_count} INHERITS edge"
                f"{'s' if pm.inherits_count != 1 else ''} propagate the impact "
                f"(inheritance cascades increase testing burden)."
            )
        if pm.decorates_count > 0:
            lines.append(
                f"{pm.decorates_count} DECORATES edge"
                f"{'s' if pm.decorates_count != 1 else ''} propagate the impact."
            )

        # --- Overall verdict ---
        lines.append(
            f"Overall Impact Score: {result.impact_score:.1f}/100 "
            f"({result.impact_level.value.upper()})."
        )

        return lines


# ============================================================================
# Convenience functions
# ============================================================================

def score_impact(
    base_result: ImpactAnalysisResult,
    change_set: ChangeSet,
    graph: CustomDependencyGraph,
) -> ImpactScoringResult:
    """
    Convenience function: run Phase 6 scoring on a Phase 5 result.

    Args:
        base_result : ImpactAnalysisResult from Phase 5 ImpactAnalyzer
        change_set  : The ChangeSet that produced base_result
        graph       : The CustomDependencyGraph used during Phase 5 analysis

    Returns:
        ImpactScoringResult (Phase 6 full result)
    """
    scorer = ImpactScorer(graph)
    return scorer.score(base_result, change_set)


def analyze_and_score(
    change_set: ChangeSet,
    graph: CustomDependencyGraph,
) -> ImpactScoringResult:
    """
    Convenience function: run Phase 5 impact analysis + Phase 6 scoring.

    Args:
        change_set : ChangeSet from Phase 5 ChangeAnalyzer
        graph      : Populated CustomDependencyGraph (Phase 4)

    Returns:
        ImpactScoringResult
    """
    base_result = _p5_analyze_impact(change_set, graph)
    scorer = ImpactScorer(graph)
    return scorer.score(base_result, change_set)
