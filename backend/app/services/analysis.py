from __future__ import annotations
import math
from pathlib import Path
from typing import Callable, Mapping, Optional
from analyzer.change_analyzer import ChangeAnalyzer, ImpactAnalyzer
from analyzer.impact_scorer import score_impact
from analyzer.graph_engine import CustomDependencyGraph, GraphBuilder
from analyzer.resolver_coordinator import resolve_project
from analyzer.languages.base import ProjectContext
from analyzer.languages.registry import default_registry
from dataset.config import ALL_FEATURES
from backend.app.schemas.models import *
from .decision import DecisionEngine
from .deployment import DeploymentService
from .monitoring import MonitoringService
from .repair import RepairService
from .risk import RiskPredictionService
from .shap import ShapService
from .validation import ValidationService


class AnalysisPipelineError(RuntimeError):
    """A deterministic analysis stage failed; callers must not treat it as empty evidence."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)

class AnalysisService:
    def __init__(self, risk=None, shap=None, decision=None, acquire: Callable | None = None):
        self.risk = risk or RiskPredictionService()
        self.shap = shap or ShapService(getattr(self.risk, "_model", None))
        self.decision = decision or DecisionEngine()
        self.acquire = acquire
        self.repair = RepairService(); self.validation = ValidationService()
        self.deployment = DeploymentService(); self.monitoring = MonitoringService()

    def _acquire(self, request: "PRAnalysisRequest") -> tuple[str, Mapping[str, str], list[str]]:
        if self.acquire: return self.acquire(request)
        sources = dict(request.sources or {})
        if request.diff is not None: return request.diff, sources, []
        if not request.repository_path: raise ValueError("Provide diff or repository_path")
        root = Path(request.repository_path).resolve()
        from git import Repo
        repo = Repo(root)
        base = request.base_sha or "HEAD~1"; head = request.head_sha or "HEAD"
        diff = repo.git.diff(base, head)
        for path in set(self._paths(diff)):
            try: sources[path] = (repo.git.show(f"{head}:{path}"))
            except Exception: pass
        return diff, sources, []

    @staticmethod
    def _paths(diff: str) -> list[str]:
        return [line.split(" b/", 1)[1] for line in diff.splitlines() if line.startswith("diff --git a/") and " b/" in line]

    def analyze(self, request: "PRAnalysisRequest", notify: Callable[[str, dict], None] | None = None) -> AnalysisResponse:
        notify = notify or (lambda event_name, payload: None)
        notify('ANALYSIS_STARTED', {})
        diff, sources, warnings = self._acquire(request)
        def provider(path, version):
            # Source snapshots supplied by adapters/tests may be keyed by version.
            # Never silently fall back from an explicitly requested snapshot to the
            # other version.
            snapshot = (request.sources or {})
            version_snapshot = request.base_sources if version == 'old' else request.head_sources
            if version_snapshot is not None:
                return version_snapshot.get(path)
            version_key = (path, version)
            if version_key in snapshot:
                return snapshot[version_key]
            nested = snapshot.get(version)
            if isinstance(nested, Mapping):
                return nested.get(path)
            if request.repository_path and (version == 'old' and request.base_sha or version == 'new' and request.head_sha):
                import subprocess
                sha = request.base_sha if version == 'old' else request.head_sha
                result = subprocess.run(['git', 'show', f'{sha}:{path}'],
                    cwd=request.repository_path, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
                return result.stdout if result.returncode == 0 else None
            if version == 'old' and request.base_sha:
                return None
            return sources.get(path)
        analyzer = ChangeAnalyzer(source_provider=provider, language_registry=default_registry)
        changes = analyzer.analyze_diff(diff)
        notify('SOURCE_PARSED', {})
        notify('GRAPH_STARTED', {})
        repository_root = Path(request.repository_path).expanduser().resolve() if request.repository_path else None
        graph_evidence = {}
        try:
            graph = self._build_graph(sources, repository_root, graph_evidence)
        except AnalysisPipelineError as exc:
            notify('GRAPH_FAILED', {'stage': exc.stage, 'message': str(exc)})
            raise
        notify('GRAPH_BUILT' if graph.nodes else 'GRAPH_UNAVAILABLE', graph_evidence)
        notify('IMPACT_STARTED', {})
        impact = ImpactAnalyzer(graph).analyze_impact(changes)
        scored_impact = score_impact(impact, changes, graph)
        critical_node_ids = set(scored_impact.critical_changed_nodes)
        origins = {}
        for changed in impact.directly_changed_nodes:
            for node_id in {changed.node_id, *graph.get_transitive_dependents(changed.node_id)}:
                origins.setdefault(node_id, []).append(changed.node_id)
        graph_payload = {'nodes': [{'id': n.node_id, 'label': n.name, 'qualified_name': n.qualified_name, 'file': n.file_path, 'module': n.file_path, 'type': n.node_type.value, 'depth': impact.impact_depth_map.get(n.node_id), 'changed': impact.impact_depth_map.get(n.node_id) == 0, 'critical': n.node_id in critical_node_ids, 'origins': origins.get(n.node_id, []), 'dependencies': [x.node_id for x in graph.get_dependencies(n.node_id)], 'dependents': [x.node_id for x in graph.get_dependents(n.node_id)]} for n in graph.nodes.values()], 'edges': [{'source': e.source, 'target': e.target, 'type': e.edge_type.value} for e in graph.edges], 'pipeline_evidence': graph_evidence}
        def symbol_payload(symbol):
            return {"name": symbol.qualified_name, "type": symbol.symbol_type.value,
                    "change_type": symbol.change_type.value, "file": symbol.file_path,
                    "module": symbol.module_name, "start_line": symbol.start_line, "end_line": symbol.end_line}
        change_details = {
            "changed_files": list(changes.changed_files),
            "changed_symbols": [symbol_payload(item) for item in changes.changed_symbols],
            "changed_functions": [symbol_payload(item) for item in changes.changed_functions],
            "changed_classes": [symbol_payload(item) for item in changes.changed_classes],
            "changed_methods": [symbol_payload(item) for item in changes.changed_methods],
            "dependency_changes": [{"file": item.file_path, "change_type": item.change_type.value,
                                    "dependency_type": item.dep_type.value, "module": item.module,
                                    "names": list(item.names), "old_line": item.old_line, "new_line": item.new_line}
                                   for item in changes.dependency_changes],
            "unsupported_files": [{"file": item[0], "reason": item[1]} for item in changes.unsupported_files],
        }
        impacted_ids = [item.node_id for item in impact.impacted_nodes]
        modules = sorted({graph.nodes[node_id].file_path or graph.nodes[node_id].qualified_name.rpartition(".")[0] or "unknown"
                          for node_id in impact.impact_depth_map if node_id in graph.nodes})
        propagation_paths = self._propagation_paths(graph, [item.node_id for item in impact.directly_changed_nodes], set(impact.impact_depth_map))
        impact_details = {
            "directly_changed_nodes": [item.node_id for item in impact.directly_changed_nodes],
            "impacted_nodes": impacted_ids,
            "impacted_modules": modules,
            "impact_depth": dict(impact.impact_depth_map),
            "critical_nodes": list(scored_impact.critical_changed_nodes),
            "propagation_paths": propagation_paths,
            "cross_module_impact": len(modules) > 1,
        }
        notify('IMPACT_ANALYZED', {'graph': graph_payload})
        features = self._features(changes, impact, scored_impact.features)
        notify('RISK_STARTED', {})
        risk = self.risk.predict(features)
        notify('RISK_COMPLETE' if risk.model_status == 'available' else 'RISK_UNAVAILABLE', {})
        notify('SHAP_STARTED', {})
        shap = self.shap.explain(features, risk)
        notify('SHAP_COMPLETE' if shap.shap_status == 'available' else 'SHAP_UNAVAILABLE', {})
        if changes.unsupported_files:
            warnings.append(f"Unsupported language evidence: {len(changes.unsupported_files)} file(s) remained at file level only.")
            notify('UNSUPPORTED_LANGUAGE', {'files':[item[0] for item in changes.unsupported_files]})
        if not changes.file_diffs:
            warnings.append("The diff contains no analyzable file changes.")
            notify('EMPTY_DIFF', {})
        if not graph.nodes:
            warnings.append("Dependency graph unavailable; no graph facts were fabricated.")
        complete = not bool(changes.unsupported_files) and (bool(changes.file_diffs) or (bool(sources) and request.diff == ''))
        impact_score = features.get("impact_score")
        notify('DECISION_STARTED', {})
        decision = self.decision.decide(risk=risk, impact_score=None if isinstance(impact_score, float) and math.isnan(impact_score) else impact_score, blast_radius=impact.affected_node_count, critical_nodes=int(features["critical_node_count"] if not math.isnan(features["critical_node_count"]) else 0), analysis_complete=complete)
        notify('DECISION_COMPLETE', {})
        languages = sorted({Path(f).suffix.lower().lstrip(".") for f in changes.changed_files if Path(f).suffix})
        return AnalysisResponse(
            graph=graph_payload,
            pr_metadata=PRMetadata(repository=request.repository, pull_request=request.pull_request, base_sha=request.base_sha, head_sha=request.head_sha, title=request.title, body=request.body),
            detected_languages=languages,
            change_summary=ChangeSummary(files_changed=changes.files_changed_count, files_added=len(changes.files_added), files_deleted=len(changes.files_deleted), files_modified=len(changes.files_modified), lines_added=changes.lines_added, lines_deleted=changes.lines_deleted, symbols_changed=len(changes.changed_symbols), unsupported_files=len(changes.unsupported_files)),
            dependency_change_summary=DependencyChangeSummary(added=len(changes.dependencies_added), removed=len(changes.dependencies_removed), modified=len(changes.dependencies_modified)),
            impact_summary=ImpactSummary(impact_score=None if math.isnan(features["impact_score"]) else features["impact_score"], impact_level=scored_impact.impact_level.value, blast_radius=impact.affected_node_count, direct_nodes=len(impact.directly_changed_nodes), impacted_nodes=len(impact.impacted_nodes), maximum_depth=impact.maximum_impact_depth, critical_nodes=int(features["critical_node_count"] if not math.isnan(features["critical_node_count"] ) else 0)),
            change_details=change_details, impact_details=impact_details,
            features={k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features.items()}, model_status=risk.model_status, risk_prediction=risk, shap=shap,
            decision=decision, validation=self.validation.not_run(), repair=RepairSummary(status='unavailable'), deployment=DeploymentSummary(), monitoring=self.monitoring.current(), warnings=warnings, errors=[], analysis_complete=complete)

    @staticmethod
    def _propagation_paths(graph: CustomDependencyGraph, origins: list[str], allowed: set[str]) -> list[list[str]]:
        """Return auditable shortest paths followed by the impact traversal."""
        paths: list[list[str]] = []
        for origin in origins:
            queue = [(origin, [origin])]
            seen = {origin}
            while queue:
                current, path = queue.pop(0)
                if len(path) > 1:
                    paths.append(path)
                for node in graph.get_dependents(current):
                    if node.node_id in allowed and node.node_id not in seen:
                        seen.add(node.node_id)
                        queue.append((node.node_id, [*path, node.node_id]))
        return paths

    def _build_graph(self, sources: Mapping[str, str], repository_root: Path | None = None, evidence: dict | None = None) -> CustomDependencyGraph:
        """Run the existing adapter → IR → resolver → custom graph path."""
        parsed = []
        from analyzer.languages.detector import LanguageDetector
        detector = LanguageDetector()
        adapter_languages = []
        excluded_sources = []
        for name, source in sources.items():
            exclusion_reason = self._graph_source_exclusion_reason(name)
            if exclusion_reason:
                excluded_sources.append({'file': name, 'reason': exclusion_reason})
                continue
            try:
                detected = detector.detect(name)
                adapter = default_registry.get(detected.language)
                if adapter is None:
                    continue
                item = adapter.parse(source, name)
                if not item.has_errors:
                    parsed.append(item)
                    adapter_languages.append(detected.language.value if hasattr(detected.language, 'value') else str(detected.language))
                else:
                    raise AnalysisPipelineError('SOURCE_PARSED', f'Parser reported errors for {name}')
            except Exception as exc:
                raise AnalysisPipelineError('SOURCE_PARSED', f'Parser failed for {name}: {type(exc).__name__}') from exc
        if not parsed:
            return CustomDependencyGraph()
        try:
            # A supplied checkout is authoritative.  cwd is only a context for
            # legacy patch-only callers that have no repository root.
            root = repository_root or Path.cwd()
            resolved = resolve_project(root, parsed)
            graph = GraphBuilder(root).build_from_ir(resolved)
            if evidence is not None:
                evidence.update({'adapter_languages':sorted(set(adapter_languages)), 'parsed_source_count':len(parsed), 'excluded_source_count':len(excluded_sources), 'excluded_sources':excluded_sources, 'ir_symbol_count':sum(len(item.symbols) for item in parsed), 'ir_dependency_count':sum(len(item.dependencies) for item in parsed), 'resolver_status':'completed', 'graph_node_count':len(graph.nodes), 'graph_edge_count':len(graph.edges)})
            return graph
        except Exception as exc:
            # A missing checkout context must not turn a valid diff into fake graph facts.
            raise AnalysisPipelineError('GRAPH_BUILT', f'Resolver or graph construction failed: {type(exc).__name__}') from exc

    @staticmethod
    def _graph_source_exclusion_reason(name: str) -> str | None:
        """Exclude generated third-party artifacts, not relevant first-party source."""
        path = Path(name.replace('\\', '/'))
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts & {'node_modules', 'vendor', 'vendors', 'third_party', 'third-party'}:
            return 'third-party dependency source'
        lowered_name = path.name.lower()
        if lowered_name.endswith(('.min.js', '.min.css')):
            return 'minified generated asset'
        return None

    def _features(self, c, i, scored_features=None) -> dict[str, float]:
        # Patch-only fields are populated below. Repository/graph-dependent fields
        # remain NaN when the request did not supply enough source context.
        f = {name: float("nan") for name in ALL_FEATURES}
        # ImpactAnalyzer is the canonical source for graph-derived features.
        # Keep its complete 42-key result and only overlay compatibility fields.
        f.update(getattr(i, "features", {}) or {})
        f.update(scored_features or {})
        # The canonical analyzer owns these definitions.  This keeps graph
        # values (including critical nodes and propagation depth) consistent.
        # The initial full-profile template uses NaN for unavailable graph
        # facts, so use direct patch evidence whenever that placeholder is
        # still present. This makes the model subset available even for a
        # patch-only analysis without inventing any graph values.
        def populate_if_unavailable(name: str, value: float) -> None:
            current = f.get(name)
            if current is None or (isinstance(current, float) and math.isnan(current)):
                f[name] = value

        populate_if_unavailable("files_changed", float(c.files_changed_count))
        populate_if_unavailable("lines_added", float(c.lines_added))
        populate_if_unavailable("lines_deleted", float(c.lines_deleted))
        populate_if_unavailable("total_lines_changed", float(c.total_lines_changed))
        populate_if_unavailable("functions_changed", float(len(c.changed_functions)))
        populate_if_unavailable("classes_changed", float(len(c.changed_classes)))
        populate_if_unavailable("methods_changed", float(len(c.changed_methods)))
        populate_if_unavailable("dependencies_added", float(len(c.dependencies_added)))
        populate_if_unavailable("dependencies_removed", float(len(c.dependencies_removed)))
        populate_if_unavailable("dependencies_changed", float(len(c.dependency_changes)))
        f.setdefault("direct_nodes_changed", float(len(i.directly_changed_nodes)))
        f.setdefault("impacted_nodes", float(len(i.impacted_nodes)))
        f.setdefault("maximum_impact_depth", float(i.maximum_impact_depth))
        f.setdefault("average_impact_depth", float(i.average_impact_depth))
        return {k: float(v) if not (isinstance(v, float) and math.isnan(v)) else v for k, v in f.items()}

class PRAnalysisRequest(BaseModel):
    repository: str = "local"
    pull_request: Optional[int] = None
    base_sha: Optional[str] = None
    head_sha: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None
    repository_path: Optional[str] = None
    diff: Optional[str] = None
    sources: Optional[dict[str, Any]] = None
    base_sources: Optional[dict[str, str]] = None
    head_sources: Optional[dict[str, str]] = None

class FeatureAnalysisRequest(PRAnalysisRequest):
    pass
