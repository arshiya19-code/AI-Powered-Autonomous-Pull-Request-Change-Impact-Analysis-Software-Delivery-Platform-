"""
AI Change Impact Platform - Analyzer Package

Phases:
1. AST Parser (ast_parser)
2. Dependency Extractor (dependency_extractor)
3. Name Resolver (resolver)
4. Graph Engine (graph_engine)
5. Change Analyzer (change_analyzer)
6. Impact Scorer (impact_scorer)
7. Multi-Language Architecture Foundation (languages)
"""

from .ast_parser import (
    ASTParser,
    ParsedModule,
    ImportInfo,
    ClassInfo,
    FunctionInfo,
    SourceLocation,
    ImportType,
    parse_file,
)

from .dependency_extractor import (
    DependencyExtractor,
    ModuleDependencies,
    Dependency,
    DependencyType,
    extract_dependencies,
)

from .resolver import (
    NameResolver,
    ModuleDiscovery,
    SymbolTableBuilder,
    ResolutionStatus,
    ResolvedDependency,
)

from .graph_engine import (
    CustomDependencyGraph,
    GraphBuilder,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    build_graph,
)

from .change_analyzer import (
    ChangeAnalyzer,
    ImpactAnalyzer,
    SymbolMapper,
    DependencyChangeDetector,
    ChangeSet,
    FileDiff,
    DiffHunk,
    ChangeType,
    DependencyChange,
    DependencyChangeType,
    ChangedSymbol,
    SymbolType,
    ImpactLevel,
    ImpactedNode,
    ImpactAnalysisResult,
    UnifiedDiffParser,
    analyze_diff,
    analyze_impact,
    analyze_diff_and_impact,
)

from .impact_scorer import (
    ImpactScorer,
    ImpactScoringResult,
    PropagationMetrics,
    ConnectivityMetrics,
    NodeConnectivityMetrics,
    OverallImpactLevel,
    score_impact,
    analyze_and_score,
)

# Phase 7 — Multi-Language Architecture Foundation
from .languages import (
    LanguageAdapter,
    ParsedSource,
    Symbol,
    SymbolKind,
    Dependency as LanguageDependency,
    LanguageDependencyType,
    SourceLocation as LanguageSourceLocation,
    ProjectContext,
    DetectedLanguage,
    LanguageDetector,
    LanguageRegistry,
    DuplicateAdapterError,
    UnsupportedLanguageError,
    default_registry,
    PythonLanguageAdapter,
    TypeScriptLanguageAdapter,
    JavaScriptLanguageAdapter,
    CSharpLanguageAdapter,
    RustLanguageAdapter,
    CppLanguageAdapter,
)

__all__ = [
    # Phase 7
    "LanguageAdapter",
    "ParsedSource",
    "Symbol",
    "SymbolKind",
    "LanguageDependency",
    "LanguageDependencyType",
    "LanguageSourceLocation",
    "ProjectContext",
    "DetectedLanguage",
    "LanguageDetector",
    "LanguageRegistry",
    "DuplicateAdapterError",
    "UnsupportedLanguageError",
    "default_registry",
    "PythonLanguageAdapter",
    "TypeScriptLanguageAdapter",
    "JavaScriptLanguageAdapter",
    "CSharpLanguageAdapter",
    "RustLanguageAdapter",
    "CppLanguageAdapter",
    # Phase 1
    "ASTParser",
    "ParsedModule",
    "ImportInfo",
    "ClassInfo",
    "FunctionInfo",
    "SourceLocation",
    "ImportType",
    "parse_file",
    # Phase 2
    "DependencyExtractor",
    "ModuleDependencies",
    "Dependency",
    "DependencyType",
    "extract_dependencies",
    # Phase 3
    "NameResolver",
    "ModuleDiscovery",
    "SymbolTableBuilder",
    "ResolutionStatus",
    "ResolvedDependency",
    # Phase 4
    "CustomDependencyGraph",
    "GraphBuilder",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "EdgeType",
    "build_graph",
    # Phase 5
    "ChangeAnalyzer",
    "ImpactAnalyzer",
    "SymbolMapper",
    "DependencyChangeDetector",
    "ChangeSet",
    "FileDiff",
    "DiffHunk",
    "ChangeType",
    "DependencyChange",
    "DependencyChangeType",
    "ChangedSymbol",
    "SymbolType",
    "ImpactLevel",
    "ImpactedNode",
    "ImpactAnalysisResult",
    "UnifiedDiffParser",
    "analyze_diff",
    "analyze_impact",
    "analyze_diff_and_impact",
    # Phase 6
    "ImpactScorer",
    "ImpactScoringResult",
    "PropagationMetrics",
    "ConnectivityMetrics",
    "NodeConnectivityMetrics",
    "OverallImpactLevel",
    "score_impact",
    "analyze_and_score",
]

__version__ = "0.7.0"
