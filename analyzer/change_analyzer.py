"""
Custom Change Analyzer — Phase 5.

Analyses a Git unified diff and determines:
  1. What files / lines changed
  2. What functions, classes and methods changed
  3. What import-level dependencies changed
  4. Which graph nodes were directly changed
  5. Which graph nodes are potentially impacted (transitive)
  6. Quantitative change-magnitude features for the risk model

Architecture
------------
Git Diff (text)
    │
    ▼
UnifiedDiffParser  ──► List[FileDiff]
    │
    ▼
ChangeAnalyzer
    ├── language-aware IR mappers (adapter → ParsedSource)
    └── legacy Python mappers (only when no registry is supplied)
    └── ChangeSet
         │
         ▼
ImpactAnalyzer  ──► (uses Phase 4 CustomDependencyGraph)
         │
         ▼
ImpactAnalysisResult

Design constraints
------------------
- No NetworkX, Neo4j, or external graph libraries.
- No heavyweight git-diff library; diff is parsed from a plain text string.
- The IR-first path integrates only with language-neutral adapter output and
  the custom graph; legacy Python helpers remain an explicit compatibility
  fallback.
- analyze_diff(diff_text) works without a real Git repository.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Callable, Dict, Iterator, List, Optional, Set, Tuple
)

from analyzer.ast_parser import (
    ASTParser,
    ImportInfo,
    ImportType,
)
from analyzer.graph_engine import (
    CustomDependencyGraph,
    GraphNode,
    NodeType,
)
from analyzer.languages.base import (
    Dependency,
    LanguageAdapter,
    SymbolKind,
    LanguageDependencyType,
)
from analyzer.languages.registry import LanguageRegistry


# ============================================================================
# Enumerations
# ============================================================================

class ChangeType(Enum):
    """How a file or symbol changed."""
    ADDED    = "added"
    DELETED  = "deleted"
    MODIFIED = "modified"
    RENAMED  = "renamed"


class SymbolType(Enum):
    """Type of code symbol."""
    MODULE   = "module"
    CLASS    = "class"
    FUNCTION = "function"
    METHOD   = "method"


class ImpactLevel(Enum):
    """How a node is affected by a change."""
    DIRECT               = "direct"               # node itself changed
    DIRECT_DEPENDENT     = "direct_dependent"     # 1-hop dependent
    TRANSITIVE_DEPENDENT = "transitive_dependent" # 2+ hop dependent


class DependencyChangeType(Enum):
    """Type of import-level change."""
    IMPORT_ADDED    = "import_added"
    IMPORT_REMOVED  = "import_removed"
    IMPORT_MODIFIED = "import_modified"   # same module, different names


# ============================================================================
# Low-level diff data structures
# ============================================================================

@dataclass
class DiffHunk:
    """
    One unified-diff hunk: ``@@ -old_start,old_count +new_start,new_count @@``.

    ``lines`` is the raw list of lines inside the hunk, each prefixed with
    ' ' (context), '+' (added) or '-' (removed).
    """
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str] = field(default_factory=list)

    @property
    def added_line_numbers(self) -> List[int]:
        """Line numbers (1-indexed, in the new file) that were added."""
        result: List[int] = []
        new_line = self.new_start
        for raw in self.lines:
            if raw.startswith("-"):
                pass  # old-file line; new file number does not advance
            elif raw.startswith("+"):
                result.append(new_line)
                new_line += 1
            else:
                new_line += 1  # context line
        return result

    @property
    def deleted_line_numbers(self) -> List[int]:
        """Line numbers (1-indexed, in the old file) that were deleted."""
        result: List[int] = []
        old_line = self.old_start
        for raw in self.lines:
            if raw.startswith("-"):
                result.append(old_line)
                old_line += 1
            elif raw.startswith("+"):
                pass  # new-file line; old file number does not advance
            else:
                old_line += 1  # context line
        return result

    @property
    def affected_new_range(self) -> Tuple[int, int]:
        """(start, end) line range in the new file touched by this hunk."""
        if self.new_count == 0:
            return (self.new_start, self.new_start)
        return (self.new_start, self.new_start + self.new_count - 1)

    @property
    def affected_old_range(self) -> Tuple[int, int]:
        """(start, end) line range in the old file touched by this hunk."""
        if self.old_count == 0:
            return (self.old_start, self.old_start)
        return (self.old_start, self.old_start + self.old_count - 1)


@dataclass
class FileDiff:
    """
    The complete diff for a single file.

    ``file_path`` is the path as it appears in the new tree (right side).
    For deleted files it is the old path.
    ``old_path`` differs from ``file_path`` only for renames.
    """
    file_path: str
    old_path:  str
    change_type: ChangeType
    hunks: List[DiffHunk] = field(default_factory=list)

    @property
    def added_line_numbers(self) -> List[int]:
        """All new-file line numbers that were added."""
        result: List[int] = []
        for hunk in self.hunks:
            result.extend(hunk.added_line_numbers)
        return result

    @property
    def deleted_line_numbers(self) -> List[int]:
        """All old-file line numbers that were deleted."""
        result: List[int] = []
        for hunk in self.hunks:
            result.extend(hunk.deleted_line_numbers)
        return result

    @property
    def lines_added(self) -> int:
        return len(self.added_line_numbers)

    @property
    def lines_deleted(self) -> int:
        return len(self.deleted_line_numbers)

    @property
    def affected_new_ranges(self) -> List[Tuple[int, int]]:
        """New-file (start, end) ranges touched by any hunk."""
        return [h.affected_new_range for h in self.hunks]

    @property
    def affected_old_ranges(self) -> List[Tuple[int, int]]:
        """Old-file (start, end) ranges touched by any hunk."""
        return [h.affected_old_range for h in self.hunks]


# ============================================================================
# Symbol and dependency change structures
# ============================================================================

@dataclass
class ChangedSymbol:
    """
    A function, class, method, or module that was directly affected by a change.

    Attributes
    ----------
    qualified_name : Fully-qualified Python name, e.g. ``services.payment.PaymentService.process``
    symbol_type    : MODULE / CLASS / FUNCTION / METHOD
    file_path      : Source file (new path for ADDED/MODIFIED, old for DELETED)
    change_type    : ADDED / DELETED / MODIFIED
    module_name    : Enclosing module name (dotted)
    class_name     : Enclosing class name if this is a method (None otherwise)
    start_line     : Start line in the new file (0 if not applicable)
    end_line       : End line in the new file (0 if not applicable)
    old_start_line : Start line in the old file (0 if not applicable)
    old_end_line   : End line in the old file (0 if not applicable)
    """
    qualified_name: str
    symbol_type:    SymbolType
    file_path:      str
    change_type:    ChangeType
    module_name:    str
    class_name:     Optional[str] = None
    start_line:     int = 0
    end_line:       int = 0
    old_start_line: int = 0
    old_end_line:   int = 0

    def __repr__(self) -> str:
        return (
            f"ChangedSymbol({self.symbol_type.value}:{self.qualified_name!r} "
            f"[{self.change_type.value}] @ {self.file_path}:{self.start_line})"
        )


@dataclass
class DependencyChange:
    """
    An explicit import-level change detected between old and new versions of a file.

    Example: ``from services.auth import validate_token`` →
             ``from services.auth import validate_user``
    would produce one IMPORT_REMOVED + one IMPORT_ADDED.

    Attributes
    ----------
    file_path   : File where the change occurred
    change_type : IMPORT_ADDED / IMPORT_REMOVED / IMPORT_MODIFIED
    dep_type    : ImportType.IMPORT or ImportType.FROM_IMPORT
    module      : The imported module (e.g. ``services.auth``)
    names       : The names imported (e.g. ``['validate_user']``)
    old_line    : Line number in old file (0 if added)
    new_line    : Line number in new file (0 if deleted)
    """
    file_path:   str
    change_type: DependencyChangeType
    dep_type:    ImportType
    module:      str
    names:       List[str]
    old_line:    int = 0
    new_line:    int = 0

    def __repr__(self) -> str:
        return (
            f"DependencyChange({self.change_type.value} "
            f"{self.module}.{self.names} @ {self.file_path})"
        )


# ============================================================================
# ChangeSet — top-level change summary
# ============================================================================

@dataclass
class ChangeSet:
    """
    Complete representation of all changes in a diff.

    Fields are populated by ChangeAnalyzer.  Computed properties derive
    aggregate counts so they stay consistent without manual bookkeeping.
    """
    # --- File-level ---
    files_added:    List[str] = field(default_factory=list)
    files_deleted:  List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_renamed:  List[Tuple[str, str]] = field(default_factory=list)  # (old, new)

    # Raw line counts
    lines_added:   int = 0
    lines_deleted: int = 0

    # --- Symbol-level ---
    changed_symbols: List[ChangedSymbol] = field(default_factory=list)

    # --- Dependency-level ---
    dependency_changes: List[DependencyChange] = field(default_factory=list)

    # --- Graph node IDs directly changed (matched against graph) ---
    changed_node_ids: Set[str] = field(default_factory=set)

    # --- Raw file diffs for downstream consumers ---
    file_diffs: List[FileDiff] = field(default_factory=list)

    # Files whose language was unrecognised or has no registered adapter.
    # They remain visible at file/line level but deliberately produce no
    # inferred symbols or dependency facts.
    unsupported_files: List[Tuple[str, str]] = field(default_factory=list)

    # ----------------------------------------------------------------
    # Computed properties — file level
    # ----------------------------------------------------------------

    @property
    def changed_files(self) -> List[str]:
        """All files touched (added + deleted + modified + renamed destinations)."""
        result = list(self.files_added) + list(self.files_deleted) + list(self.files_modified)
        result += [new for _, new in self.files_renamed]
        return result

    @property
    def total_lines_changed(self) -> int:
        return self.lines_added + self.lines_deleted

    @property
    def files_changed_count(self) -> int:
        return len(self.files_added) + len(self.files_deleted) + len(self.files_modified) + len(self.files_renamed)

    # ----------------------------------------------------------------
    # Computed properties — symbol level
    # ----------------------------------------------------------------

    def _symbols_of_type(self, stype: SymbolType) -> List[ChangedSymbol]:
        return [s for s in self.changed_symbols if s.symbol_type == stype]

    @property
    def changed_functions(self) -> List[ChangedSymbol]:
        return self._symbols_of_type(SymbolType.FUNCTION)

    @property
    def changed_classes(self) -> List[ChangedSymbol]:
        return self._symbols_of_type(SymbolType.CLASS)

    @property
    def changed_methods(self) -> List[ChangedSymbol]:
        return self._symbols_of_type(SymbolType.METHOD)

    @property
    def changed_modules(self) -> List[ChangedSymbol]:
        return self._symbols_of_type(SymbolType.MODULE)

    def _symbols_by_change(self, stype: SymbolType, ctype: ChangeType) -> List[ChangedSymbol]:
        return [s for s in self.changed_symbols
                if s.symbol_type == stype and s.change_type == ctype]

    @property
    def added_functions(self) -> List[ChangedSymbol]:
        return self._symbols_by_change(SymbolType.FUNCTION, ChangeType.ADDED)

    @property
    def deleted_functions(self) -> List[ChangedSymbol]:
        return self._symbols_by_change(SymbolType.FUNCTION, ChangeType.DELETED)

    @property
    def modified_functions(self) -> List[ChangedSymbol]:
        return self._symbols_by_change(SymbolType.FUNCTION, ChangeType.MODIFIED)

    @property
    def added_classes(self) -> List[ChangedSymbol]:
        return self._symbols_by_change(SymbolType.CLASS, ChangeType.ADDED)

    @property
    def deleted_classes(self) -> List[ChangedSymbol]:
        return self._symbols_by_change(SymbolType.CLASS, ChangeType.DELETED)

    @property
    def modified_classes(self) -> List[ChangedSymbol]:
        return self._symbols_by_change(SymbolType.CLASS, ChangeType.MODIFIED)

    # ----------------------------------------------------------------
    # Computed properties — dependency level
    # ----------------------------------------------------------------

    @property
    def dependencies_added(self) -> List[DependencyChange]:
        return [d for d in self.dependency_changes
                if d.change_type == DependencyChangeType.IMPORT_ADDED]

    @property
    def dependencies_removed(self) -> List[DependencyChange]:
        return [d for d in self.dependency_changes
                if d.change_type == DependencyChangeType.IMPORT_REMOVED]

    @property
    def dependencies_modified(self) -> List[DependencyChange]:
        return [d for d in self.dependency_changes
                if d.change_type == DependencyChangeType.IMPORT_MODIFIED]

    def __repr__(self) -> str:
        return (
            f"ChangeSet(files={self.files_changed_count}, "
            f"lines=+{self.lines_added}/-{self.lines_deleted}, "
            f"symbols={len(self.changed_symbols)}, "
            f"dep_changes={len(self.dependency_changes)})"
        )


# ============================================================================
# Impacted node structure
# ============================================================================

@dataclass
class ImpactedNode:
    """
    A graph node affected by the change, with impact metadata.

    Attributes
    ----------
    node_id      : Graph node identifier
    node_type    : NodeType (MODULE, CLASS, FUNCTION, METHOD)
    file_path    : Source file (may be None for synthetic nodes)
    impact_level : DIRECT / DIRECT_DEPENDENT / TRANSITIVE_DEPENDENT
    impact_depth : BFS hop distance from the nearest directly changed node
                   (0 = directly changed, 1 = direct dependent, ≥2 = transitive)
    """
    node_id:      str
    node_type:    NodeType
    file_path:    Optional[str]
    impact_level: ImpactLevel
    impact_depth: int

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImpactedNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __repr__(self) -> str:
        return (
            f"ImpactedNode({self.node_type.value}:{self.node_id!r} "
            f"depth={self.impact_depth} [{self.impact_level.value}])"
        )


@dataclass
class ImpactAnalysisResult:
    """
    Result of propagating a ChangeSet through the dependency graph.

    Directly changed nodes are depth-0.
    Direct dependents are depth-1.
    Transitive dependents are depth-2+.

    The ``features`` dict contains all numeric features for the risk model.
    """
    directly_changed_nodes:  List[ImpactedNode] = field(default_factory=list)
    impacted_nodes:          List[ImpactedNode] = field(default_factory=list)  # all except directly changed

    # Convenience breakdowns (subsets of impacted_nodes)
    impacted_modules:   List[ImpactedNode] = field(default_factory=list)
    impacted_classes:   List[ImpactedNode] = field(default_factory=list)
    impacted_functions: List[ImpactedNode] = field(default_factory=list)
    impacted_methods:   List[ImpactedNode] = field(default_factory=list)

    # Graph-level depth metrics
    impact_depth_map: Dict[str, int] = field(default_factory=dict)  # node_id → depth

    # Change-magnitude features (populated by ImpactAnalyzer.build_features)
    features: Dict[str, float] = field(default_factory=dict)

    # Source ChangeSet for reference
    change_set: Optional[ChangeSet] = None

    # ----------------------------------------------------------------
    # Computed aggregates
    # ----------------------------------------------------------------

    @property
    def affected_node_count(self) -> int:
        return len(self.directly_changed_nodes) + len(self.impacted_nodes)

    @property
    def affected_module_count(self) -> int:
        direct_modules = {n.node_id for n in self.directly_changed_nodes
                          if n.node_type == NodeType.MODULE}
        return len(direct_modules) + len(self.impacted_modules)

    @property
    def affected_function_count(self) -> int:
        direct_funcs = {n.node_id for n in self.directly_changed_nodes
                        if n.node_type == NodeType.FUNCTION}
        return len(direct_funcs) + len(self.impacted_functions)

    @property
    def affected_class_count(self) -> int:
        direct_cls = {n.node_id for n in self.directly_changed_nodes
                      if n.node_type == NodeType.CLASS}
        return len(direct_cls) + len(self.impacted_classes)

    @property
    def affected_method_count(self) -> int:
        direct_methods = {n.node_id for n in self.directly_changed_nodes
                          if n.node_type == NodeType.METHOD}
        return len(direct_methods) + len(self.impacted_methods)

    @property
    def maximum_impact_depth(self) -> int:
        if not self.impact_depth_map:
            return 0
        return max(self.impact_depth_map.values())

    @property
    def average_impact_depth(self) -> float:
        if not self.impact_depth_map:
            return 0.0
        depths = list(self.impact_depth_map.values())
        return sum(depths) / len(depths)

    def __repr__(self) -> str:
        return (
            f"ImpactAnalysisResult("
            f"direct={len(self.directly_changed_nodes)}, "
            f"impacted={len(self.impacted_nodes)}, "
            f"max_depth={self.maximum_impact_depth})"
        )


# ============================================================================
# Unified Diff Parser
# ============================================================================

# Regex for unified diff hunk header
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)

# Regex for file header lines
_DIFF_FILE_RE  = re.compile(r"^diff --git a/(.+) b/(.+)$")
_OLD_FILE_RE   = re.compile(r"^--- (.+)$")
_NEW_FILE_RE   = re.compile(r"^\+\+\+ (.+)$")
_RENAME_FROM   = re.compile(r"^rename from (.+)$")
_RENAME_TO     = re.compile(r"^rename to (.+)$")


class UnifiedDiffParser:
    """
    Parses a unified diff string into a list of FileDiff objects.

    Supports standard ``git diff`` output format including:
      - added files (new file mode)
      - deleted files (deleted file mode)
      - modified files
      - renames (rename from / rename to)
      - standard ``@@ -a,b +c,d @@`` hunk headers
    """

    def parse(self, diff_text: str) -> List[FileDiff]:
        """
        Parse a unified diff string.

        Args:
            diff_text: Raw unified diff text (e.g. from ``git diff``)

        Returns:
            List of FileDiff objects, one per changed file.
        """
        if not diff_text or not diff_text.strip():
            return []

        lines = diff_text.splitlines()
        file_diffs: List[FileDiff] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Detect start of a new file diff block
            m = _DIFF_FILE_RE.match(line)
            if m:
                a_path = m.group(1)
                b_path = m.group(2)
                i, file_diff = self._parse_file_diff(lines, i, a_path, b_path)
                if file_diff:
                    file_diffs.append(file_diff)
            else:
                i += 1

        return file_diffs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_file_diff(
        self,
        lines: List[str],
        start: int,
        a_path: str,
        b_path: str,
    ) -> Tuple[int, Optional[FileDiff]]:
        """
        Parse a single file's diff block starting at ``start``.

        Returns (new_index, FileDiff or None).
        """
        i = start + 1
        change_type = ChangeType.MODIFIED
        old_path = a_path
        new_path = b_path
        hunks: List[DiffHunk] = []

        # Read metadata lines until we hit the first hunk or the next diff header
        while i < len(lines):
            line = lines[i]

            if line.startswith("new file mode"):
                change_type = ChangeType.ADDED
                i += 1
            elif line.startswith("deleted file mode"):
                change_type = ChangeType.DELETED
                i += 1
            elif _RENAME_FROM.match(line):
                m = _RENAME_FROM.match(line)
                old_path = m.group(1)  # type: ignore[union-attr]
                change_type = ChangeType.RENAMED
                i += 1
            elif _RENAME_TO.match(line):
                m = _RENAME_TO.match(line)
                new_path = m.group(1)  # type: ignore[union-attr]
                i += 1
            elif line.startswith("--- "):
                # --- /dev/null means the file is new
                old_file_raw = line[4:].strip()
                if old_file_raw == "/dev/null":
                    change_type = ChangeType.ADDED
                i += 1
            elif line.startswith("+++ "):
                # +++ /dev/null means the file is deleted
                new_file_raw = line[4:].strip()
                if new_file_raw == "/dev/null":
                    change_type = ChangeType.DELETED
                i += 1
            elif _HUNK_HEADER_RE.match(line):
                # Parse all hunks
                i, file_hunks = self._parse_hunks(lines, i)
                hunks = file_hunks
                break
            elif _DIFF_FILE_RE.match(line):
                # Next file starts; stop here without consuming this line
                break
            else:
                i += 1

        # Determine the canonical file path for this diff
        if change_type == ChangeType.DELETED:
            canonical = self._strip_git_prefix(old_path)
        else:
            canonical = self._strip_git_prefix(new_path)

        old_canonical = self._strip_git_prefix(old_path)

        # Skip binary / submodule diffs that produced no hunks but aren't add/del
        if not hunks and change_type == ChangeType.MODIFIED:
            return i, None

        file_diff = FileDiff(
            file_path=canonical,
            old_path=old_canonical,
            change_type=change_type,
            hunks=hunks,
        )
        return i, file_diff

    def _parse_hunks(
        self, lines: List[str], start: int
    ) -> Tuple[int, List[DiffHunk]]:
        """Parse all consecutive hunks from ``start``."""
        hunks: List[DiffHunk] = []
        i = start

        while i < len(lines):
            line = lines[i]
            m = _HUNK_HEADER_RE.match(line)
            if not m:
                break

            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1

            hunk_lines: List[str] = []
            i += 1
            while i < len(lines):
                hunk_line = lines[i]
                if _HUNK_HEADER_RE.match(hunk_line) or _DIFF_FILE_RE.match(hunk_line):
                    break
                hunk_lines.append(hunk_line)
                i += 1

            hunks.append(DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=hunk_lines,
            ))

        return i, hunks

    @staticmethod
    def _strip_git_prefix(path: str) -> str:
        """Remove ``a/`` or ``b/`` git prefix from diff paths."""
        if path.startswith(("a/", "b/")):
            return path[2:]
        return path


# ============================================================================
# Symbol Mapper — maps changed line ranges → AST symbols
# ============================================================================

class SymbolMapper:
    """
    Maps changed line ranges to the functions, classes, and methods that
    contain them using the Phase 1 AST parser.

    Uses ``ASTParser.parse_source()`` so it works without access to the
    real filesystem — the caller supplies the source text.
    """

    def __init__(self) -> None:
        self._parser = ASTParser(include_docstrings=False, include_calls=False)

    def map_changed_lines(
        self,
        file_path: str,
        line_ranges: List[Tuple[int, int]],
        source: str,
        module_name: str,
        change_type: ChangeType,
    ) -> List[ChangedSymbol]:
        """
        Return the set of symbols that overlap with any of the given line ranges.

        Args:
            file_path   : File path (used for labelling; not read from disk)
            line_ranges : List of (start, end) line numbers (1-indexed, inclusive)
            source      : Complete source text of the new (or old) file version
            module_name : Dotted module name (e.g. "services.payment")
            change_type : The kind of change (ADDED/DELETED/MODIFIED)

        Returns:
            List of ChangedSymbol for every overlapping symbol.
        """
        if not source or not line_ranges:
            return []

        parsed = self._parser.parse_source(source, file_path)
        if parsed.syntax_errors:
            return []

        symbols: List[ChangedSymbol] = []
        seen_ids: Set[str] = set()

        for line_start, line_end in line_ranges:
            # Classes
            for cls in parsed.classes:
                if cls.location and self._overlaps(
                    cls.location.start_line, cls.location.end_line,
                    line_start, line_end,
                ):
                    qname = f"{module_name}.{cls.name}"
                    if qname not in seen_ids:
                        seen_ids.add(qname)
                        symbols.append(ChangedSymbol(
                            qualified_name=qname,
                            symbol_type=SymbolType.CLASS,
                            file_path=file_path,
                            change_type=change_type,
                            module_name=module_name,
                            start_line=cls.location.start_line,
                            end_line=cls.location.end_line,
                        ))

                # Methods
                for method in cls.methods:
                    if method.location and self._overlaps(
                        method.location.start_line, method.location.end_line,
                        line_start, line_end,
                    ):
                        qname = f"{module_name}.{cls.name}.{method.name}"
                        if qname not in seen_ids:
                            seen_ids.add(qname)
                            symbols.append(ChangedSymbol(
                                qualified_name=qname,
                                symbol_type=SymbolType.METHOD,
                                file_path=file_path,
                                change_type=change_type,
                                module_name=module_name,
                                class_name=cls.name,
                                start_line=method.location.start_line,
                                end_line=method.location.end_line,
                            ))

            # Standalone functions
            for func in parsed.functions:
                if func.location and self._overlaps(
                    func.location.start_line, func.location.end_line,
                    line_start, line_end,
                ):
                    qname = f"{module_name}.{func.name}"
                    if qname not in seen_ids:
                        seen_ids.add(qname)
                        symbols.append(ChangedSymbol(
                            qualified_name=qname,
                            symbol_type=SymbolType.FUNCTION,
                            file_path=file_path,
                            change_type=change_type,
                            module_name=module_name,
                            start_line=func.location.start_line,
                            end_line=func.location.end_line,
                        ))

        return symbols

    @staticmethod
    def _overlaps(
        sym_start: int, sym_end: int, range_start: int, range_end: int
    ) -> bool:
        """Return True if [sym_start, sym_end] overlaps [range_start, range_end]."""
        return sym_start <= range_end and sym_end >= range_start


# ============================================================================
# Dependency Change Detector
# ============================================================================

class DependencyChangeDetector:
    """
    Detects import-level changes between old and new source versions of a file.

    Uses the Phase 1 ASTParser to extract import lists from both versions
    and computes the symmetric difference.
    """

    def __init__(self) -> None:
        self._parser = ASTParser(include_docstrings=False, include_calls=False)

    def detect(
        self,
        file_path: str,
        old_source: Optional[str],
        new_source: Optional[str],
    ) -> List[DependencyChange]:
        """
        Compare imports in old and new source; return explicit change records.

        Args:
            file_path  : File path for labelling.
            old_source : Source text before the change (None for added files).
            new_source : Source text after the change (None for deleted files).

        Returns:
            List of DependencyChange records.
        """
        old_imports = self._parse_imports(old_source, file_path) if old_source else {}
        new_imports = self._parse_imports(new_source, file_path) if new_source else {}

        changes: List[DependencyChange] = []

        # Keys are (module, frozenset(names), ImportType)
        old_keys = set(old_imports.keys())
        new_keys = set(new_imports.keys())

        added_keys   = new_keys - old_keys
        removed_keys = old_keys - new_keys

        for key in added_keys:
            imp_info = new_imports[key]
            changes.append(DependencyChange(
                file_path=file_path,
                change_type=DependencyChangeType.IMPORT_ADDED,
                dep_type=imp_info.import_type,
                module=imp_info.module,
                names=list(imp_info.names),
                new_line=imp_info.location.start_line if imp_info.location else 0,
            ))

        for key in removed_keys:
            imp_info = old_imports[key]
            changes.append(DependencyChange(
                file_path=file_path,
                change_type=DependencyChangeType.IMPORT_REMOVED,
                dep_type=imp_info.import_type,
                module=imp_info.module,
                names=list(imp_info.names),
                old_line=imp_info.location.start_line if imp_info.location else 0,
            ))

        return changes

    def _parse_imports(
        self, source: str, file_path: str
    ) -> Dict[Tuple[str, frozenset[str], ImportType], ImportInfo]:
        """
        Parse imports from source, returning a dict keyed by
        (module, frozenset(names), import_type).
        """
        parsed = self._parser.parse_source(source, file_path)
        result: Dict[Tuple[str, frozenset[str], ImportType], ImportInfo] = {}
        for imp in parsed.imports:
            key = (imp.module, frozenset(imp.names), imp.import_type)
            result[key] = imp
        return result


# ============================================================================
# Language-Aware Symbol Mapper (Phase 7B)
# ============================================================================

class LanguageAwareSymbolMapper:
    """
    Maps changed line ranges to symbols using a LanguageAdapter.

    This is the IR-first replacement for SymbolMapper. It works with
    ParsedSource objects (which contain language-neutral Symbol objects)
    instead of directly using the Python AST parser.

    Usage:
        adapter = default_registry.get("python")
        parsed_source = adapter.parse(source, file_path)
        mapper = LanguageAwareSymbolMapper(adapter)
        symbols = mapper.map_changed_lines(...)
    """

    def __init__(self, adapter: LanguageAdapter) -> None:
        self._adapter = adapter

    def map_changed_lines(
        self,
        file_path: str,
        line_ranges: List[Tuple[int, int]],
        source: str,
        module_name: str,
        change_type: ChangeType,
    ) -> List[ChangedSymbol]:
        """
        Return the set of symbols that overlap with any of the given line ranges.

        This uses the adapter's parse() to get a ParsedSource with neutral symbols.
        """
        if not source or not line_ranges:
            return []

        parsed = self._adapter.parse(source, file_path)
        if parsed.has_errors:
            return []

        symbols: List[ChangedSymbol] = []
        seen_ids: Set[str] = set()

        for line_start, line_end in line_ranges:
            # Classes
            for sym in parsed.class_symbols:
                if sym.start_line and sym.end_line and self._overlaps(
                    sym.start_line, sym.end_line, line_start, line_end,
                ):
                    if sym.qualified_name not in seen_ids:
                        seen_ids.add(sym.qualified_name)
                        symbols.append(ChangedSymbol(
                            qualified_name=sym.qualified_name,
                            symbol_type=self._ir_symbol_kind_to_type(sym.symbol_kind),
                            file_path=file_path,
                            change_type=change_type,
                            module_name=module_name,
                            start_line=sym.start_line,
                            end_line=sym.end_line,
                        ))

            # Methods
            for sym in parsed.method_symbols:
                if sym.start_line and sym.end_line and self._overlaps(
                    sym.start_line, sym.end_line, line_start, line_end,
                ):
                    if sym.qualified_name not in seen_ids:
                        seen_ids.add(sym.qualified_name)
                        symbols.append(ChangedSymbol(
                            qualified_name=sym.qualified_name,
                            symbol_type=self._ir_symbol_kind_to_type(sym.symbol_kind),
                            file_path=file_path,
                            change_type=change_type,
                            module_name=module_name,
                            class_name=sym.parent,
                            start_line=sym.start_line,
                            end_line=sym.end_line,
                        ))

            # Standalone functions
            for sym in parsed.function_symbols:
                if sym.start_line and sym.end_line and self._overlaps(
                    sym.start_line, sym.end_line, line_start, line_end,
                ):
                    if sym.qualified_name not in seen_ids:
                        seen_ids.add(sym.qualified_name)
                        symbols.append(ChangedSymbol(
                            qualified_name=sym.qualified_name,
                            symbol_type=self._ir_symbol_kind_to_type(sym.symbol_kind),
                            file_path=file_path,
                            change_type=change_type,
                            module_name=module_name,
                            start_line=sym.start_line,
                            end_line=sym.end_line,
                        ))

        return symbols

    @staticmethod
    def _ir_symbol_kind_to_type(kind: SymbolKind) -> SymbolType:
        """Map IR SymbolKind to ChangeAnalyzer SymbolType."""
        mapping = {
            SymbolKind.MODULE: SymbolType.MODULE,
            SymbolKind.CLASS: SymbolType.CLASS,
            SymbolKind.FUNCTION: SymbolType.FUNCTION,
            SymbolKind.METHOD: SymbolType.METHOD,
            SymbolKind.VARIABLE: SymbolType.FUNCTION,  # fallback
        }
        return mapping.get(kind, SymbolType.FUNCTION)

    @staticmethod
    def _overlaps(
        sym_start: int, sym_end: int, range_start: int, range_end: int
    ) -> bool:
        """Return True if [sym_start, sym_end] overlaps [range_start, range_end]."""
        return sym_start <= range_end and sym_end >= range_start


# ============================================================================
# Language-Aware Dependency Change Detector (Phase 7B)
# ============================================================================

class LanguageAwareDependencyChangeDetector:
    """
    Detects import-level changes between old and new source versions
    using a LanguageAdapter.

    This is the IR-first replacement for DependencyChangeDetector.
    It works with ParsedSource objects instead of directly using the
    Python AST parser.
    """

    def __init__(self, adapter: LanguageAdapter) -> None:
        self._adapter = adapter

    def detect(
        self,
        file_path: str,
        old_source: Optional[str],
        new_source: Optional[str],
    ) -> List[DependencyChange]:
        """
        Compare imports in old and new source; return explicit change records.

        Uses the adapter's parse() and extract_dependencies() to get
        language-neutral Dependency objects.
        """
        old_deps = self._extract_dependencies(old_source, file_path) if old_source else []
        new_deps = self._extract_dependencies(new_source, file_path) if new_source else []

        changes: List[DependencyChange] = []

        # Convert to comparable keys: (dep_type, target, source)
        # Note: We use source to distinguish imports from different modules
        old_keys = {
            (d.dep_type, d.target, d.source)
            for d in old_deps
        }
        new_keys = {
            (d.dep_type, d.target, d.source)
            for d in new_deps
        }

        added_keys = new_keys - old_keys
        removed_keys = old_keys - new_keys

        # Create a lookup for detailed info
        old_by_key = {(d.dep_type, d.target, d.source): d for d in old_deps}
        new_by_key = {(d.dep_type, d.target, d.source): d for d in new_deps}

        for key in added_keys:
            dep = new_by_key[key]
            changes.append(DependencyChange(
                file_path=file_path,
                change_type=DependencyChangeType.IMPORT_ADDED,
                dep_type=self._ir_dep_type_to_import_type(dep.dep_type),
                module=self._extract_module_from_target(dep.target),
                names=[self._extract_name_from_target(dep.target)],
                new_line=dep.source_line,
            ))

        for key in removed_keys:
            dep = old_by_key[key]
            changes.append(DependencyChange(
                file_path=file_path,
                change_type=DependencyChangeType.IMPORT_REMOVED,
                dep_type=self._ir_dep_type_to_import_type(dep.dep_type),
                module=self._extract_module_from_target(dep.target),
                names=[self._extract_name_from_target(dep.target)],
                old_line=dep.source_line,
            ))

        return changes

    def _extract_dependencies(
        self, source: str, file_path: str
    ) -> List[Dependency]:
        """Parse source and extract dependencies using the adapter."""
        if not source:
            return []
        parsed = self._adapter.parse(source, file_path)
        if parsed.has_errors:
            return []
        return self._adapter.extract_dependencies(parsed)

    @staticmethod
    def _ir_dep_type_to_import_type(dep_type: LanguageDependencyType) -> ImportType:
        """Map IR LanguageDependencyType to ChangeAnalyzer ImportType."""
        if dep_type == LanguageDependencyType.IMPORTS:
            return ImportType.FROM_IMPORT  # closest match
        return ImportType.IMPORT

    @staticmethod
    def _extract_module_from_target(target: str) -> str:
        """Extract module part from a qualified target name."""
        parts = target.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:-1])
        return target

    @staticmethod
    def _extract_name_from_target(target: str) -> str:
        """Extract name part from a qualified target name."""
        return target.split(".")[-1]


# ============================================================================
# Source Provider Interface
# ============================================================================

# A source provider maps a file path to its old/new source text.
# ``None`` means the file does not exist on that side (added or deleted).
SourceProvider = Callable[[str, str], Optional[str]]
"""
Signature: ``provider(file_path: str, version: str) -> Optional[str]``
where ``version`` is either ``"old"`` or ``"new"``.
"""


# ============================================================================
# Change Analyzer — main orchestrator
# ============================================================================

class ChangeAnalyzer:
    """
    Orchestrates the analysis of a unified diff into a ChangeSet.

    Usage (with real files)::

        analyzer = ChangeAnalyzer(source_provider=my_provider)
        change_set = analyzer.analyze_diff(diff_text)

    Usage (self-contained from strings)::

        change_set = ChangeAnalyzer.analyze_diff_simple(diff_text)

    Usage (IR-first with language adapters)::

        analyzer = ChangeAnalyzer(source_provider=my_provider,
                                  language_registry=default_registry)
        change_set = analyzer.analyze_diff(diff_text)

    The ``source_provider`` is called with ``(file_path, version)`` where
    ``version`` is ``"old"`` or ``"new"``.  When a provider is not supplied
    (or returns None), symbol-level analysis is skipped for that file but
    file/line-level information is still extracted.

    When ``language_registry`` is provided, the analyzer will use
    language-specific adapters for symbol mapping and dependency detection,
    enabling multi-language support.
    """

    def __init__(
        self,
        source_provider: Optional[SourceProvider] = None,
        module_name_resolver: Optional[Callable[[str], str]] = None,
        language_registry: Optional[LanguageRegistry] = None,
    ) -> None:
        """
        Args:
            source_provider      : Callable returning source text for a file/version.
            module_name_resolver : Maps file_path → dotted module name.
                                   Defaults to stripping ".py" and replacing "/" with ".".
            language_registry    : Optional LanguageRegistry for IR-first analysis.
                                   When provided, enables multi-language symbol
                                   mapping and dependency detection.
        """
        self._source_provider       = source_provider
        self._module_name_resolver  = module_name_resolver or self._default_module_name
        self._diff_parser           = UnifiedDiffParser()
        self._language_registry     = language_registry
        self._symbol_mapper         = SymbolMapper()  # legacy fallback
        self._dep_detector          = DependencyChangeDetector()  # legacy fallback

        # Language-aware mappers (created on demand per language)
        self._lang_symbol_mappers: Dict[str, LanguageAwareSymbolMapper] = {}
        self._lang_dep_detectors: Dict[str, LanguageAwareDependencyChangeDetector] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_diff(self, diff_text: str) -> ChangeSet:
        """
        Analyse a unified diff string and return a ChangeSet.

        This works without a real Git repository; pass in the raw diff text.
        Source text is fetched via the ``source_provider`` if one was supplied.
        When no provider is available, symbol-level analysis is skipped for
        that file but file/line-level information is still extracted.

        Args:
            diff_text: Raw unified diff (e.g. from ``git diff HEAD~1``)

        Returns:
            ChangeSet with all extracted change information.
        """
        file_diffs = self._diff_parser.parse(diff_text)

        cs = ChangeSet()
        cs.file_diffs = file_diffs

        for fd in file_diffs:
            self._process_file_diff(fd, cs)

        return cs

    # ------------------------------------------------------------------
    # Convenience static API
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_diff_simple(diff_text: str) -> ChangeSet:
        """
        Analyse a diff without a source provider.

        File / line level information is always available.
        Symbol and dependency analysis requires source text and is skipped.
        """
        return ChangeAnalyzer().analyze_diff(diff_text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_file_diff(self, fd: FileDiff, cs: ChangeSet) -> None:
        """Update a ChangeSet with data from one FileDiff.

        When a language_registry is available, uses language-aware IR-first
        components; otherwise falls back to legacy Python-specific analysis.
        """
        # --- File classification ---
        if fd.change_type == ChangeType.ADDED:
            cs.files_added.append(fd.file_path)
        elif fd.change_type == ChangeType.DELETED:
            cs.files_deleted.append(fd.file_path)
        elif fd.change_type == ChangeType.RENAMED:
            cs.files_renamed.append((fd.old_path, fd.file_path))
        else:
            cs.files_modified.append(fd.file_path)

        # --- Line counts ---
        cs.lines_added   += fd.lines_added
        cs.lines_deleted += fd.lines_deleted

        # --- Symbol mapping (requires source text) ---
        new_source = self._get_source(fd.file_path, "new")
        old_source = self._get_source(fd.old_path, "old") if fd.old_path != fd.file_path else self._get_source(fd.file_path, "old")

        module_name = self._module_name_resolver(fd.file_path)

        # Detect the file language even when no registry was supplied.  The
        # legacy parser is Python-only and must never receive another language.
        symbol_mapper: SymbolMapper | LanguageAwareSymbolMapper = self._symbol_mapper
        dep_detector: DependencyChangeDetector | LanguageAwareDependencyChangeDetector = self._dep_detector
        adapter = None
        from analyzer.languages.detector import LanguageDetector
        detector = LanguageDetector()
        detected = detector.detect(fd.file_path)
        language = detected.language
        registry = self._language_registry
        if registry is not None:
            adapter = registry.get(language)
            if adapter is not None:
                # Get or create language-aware mapper.
                if language not in self._lang_symbol_mappers:
                    self._lang_symbol_mappers[language] = LanguageAwareSymbolMapper(adapter)
                symbol_mapper = self._lang_symbol_mappers[language]

                # Get or create language-aware dependency detector.
                if language not in self._lang_dep_detectors:
                    self._lang_dep_detectors[language] = LanguageAwareDependencyChangeDetector(adapter)
                dep_detector = self._lang_dep_detectors[language]
            elif new_source or old_source:
                cs.unsupported_files.append((fd.file_path, language))
        elif language != "python":
            # No registry is an explicit legacy-Python mode, not permission to
            # parse arbitrary source with ast.parse().
            if new_source or old_source:
                cs.unsupported_files.append((fd.file_path, language))
        # The old compatibility path is valid only for Python.
        legacy_python = registry is None and language == "python"
        if adapter is None and not legacy_python and (fd.file_path, language) not in cs.unsupported_files:
            cs.unsupported_files.append((fd.file_path, language))

        if fd.change_type == ChangeType.ADDED:
            # Whole file is new; mark the module and all its symbols as added
            if new_source and (adapter is not None or legacy_python):
                symbols = self._all_symbols_from_source(
                    fd.file_path, new_source, module_name, ChangeType.ADDED, adapter
                )
                cs.changed_symbols.extend(symbols)
            elif not new_source and (adapter is not None or legacy_python):
                # No source available — record module-level change
                cs.changed_symbols.append(ChangedSymbol(
                    qualified_name=module_name,
                    symbol_type=SymbolType.MODULE,
                    file_path=fd.file_path,
                    change_type=ChangeType.ADDED,
                    module_name=module_name,
                ))

        elif fd.change_type == ChangeType.DELETED:
            if old_source and (adapter is not None or legacy_python):
                # For deleted files, use old_source to detect language
                delete_adapter = adapter
                if delete_adapter is None and self._language_registry is not None:
                    delete_adapter = self._language_registry.get(language)
                symbols = self._all_symbols_from_source(
                    fd.file_path, old_source, module_name, ChangeType.DELETED, delete_adapter
                )
                cs.changed_symbols.extend(symbols)
            elif not old_source and (adapter is not None or legacy_python):
                cs.changed_symbols.append(ChangedSymbol(
                    qualified_name=module_name,
                    symbol_type=SymbolType.MODULE,
                    file_path=fd.file_path,
                    change_type=ChangeType.DELETED,
                    module_name=module_name,
                ))

        else:
            # MODIFIED or RENAMED — map specific changed lines to symbols
            if new_source and fd.affected_new_ranges and (adapter is not None or legacy_python):
                symbols = symbol_mapper.map_changed_lines(
                    fd.file_path,
                    fd.affected_new_ranges,
                    new_source,
                    module_name,
                    ChangeType.MODIFIED,
                )
                cs.changed_symbols.extend(symbols)

        # --- Dependency change detection ---
        if adapter is not None or legacy_python:
            dep_changes = dep_detector.detect(fd.file_path, old_source, new_source)
            cs.dependency_changes.extend(dep_changes)

    def _get_source(self, file_path: str, version: str) -> Optional[str]:
        """Fetch source text via the provider, or return None."""
        if self._source_provider is None:
            return None
        # Provider failures are analysis failures, not evidence that the file
        # did not exist. Callers need to distinguish an absent snapshot from a
        # checkout/read error instead of silently degrading to file-only data.
        return self._source_provider(file_path, version)

    def _all_symbols_from_source(
        self,
        file_path: str,
        source: str,
        module_name: str,
        change_type: ChangeType,
        adapter: Optional[LanguageAdapter] = None,
    ) -> List[ChangedSymbol]:
        """Extract ALL symbols from source and mark them with ``change_type``.

        If ``adapter`` is provided, uses the language-neutral IR via the adapter.
        Otherwise falls back to the Python-specific AST parser.
        """
        if adapter is not None:
            return self._all_symbols_from_source_lang(
                file_path, source, module_name, change_type, adapter
            )

        parser = ASTParser(include_docstrings=False, include_calls=False)
        parsed = parser.parse_source(source, file_path)
        symbols: List[ChangedSymbol] = []

        for cls in parsed.classes:
            qname = f"{module_name}.{cls.name}"
            symbols.append(ChangedSymbol(
                qualified_name=qname,
                symbol_type=SymbolType.CLASS,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
                start_line=cls.location.start_line if cls.location else 0,
                end_line=cls.location.end_line if cls.location else 0,
            ))
            for method in cls.methods:
                symbols.append(ChangedSymbol(
                    qualified_name=f"{qname}.{method.name}",
                    symbol_type=SymbolType.METHOD,
                    file_path=file_path,
                    change_type=change_type,
                    module_name=module_name,
                    class_name=cls.name,
                    start_line=method.location.start_line if method.location else 0,
                    end_line=method.location.end_line if method.location else 0,
                ))

        for func in parsed.functions:
            symbols.append(ChangedSymbol(
                qualified_name=f"{module_name}.{func.name}",
                symbol_type=SymbolType.FUNCTION,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
                start_line=func.location.start_line if func.location else 0,
                end_line=func.location.end_line if func.location else 0,
            ))

        # If nothing found, record module-level change
        if not symbols:
            symbols.append(ChangedSymbol(
                qualified_name=module_name,
                symbol_type=SymbolType.MODULE,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
            ))

        return symbols

    def _all_symbols_from_source_lang(
        self,
        file_path: str,
        source: str,
        module_name: str,
        change_type: ChangeType,
        adapter: LanguageAdapter,
    ) -> List[ChangedSymbol]:
        """Extract ALL symbols using a LanguageAdapter (IR-first path)."""
        parsed = adapter.parse(source, file_path)
        if parsed.has_errors:
            # Fall back to module-level change
            return [ChangedSymbol(
                qualified_name=module_name,
                symbol_type=SymbolType.MODULE,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
            )]

        symbols: List[ChangedSymbol] = []

        for sym in parsed.class_symbols:
            symbols.append(ChangedSymbol(
                qualified_name=sym.qualified_name,
                symbol_type=SymbolType.CLASS,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
                start_line=sym.start_line or 0,
                end_line=sym.end_line or 0,
            ))

        for sym in parsed.method_symbols:
            symbols.append(ChangedSymbol(
                qualified_name=sym.qualified_name,
                symbol_type=SymbolType.METHOD,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
                class_name=sym.parent,
                start_line=sym.start_line or 0,
                end_line=sym.end_line or 0,
            ))

        for sym in parsed.function_symbols:
            symbols.append(ChangedSymbol(
                qualified_name=sym.qualified_name,
                symbol_type=SymbolType.FUNCTION,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
                start_line=sym.start_line or 0,
                end_line=sym.end_line or 0,
            ))

        if not symbols:
            symbols.append(ChangedSymbol(
                qualified_name=module_name,
                symbol_type=SymbolType.MODULE,
                file_path=file_path,
                change_type=change_type,
                module_name=module_name,
            ))

        return symbols

    @staticmethod
    def _default_module_name(file_path: str) -> str:
        """
        Derive a dotted module name from a file path.

        ``services/payment.py`` → ``services.payment``
        ``services/payment/__init__.py`` → ``services.payment``
        """
        p = Path(file_path)
        parts = list(p.parts)
        if parts and parts[-1].endswith(".py"):
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
        return ".".join(parts) if parts else file_path


# ============================================================================
# Impact Analyzer — graph traversal
# ============================================================================

class ImpactAnalyzer:
    """
    Propagates a ChangeSet through a CustomDependencyGraph and produces an
    ImpactAnalysisResult.

    This is the central component for change-impact analysis.

    Algorithm
    ---------
    1. For every changed symbol in the ChangeSet, locate the corresponding
       graph node(s) using name-matching heuristics.
    2. Mark those nodes as DIRECT (depth=0).
    3. Use ``get_transitive_dependents()`` (BFS reverse traversal already
       implemented in Phase 4) to discover all transitively impacted nodes.
    4. Track impact depth for each newly discovered node as BFS progresses.
    5. Classify nodes as DIRECT_DEPENDENT (depth=1) or TRANSITIVE (depth≥2).
    6. Compute the feature dict.
    """

    def __init__(self, graph: CustomDependencyGraph) -> None:
        """
        Args:
            graph: A populated CustomDependencyGraph (from Phase 4).
        """
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_impact(self, change_set: ChangeSet) -> ImpactAnalysisResult:
        """
        Analyse impact from a ChangeSet through the dependency graph.

        Args:
            change_set: Output of ChangeAnalyzer.analyze_diff()

        Returns:
            ImpactAnalysisResult with all impacted nodes and feature values.
        """
        result = ImpactAnalysisResult(change_set=change_set)

        # Step 1: Find directly changed graph nodes
        directly_changed = self._find_changed_nodes(change_set)

        # Update ChangeSet with found node IDs
        change_set.changed_node_ids = {n.node_id for n in directly_changed}

        for node in directly_changed:
            impacted = ImpactedNode(
                node_id=node.node_id,
                node_type=node.node_type,
                file_path=node.file_path,
                impact_level=ImpactLevel.DIRECT,
                impact_depth=0,
            )
            result.directly_changed_nodes.append(impacted)
            result.impact_depth_map[node.node_id] = 0

        # Step 2: BFS with depth tracking from all directly changed nodes
        all_impacted_ids: Dict[str, int] = {}  # node_id → minimum depth

        for node in directly_changed:
            depths = self._bfs_with_depth(node.node_id)
            for nid, depth in depths.items():
                if nid not in all_impacted_ids or depth < all_impacted_ids[nid]:
                    all_impacted_ids[nid] = depth

        # Remove directly changed nodes from impacted (they're in their own list)
        directly_changed_ids = {n.node_id for n in directly_changed}
        for nid, depth in all_impacted_ids.items():
            if nid in directly_changed_ids:
                continue  # already in directly_changed

            graph_node = self._graph.get_node(nid)
            if graph_node is None:
                continue

            level = ImpactLevel.DIRECT_DEPENDENT if depth == 1 else ImpactLevel.TRANSITIVE_DEPENDENT
            impacted_node = ImpactedNode(
                node_id=nid,
                node_type=graph_node.node_type,
                file_path=graph_node.file_path,
                impact_level=level,
                impact_depth=depth,
            )
            result.impacted_nodes.append(impacted_node)
            result.impact_depth_map[nid] = depth

        # Step 3: Populate type-specific lists
        for inode in result.impacted_nodes:
            if inode.node_type == NodeType.MODULE:
                result.impacted_modules.append(inode)
            elif inode.node_type == NodeType.CLASS:
                result.impacted_classes.append(inode)
            elif inode.node_type == NodeType.FUNCTION:
                result.impacted_functions.append(inode)
            elif inode.node_type == NodeType.METHOD:
                result.impacted_methods.append(inode)

        # Step 4: Build feature dict
        result.features = self._build_features(change_set, result)

        return result

    # ------------------------------------------------------------------
    # Node matching
    # ------------------------------------------------------------------

    def _find_changed_nodes(self, change_set: ChangeSet) -> List[GraphNode]:
        """
        Map changed symbols to graph nodes using a multi-strategy lookup.

        Strategies (applied in order):
        1. Exact qualified_name match          e.g. "services.payment.process"
        2. File-path-based module node match   fall back to module node
        3. All nodes in the changed file       if file-level-only change
        """
        found: Dict[str, GraphNode] = {}  # node_id → node

        for symbol in change_set.changed_symbols:
            node = self._graph.get_node(symbol.qualified_name)
            if node and node.node_id not in found:
                found[node.node_id] = node
                continue

            # Try without leading path segments (partial match fallback)
            parts = symbol.qualified_name.split(".")
            for length in range(len(parts), 0, -1):
                candidate = ".".join(parts[-length:])
                node = self._graph.get_node(candidate)
                if node and node.node_id not in found:
                    found[node.node_id] = node
                    break

        # Also include module nodes for changed files that had no matched symbol
        all_matched_files: Set[str] = {n.file_path for n in found.values() if n.file_path}
        for file_path in change_set.changed_files:
            if file_path not in all_matched_files:
                for node in self._graph.nodes_in_file(file_path):
                    if node.node_type == NodeType.MODULE and node.node_id not in found:
                        found[node.node_id] = node

        return list(found.values())

    # ------------------------------------------------------------------
    # BFS with depth tracking
    # ------------------------------------------------------------------

    def _bfs_with_depth(self, start_node_id: str) -> Dict[str, int]:
        """
        BFS in reverse direction (who depends on ``start_node_id``?).

        Returns a dict mapping node_id → depth (minimum hops from start).
        The start node itself is NOT included.
        """
        result: Dict[str, int] = {}
        visited: Set[str] = {start_node_id}
        queue: deque[Tuple[str, int]] = deque([(start_node_id, 0)])

        while queue:
            current, depth = queue.popleft()
            dependents = self._graph.get_dependents(current)
            for dep_node in dependents:
                nid = dep_node.node_id
                if nid not in visited:
                    visited.add(nid)
                    new_depth = depth + 1
                    result[nid] = new_depth
                    queue.append((nid, new_depth))

        return result

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _build_features(
        self,
        cs: ChangeSet,
        result: ImpactAnalysisResult,
    ) -> Dict[str, float]:
        """
        Build the quantitative feature dict for the risk model.

        Each feature has a clear semantic definition:

        files_changed           - total files touched (scope of change)
        lines_added             - new lines introduced
        lines_deleted           - lines removed
        total_lines_changed     - churn volume
        functions_changed       - count of changed standalone functions
        classes_changed         - count of changed classes
        methods_changed         - count of changed methods
        dependencies_added      - new imports (integration expansion risk)
        dependencies_removed    - removed imports (dead-code or breaking change risk)
        dependencies_changed    - total dep churn (added + removed)
        direct_nodes_changed    - graph nodes directly touched
        impacted_nodes          - blast radius size (total impacted)
        impacted_modules        - module-level blast radius
        impacted_functions      - function-level blast radius
        impacted_classes        - class-level blast radius
        impacted_methods        - method-level blast radius
        maximum_impact_depth    - how far the change propagates (coupling depth)
        impact_ratio            - impacted / total_graph_nodes (normalised)
        dependency_change_ratio - dep_changes / (direct_nodes + 1)  (import churn)
        average_impact_depth    - average BFS depth across all impacted nodes
        """
        total_graph_nodes = max(len(self._graph.nodes), 1)
        deps_added   = len(cs.dependencies_added)
        deps_removed = len(cs.dependencies_removed)
        deps_total   = deps_added + deps_removed
        impacted_total = len(result.impacted_nodes)
        direct_total   = len(result.directly_changed_nodes)

        return {
            "files_changed":           float(cs.files_changed_count),
            "lines_added":             float(cs.lines_added),
            "lines_deleted":           float(cs.lines_deleted),
            "total_lines_changed":     float(cs.total_lines_changed),
            "functions_changed":       float(len(cs.changed_functions)),
            "classes_changed":         float(len(cs.changed_classes)),
            "methods_changed":         float(len(cs.changed_methods)),
            "dependencies_added":      float(deps_added),
            "dependencies_removed":    float(deps_removed),
            "dependencies_changed":    float(deps_total),
            "dependency_change_ratio": deps_total / max(direct_total + 1, 1),
            "direct_nodes_changed":    float(direct_total),
            "impacted_nodes":          float(impacted_total),
            "impacted_modules":        float(len(result.impacted_modules)),
            "impacted_functions":      float(len(result.impacted_functions)),
            "impacted_classes":        float(len(result.impacted_classes)),
            "impacted_methods":        float(len(result.impacted_methods)),
            "maximum_impact_depth":    float(result.maximum_impact_depth),
            # impact_ratio: normalised blast radius — useful because raw counts
            # are graph-size-dependent; the ratio is stable across projects.
            "impact_ratio":            impacted_total / total_graph_nodes,
            # average_impact_depth: measures how deeply the change propagates
            # on average; high values indicate tight coupling or long dependency chains.
            "average_impact_depth":    result.average_impact_depth,
        }


# ============================================================================
# Convenience functions
# ============================================================================

def analyze_diff(
    diff_text: str,
    source_provider: Optional[SourceProvider] = None,
) -> ChangeSet:
    """
    Convenience function: parse a unified diff into a ChangeSet.

    Args:
        diff_text       : Raw unified diff text.
        source_provider : Optional callable ``(file_path, version) → Optional[str]``.

    Returns:
        ChangeSet
    """
    analyzer = ChangeAnalyzer(source_provider=source_provider)
    return analyzer.analyze_diff(diff_text)


def analyze_impact(
    change_set: ChangeSet,
    graph: CustomDependencyGraph,
) -> ImpactAnalysisResult:
    """
    Convenience function: propagate a ChangeSet through a dependency graph.

    Args:
        change_set : Output of ``analyze_diff()``.
        graph      : A populated CustomDependencyGraph.

    Returns:
        ImpactAnalysisResult
    """
    impact_analyzer = ImpactAnalyzer(graph)
    return impact_analyzer.analyze_impact(change_set)


def analyze_diff_and_impact(
    diff_text: str,
    graph: CustomDependencyGraph,
    source_provider: Optional[SourceProvider] = None,
) -> Tuple[ChangeSet, ImpactAnalysisResult]:
    """
    End-to-end convenience: diff → ChangeSet → ImpactAnalysisResult.

    Args:
        diff_text       : Raw unified diff text.
        graph           : Populated CustomDependencyGraph.
        source_provider : Optional source text provider.

    Returns:
        (ChangeSet, ImpactAnalysisResult)
    """
    cs = analyze_diff(diff_text, source_provider)
    ir = analyze_impact(cs, graph)
    return cs, ir
