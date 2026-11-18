"""Changed-file-first repository analysis.

The index deliberately reads only cheap lexical metadata for the repository.
AST/IR parsing is reserved for the changed files and the dependency closure.
This keeps the graph and feature calculation identical while removing the
old repository-size rejection from the normal path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

SUPPORTED_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".cs", ".rs", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".java"}
IGNORED_DIRS = {".git", "node_modules", "vendor", "third_party", "target", "dist", "build", "out", ".venv", "venv"}
_DEF = re.compile(r"\b(?:def|class|function|fn|struct|interface|trait|enum)\s+([A-Za-z_$][\w$]*)")
_IMPORT = re.compile(r"(?:^|[;\n])\s*(?:import\s+([^;\n]+)|from\s+([^\s;]+)\s+import|using\s+([^;\n]+))", re.MULTILINE)
_TOKEN = re.compile(r"[A-Za-z_$][\w$]*")


@dataclass(frozen=True)
class IndexEntry:
    path: str
    module: str
    language: str
    symbols: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    tokens: frozenset[str] = frozenset()


@dataclass
class RepositoryIndex:
    root: Path
    revision: str
    entries: dict[str, IndexEntry]
    cache_hit: bool = False

    @property
    def supported_files(self) -> int:
        return len(self.entries)

    @property
    def indexed_files(self) -> int:
        return len(self.entries)


@dataclass
class ScopedSelection:
    files: list[Path]
    changed_files: list[str]
    closure_depth: int
    closure_complete: bool
    dependency_neighborhood_files: int
    reason: Optional[str] = None


_INDEX_CACHE: dict[tuple[str, str, bool], RepositoryIndex] = {}


def _repository_cache_key(root: Path) -> str:
    """Return a stable repository identity across linked worktrees."""
    git_marker = root / ".git"
    try:
        if git_marker.is_file():
            marker = git_marker.read_text(encoding="utf-8", errors="ignore").strip()
            return marker or str(root.resolve())
        if git_marker.is_dir():
            return str(git_marker.resolve())
    except OSError:
        pass
    return str(root.resolve())


def _module_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or path.stem


def _language(path: Path) -> str:
    return {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".cs": "csharp", ".rs": "rust", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp", ".java": "java"}.get(path.suffix.lower(), "unknown")


def build_repository_index(root: Path, revision: str = "working-tree", *, lazy: bool = False) -> RepositoryIndex:
    """Build/reuse a revision-aware path index.

    ``lazy=True`` is the production mode: Git/path metadata is indexed without
    opening source blobs.  The historical eager mode remains available for
    local semantic unit tests that exercise lexical reverse-dependency rules.
    """
    key = (_repository_cache_key(root), revision, lazy)
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return RepositoryIndex(root.resolve(), cached.revision, cached.entries, cache_hit=True)
    entries: dict[str, IndexEntry] = {}
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for name in names:
            path = Path(base) / name
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            imports: list[str] = []
            for match in _IMPORT.finditer(source):
                imports.extend(x.strip().split()[0] for x in match.groups() if x and x.strip())
            # This is intentionally lexical metadata only.  ``lazy`` means
            # no AST/adapter parsing, not an empty index: imports/symbol names
            # are needed to compute the bounded semantic neighborhood.
            entries[rel] = IndexEntry(rel, _module_for(path, root), _language(path),
                                      tuple(sorted(set(_DEF.findall(source)))),
                                      tuple(sorted(set(imports))),
                                      frozenset(_TOKEN.findall(source)))
    result = RepositoryIndex(root.resolve(), revision, entries)
    _INDEX_CACHE[key] = result
    return result


def _candidate_imports(entry: IndexEntry, index: RepositoryIndex) -> set[str]:
    result: set[str] = set()
    for target in entry.imports:
        clean = target.strip("'\"./")
        for path, other in index.entries.items():
            stem = other.module
            if clean == stem or clean.startswith(stem + ".") or stem.startswith(clean + ".") or Path(path).stem == clean.split(".")[-1]:
                result.add(path)
    return result


def select_dependency_neighborhood(index: RepositoryIndex, changed_paths: Iterable[str], *, max_files: int = 5000) -> ScopedSelection:
    """Expand imports and reverse dependents until the lexical closure stabilizes."""
    changed = [str(p).replace("\\", "/").removeprefix("a/").removeprefix("b/") for p in changed_paths]
    selected = {p for p in changed if p in index.entries}
    frontier = set(selected)
    depth = 0
    while frontier:
        if len(selected) > max_files:
            return ScopedSelection([index.root / p for p in sorted(selected)], changed, depth, False, len(selected), "DEPENDENCY_CLOSURE_LIMIT_EXCEEDED")
        additions: set[str] = set()
        for path in frontier:
            entry = index.entries[path]
            additions.update(_candidate_imports(entry, index))
            symbols = set(entry.symbols)
            if symbols:
                additions.update(p for p, other in index.entries.items() if p not in selected and symbols & other.tokens)
        additions -= selected
        if not additions:
            break
        selected.update(additions)
        frontier = additions
        depth += 1
    return ScopedSelection([index.root / p for p in sorted(selected)], changed, depth, True, len(selected))


def clear_index_cache() -> None:
    _INDEX_CACHE.clear()
