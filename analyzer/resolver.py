"""
Custom Name/Module Resolver - Resolves dependencies to actual project modules/files.

This module consumes ParsedModule and ModuleDependencies objects and resolves
imports to actual project modules and symbols. It does NOT use NetworkX, Neo4j,
GAT, or any external graph libraries.

Resolution flow:
1. Discover all project modules from the file system
2. Build a symbol table mapping module names to file paths and exported symbols
3. For each module, build an import-name lookup (alias → qualified symbol)
4. Resolve each dependency against the symbol table and module map
5. Handle relative imports by converting to absolute module names
6. Score confidence based on resolution quality

Confidence scale:
- 1.0  exact imported symbol or exact local symbol found in symbol table
- 0.9  known module attribute (module found, symbol within it)
- 0.7  parent-module match (submodule referenced through parent)
- 0.5  ambiguous / multiple candidates exist
- 0.0  unresolved

Limitations (documented, not guessed):
- Dynamic imports (__import__(), importlib.import_module()) are not resolved.
- Wildcard imports (from module import *) are not expanded.
- Conditional imports inside branches are treated like any other import.
- Type-alias symbols introduced via TYPE_CHECKING blocks are not tracked.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Set, Any, Tuple
from pathlib import Path
from collections import defaultdict

from analyzer.ast_parser import (
    ParsedModule,
    ImportInfo,
    ClassInfo,
    FunctionInfo,
    ImportType,
    SourceLocation,
)
from analyzer.dependency_extractor import (
    ModuleDependencies,
    Dependency,
    DependencyType,
    extract_dependencies,
    extract_dependencies_multiple,
)


# ---------------------------------------------------------------------------
# Standard library module names (Python 3.x top-level packages)
# Used to classify EXTERNAL dependencies without network calls.
# ---------------------------------------------------------------------------
_STDLIB_TOP_LEVEL: Set[str] = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {
    # Fallback for Python < 3.10
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
    "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
    "colorsys", "compileall", "concurrent", "configparser", "contextlib",
    "contextvars", "copy", "copyreg", "cProfile", "csv", "ctypes",
    "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib",
    "dis", "distutils", "doctest", "email", "encodings", "enum",
    "errno", "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch",
    "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "lib2to3",
    "linecache", "locale", "logging", "lzma", "mailbox", "mailcap",
    "marshal", "math", "mimetypes", "mmap", "modulefinder", "multiprocessing",
    "netrc", "nis", "nntplib", "numbers", "operator", "optparse",
    "os", "ossaudiodev", "pathlib", "pdb", "pickle", "pickletools",
    "pipes", "pkgutil", "platform", "plistlib", "poplib", "posix",
    "posixpath", "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
    "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "rlcompleter", "runpy", "sched", "secrets", "select",
    "selectors", "shelve", "shlex", "shutil", "signal", "site",
    "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
    "sqlite3", "sre_compile", "sre_constants", "sre_parse", "ssl",
    "stat", "statistics", "string", "stringprep", "struct", "subprocess",
    "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize",
    "tomllib", "trace", "traceback", "tracemalloc", "tty", "turtle",
    "turtledemo", "types", "typing", "unicodedata", "unittest",
    "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
    "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
    "_thread", "__future__",
}


# ---------------------------------------------------------------------------
# Enums & core data structures
# ---------------------------------------------------------------------------

class ResolutionStatus(Enum):
    """Status of dependency resolution."""
    RESOLVED = "resolved"           # Successfully resolved to internal module
    EXTERNAL = "external"           # Third-party / stdlib module
    UNRESOLVED = "unresolved"       # Could not be resolved
    RELATIVE_UNRESOLVED = "relative_unresolved"  # Relative import couldn't be resolved


@dataclass
class Symbol:
    """A symbol (class, function, method) defined in a module."""
    name: str
    qualified_name: str          # e.g. "services.payment.process_payment"
    module_name: str             # e.g. "services.payment"
    file_path: str               # Absolute path to source file
    symbol_type: str             # 'class' | 'function' | 'method'
    parent_class: Optional[str] = None
    is_public: bool = True       # False if name starts with '_'


@dataclass
class ModuleInfo:
    """Information about a project module."""
    module_name: str
    file_path: Path
    package: str         # Parent package (empty for top-level)
    is_package: bool     # True for __init__.py files
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    submodules: Dict[str, 'ModuleInfo'] = field(default_factory=dict)


@dataclass
class ResolvedDependency:
    """A dependency that has been resolved to its origin module/symbol."""
    source_module: str
    source_symbol: Optional[str]
    original_target: str
    dep_type: DependencyType
    source_file: str
    source_line: int
    resolved_module: Optional[str] = None    # Module that owns the target
    resolved_file: Optional[str] = None      # Absolute path to the module file
    resolved_symbol: Optional[str] = None    # Fully-qualified symbol name
    status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    confidence: float = 0.0
    resolution_reason: str = ""
    is_relative: bool = False
    relative_level: int = 0

    def is_resolved(self) -> bool:
        return self.status == ResolutionStatus.RESOLVED

    def is_external(self) -> bool:
        return self.status == ResolutionStatus.EXTERNAL


@dataclass
class ResolutionResult:
    """Complete resolution result for a single module."""
    module_name: str
    file_path: str
    resolved_dependencies: List[ResolvedDependency] = field(default_factory=list)
    symbol_table: Dict[str, Symbol] = field(default_factory=dict)

    # Statistics
    total_dependencies: int = 0
    resolved_count: int = 0
    external_count: int = 0
    unresolved_count: int = 0

    def add_resolved(self, dep: ResolvedDependency) -> None:
        self.resolved_dependencies.append(dep)
        self.total_dependencies += 1
        if dep.is_resolved():
            self.resolved_count += 1
        elif dep.is_external():
            self.external_count += 1
        else:
            self.unresolved_count += 1

    @property
    def resolution_percentage(self) -> float:
        if self.total_dependencies == 0:
            return 100.0
        return (self.resolved_count / self.total_dependencies) * 100


@dataclass
class ProjectResolutionResult:
    """Complete resolution result for an entire project."""
    project_root: Path
    module_results: Dict[str, ResolutionResult] = field(default_factory=dict)
    global_symbol_table: Dict[str, Symbol] = field(default_factory=dict)
    all_resolved_dependencies: List[ResolvedDependency] = field(default_factory=list)

    # Project-wide statistics
    total_modules: int = 0
    total_dependencies: int = 0
    total_resolved: int = 0
    total_external: int = 0
    total_unresolved: int = 0

    def add_module_result(self, result: ResolutionResult) -> None:
        self.module_results[result.module_name] = result
        self.total_modules += 1
        self.total_dependencies += result.total_dependencies
        self.total_resolved += result.resolved_count
        self.total_external += result.external_count
        self.total_unresolved += result.unresolved_count
        self.all_resolved_dependencies.extend(result.resolved_dependencies)
        self.global_symbol_table.update(result.symbol_table)

    @property
    def resolution_percentage(self) -> float:
        if self.total_dependencies == 0:
            return 100.0
        return (self.total_resolved / self.total_dependencies) * 100


# ---------------------------------------------------------------------------
# Module Discovery
# ---------------------------------------------------------------------------

class ModuleDiscovery:
    """
    Discovers and catalogs project modules from the file system.

    Walks the project root, skips excluded directories, and maps
    every .py file to a Python module name using standard package
    naming rules (__init__.py → parent package).
    """

    DEFAULT_EXCLUDE = frozenset({
        "__pycache__", ".git", "venv", "env", ".venv", ".env",
        "node_modules", "dist", "build", ".pytest_cache",
        ".mypy_cache", "htmlcov", ".idea", ".vscode",
    })

    def __init__(self, project_root: Path, exclude_dirs: Optional[List[str]] = None):
        self.project_root = Path(project_root).resolve()
        self.exclude_dirs = set(exclude_dirs or []) | self.DEFAULT_EXCLUDE
        self._modules: Dict[str, ModuleInfo] = {}

    def discover(self, source_paths: Optional[List[Path]] = None) -> Dict[str, ModuleInfo]:
        """Return mapping of module_name → ModuleInfo for all project modules."""
        self._modules = {}

        # Production scoped analysis supplies the already-selected closure.
        # Do not rediscover/parse every Python file in a checkout.  The
        # default remains repository-wide for legacy callers and tests.
        candidates = (source_paths if source_paths is not None else
                      sorted(self.project_root.rglob("*.py")))
        for py_file in sorted(Path(p).resolve() for p in candidates):
            if py_file.suffix.lower() != ".py":
                continue
            if self._should_exclude(py_file):
                continue

            rel_path = py_file.relative_to(self.project_root)
            module_name = self._file_to_module_name(rel_path)

            # __init__.py for a package already maps to the package name;
            # don't overwrite if already registered from a sub-walk.
            if module_name not in self._modules:
                module_info = self._create_module_info(module_name, py_file, rel_path)
                self._modules[module_name] = module_info

        self._build_package_hierarchy()
        return self._modules

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_exclude(self, file_path: Path) -> bool:
        try:
            rel_path = file_path.relative_to(self.project_root)
            return any(part in self.exclude_dirs for part in rel_path.parts)
        except ValueError:
            return True

    def _file_to_module_name(self, rel_path: Path) -> str:
        """Convert relative file path to dotted Python module name."""
        parts = list(rel_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        return ".".join(parts) if parts else ""

    def _create_module_info(self, module_name: str, file_path: Path,
                            rel_path: Path) -> ModuleInfo:
        parts = module_name.split(".")
        package = ".".join(parts[:-1]) if len(parts) > 1 else ""
        is_package = rel_path.name == "__init__.py"
        return ModuleInfo(
            module_name=module_name,
            file_path=file_path,
            package=package,
            is_package=is_package,
        )

    def _build_package_hierarchy(self) -> None:
        """Link submodules to their parent packages."""
        for module_name, module_info in self._modules.items():
            parts = module_name.split(".")
            if len(parts) > 1:
                parent_name = ".".join(parts[:-1])
                if parent_name in self._modules:
                    self._modules[parent_name].submodules[parts[-1]] = module_info


# ---------------------------------------------------------------------------
# Symbol Table Builder
# ---------------------------------------------------------------------------

class SymbolTableBuilder:
    """
    Builds a project-wide symbol table from a list of ParsedModules.

    Produces a flat dict  qualified_name → Symbol  where qualified names
    follow the pattern  <module_name>.<ClassName>  or
    <module_name>.<ClassName>.<method_name>  for methods.
    """

    def __init__(self) -> None:
        self.symbols: Dict[str, Symbol] = {}

    def build_from_parsed_modules(self, parsed_modules: List[ParsedModule],
                                   module_name_override: Optional[Dict[str, str]] = None
                                   ) -> Dict[str, Symbol]:
        """
        Build symbol table from parsed modules.

        Args:
            parsed_modules: Modules to index.
            module_name_override: Optional mapping of file_path → module_name
                to use instead of parsed_module.module_name (used when the
                module_name on ParsedModule is just the stem, e.g. "payment"
                but the real dotted name is "services.payment").
        """
        self.symbols = {}
        override = module_name_override or {}
        for parsed in parsed_modules:
            real_name = override.get(parsed.file_path, parsed.module_name)
            self._extract_symbols_from_module(parsed, real_name)
        return self.symbols

    def _extract_symbols_from_module(self, parsed: ParsedModule, module_name: str) -> None:
        file_path = parsed.file_path

        # Classes
        for cls in parsed.classes:
            sym = Symbol(
                name=cls.name,
                qualified_name=f"{module_name}.{cls.name}",
                module_name=module_name,
                file_path=file_path,
                symbol_type="class",
                is_public=not cls.name.startswith("_"),
            )
            self.symbols[sym.qualified_name] = sym

            # Methods
            for method in cls.methods:
                msym = Symbol(
                    name=method.name,
                    qualified_name=f"{module_name}.{cls.name}.{method.name}",
                    module_name=module_name,
                    file_path=file_path,
                    symbol_type="method",
                    parent_class=cls.name,
                    is_public=not method.name.startswith("_"),
                )
                self.symbols[msym.qualified_name] = msym

        # Standalone functions
        for func in parsed.functions:
            sym = Symbol(
                name=func.name,
                qualified_name=f"{module_name}.{func.name}",
                module_name=module_name,
                file_path=file_path,
                symbol_type="function",
                is_public=not func.name.startswith("_"),
            )
            self.symbols[sym.qualified_name] = sym

    def get_symbol(self, qualified_name: str) -> Optional[Symbol]:
        return self.symbols.get(qualified_name)

    def get_symbols_by_module(self, module_name: str) -> Dict[str, Symbol]:
        return {k: v for k, v in self.symbols.items() if v.module_name == module_name}

    def get_symbols_by_type(self, symbol_type: str) -> Dict[str, Symbol]:
        return {k: v for k, v in self.symbols.items() if v.symbol_type == symbol_type}


# ---------------------------------------------------------------------------
# Import-Name Lookup (per-module)
# ---------------------------------------------------------------------------

def _build_import_lookup(imports: List[ImportInfo]) -> Dict[str, str]:
    """
    Build a mapping of  local_name → qualified_target  from a module's imports.

    This is used by call resolution: when we see `helper()` called in a
    function, we look up `helper` here to find out which module it came from.

    Examples
    --------
    ``from utils import helper``        → ``{"helper": "utils.helper"}``
    ``import os``                       → ``{"os": "os"}``
    ``import services.payment as pay``  → ``{"pay": "services.payment"}``
    ``from services.auth import User``  → ``{"User": "services.auth.User"}``
    """
    lookup: Dict[str, str] = {}
    for imp in imports:
        if imp.import_type == ImportType.IMPORT:
            # import module [as alias]
            local = imp.alias if imp.alias else imp.module
            lookup[local] = imp.module
            # Also handle dotted access: ``import os.path`` → ``os`` → ``os``
            top = imp.module.split(".")[0]
            if top not in lookup:
                lookup[top] = imp.module
        elif imp.import_type == ImportType.FROM_IMPORT:
            # from module import name [as alias]
            for name in imp.names:
                local = imp.alias if imp.alias else name
                module_clean = imp.module.lstrip(".")
                if module_clean:
                    lookup[local] = f"{module_clean}.{name}"
                else:
                    # `from . import sibling` - the module IS the name;
                    # we don't know the absolute name yet without context,
                    # so store the raw name and let the resolver handle it.
                    lookup[local] = name
    return lookup


def _is_stdlib(top_level_name: str) -> bool:
    """Return True if top_level_name is part of the Python standard library."""
    return top_level_name in _STDLIB_TOP_LEVEL


def _is_external_module(module_name: str, module_map: Dict[str, "ModuleInfo"]) -> bool:
    """
    Classify a module name as external (stdlib or third-party).

    Strategy:
    1. If the top-level package is in the stdlib set → external.
    2. If neither the module itself nor any parent exists in our module_map → external.
    """
    top = module_name.split(".")[0]
    if _is_stdlib(top):
        return True
    # Check if any prefix exists in our project
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in module_map:
            return False  # internal
    return True  # not found → external


# ---------------------------------------------------------------------------
# Name Resolver (main class)
# ---------------------------------------------------------------------------

class NameResolver:
    """
    Resolves import dependencies to actual project modules and symbols.

    Handles:
    - Absolute imports  (``import module``, ``from module import name``)
    - Aliased imports   (``import module as alias``, ``from m import n as a``)
    - Relative imports  (``from . import name``, ``from ..pkg import sym``)
    - Symbol resolution (functions, classes, methods)
    - Cross-module call resolution using the per-module import lookup
    - Internal vs. external classification

    Must call :meth:`prepare` or :meth:`resolve_project` before resolving.
    """

    def __init__(self, project_root: Path, exclude_dirs: Optional[List[str]] = None):
        self.project_root = Path(project_root).resolve()
        self.discovery = ModuleDiscovery(project_root, exclude_dirs)
        self.symbol_builder = SymbolTableBuilder()
        self._module_map: Dict[str, ModuleInfo] = {}
        self._symbol_table: Dict[str, Symbol] = {}
        # file_path → dotted module name (filled during prepare())
        self._file_to_module: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, parsed_modules: List[ParsedModule], source_paths: Optional[List[Path]] = None) -> None:
        """
        Prepare resolver: discover project modules and build symbol tables.

        Args:
            parsed_modules: All ParsedModule objects for the project.
        """
        # 1. Discover all .py files → dotted module names
        paths = source_paths if source_paths is not None else [Path(pm.file_path) for pm in parsed_modules]
        self._module_map = self.discovery.discover(paths)
        self._file_to_module = {
            str(info.file_path): name
            for name, info in self._module_map.items()
        }

        # 2. Build override map: file_path → real dotted module name
        #    (ParsedModule.module_name may only be the stem, e.g. "payment"
        #    instead of "services.payment")
        override: Dict[str, str] = {}
        for pm in parsed_modules:
            real = self._file_to_module.get(pm.file_path)
            if real:
                override[pm.file_path] = real

        # 3. Build the global symbol table with correct qualified names
        self._symbol_table = self.symbol_builder.build_from_parsed_modules(
            parsed_modules, module_name_override=override
        )

    def resolve_module_dependencies(self, module_deps: ModuleDependencies,
                                    parsed_module: Optional[ParsedModule] = None
                                    ) -> ResolutionResult:
        """
        Resolve all dependencies for a single module.

        Args:
            module_deps: ModuleDependencies from the dependency extractor.
            parsed_module: The original ParsedModule (for import context).
                           If omitted, cross-module call resolution is disabled.

        Returns:
            ResolutionResult with all resolved dependencies.
        """
        # Determine the real dotted module name
        real_module_name = (
            self._file_to_module.get(module_deps.file_path)
            or module_deps.module_name
        )

        result = ResolutionResult(
            module_name=real_module_name,
            file_path=module_deps.file_path,
        )

        # Populate per-module symbol table slice
        result.symbol_table = {
            k: v for k, v in self._symbol_table.items()
            if v.module_name == real_module_name
        }

        # Build import lookup (local-name → qualified target) for call resolution
        import_lookup: Dict[str, str] = {}
        if parsed_module is not None:
            import_lookup = _build_import_lookup(parsed_module.imports)
        elif module_deps.imports:
            import_lookup = _build_import_lookup(module_deps.imports)

        for dep in module_deps.dependencies:
            resolved = self._resolve_dependency(dep, real_module_name, import_lookup)
            result.add_resolved(resolved)

        return result

    def resolve_project(self, parsed_modules: List[ParsedModule],
                        source_paths: Optional[List[Path]] = None) -> ProjectResolutionResult:
        """
        Resolve dependencies for an entire project.

        Args:
            parsed_modules: All ParsedModule objects.

        Returns:
            ProjectResolutionResult with all resolved dependencies.
        """
        self.prepare(parsed_modules, source_paths=source_paths)

        module_deps_list = extract_dependencies_multiple(parsed_modules, self.project_root)

        # Build a quick map file_path → ParsedModule for import context
        pm_by_file: Dict[str, ParsedModule] = {pm.file_path: pm for pm in parsed_modules}

        project_result = ProjectResolutionResult(project_root=self.project_root)
        for module_deps in module_deps_list:
            pm = pm_by_file.get(module_deps.file_path)
            module_result = self.resolve_module_dependencies(module_deps, parsed_module=pm)
            project_result.add_module_result(module_result)

        return project_result

    # ------------------------------------------------------------------
    # Dependency dispatch
    # ------------------------------------------------------------------

    def _resolve_dependency(self, dep: Dependency, current_module: str,
                            import_lookup: Dict[str, str]) -> ResolvedDependency:
        """Dispatch a single dependency to the appropriate resolver."""
        resolved = ResolvedDependency(
            source_module=dep.source,
            source_symbol=dep.source_symbol,
            original_target=dep.target,
            dep_type=dep.dep_type,
            source_file=dep.source_file,
            source_line=dep.source_line,
            is_relative=dep.is_relative,
            relative_level=dep.relative_level,
            confidence=dep.confidence,
        )

        if dep.dep_type in (DependencyType.IMPORT, DependencyType.FROM_IMPORT):
            return self._resolve_import(dep, current_module, resolved)
        elif dep.dep_type == DependencyType.INHERITANCE:
            return self._resolve_inheritance(dep, current_module, resolved, import_lookup)
        elif dep.dep_type == DependencyType.FUNCTION_CALL:
            return self._resolve_call(dep, current_module, resolved, import_lookup)
        elif dep.dep_type == DependencyType.DECORATOR:
            return self._resolve_decorator(dep, current_module, resolved, import_lookup)
        else:
            resolved.resolution_reason = f"Unknown dependency type: {dep.dep_type}"
            return resolved

    # ------------------------------------------------------------------
    # Import resolution
    # ------------------------------------------------------------------

    def _resolve_import(self, dep: Dependency, current_module: str,
                        resolved: ResolvedDependency) -> ResolvedDependency:
        """Resolve import / from-import dependency."""
        target = dep.target

        if dep.is_relative:
            return self._resolve_relative(dep, current_module, resolved)

        # Absolute resolution -------------------------------------------

        # 1. Try exact module name match (e.g. "services.payment")
        if target in self._module_map:
            resolved.resolved_module = target
            resolved.resolved_file = str(self._module_map[target].file_path)
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 1.0
            resolved.resolution_reason = f"Exact internal module: {target}"
            return resolved

        # 2. Try exact symbol match (e.g. "services.payment.process_payment")
        if target in self._symbol_table:
            sym = self._symbol_table[target]
            resolved.resolved_module = sym.module_name
            resolved.resolved_file = sym.file_path
            resolved.resolved_symbol = sym.qualified_name
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 1.0
            resolved.resolution_reason = f"Exact symbol: {sym.qualified_name}"
            return resolved

        # 3. For "module.symbol" style targets, try resolving the module part
        #    then looking up the symbol within it.
        parts = target.split(".")
        if len(parts) >= 2:
            # Try progressively shorter module prefixes
            for split_at in range(len(parts) - 1, 0, -1):
                module_candidate = ".".join(parts[:split_at])
                symbol_candidate = ".".join(parts[split_at:])
                if module_candidate in self._module_map:
                    # Module is internal; look for the symbol
                    full_sym_key = f"{module_candidate}.{symbol_candidate}"
                    if full_sym_key in self._symbol_table:
                        sym = self._symbol_table[full_sym_key]
                        resolved.resolved_module = module_candidate
                        resolved.resolved_file = sym.file_path
                        resolved.resolved_symbol = sym.qualified_name
                        resolved.status = ResolutionStatus.RESOLVED
                        resolved.confidence = 1.0
                        resolved.resolution_reason = (
                            f"Symbol {symbol_candidate} in module {module_candidate}"
                        )
                    else:
                        # Module found but symbol unknown → partial resolution
                        resolved.resolved_module = module_candidate
                        resolved.resolved_file = str(self._module_map[module_candidate].file_path)
                        resolved.status = ResolutionStatus.RESOLVED
                        resolved.confidence = 0.9
                        resolved.resolution_reason = (
                            f"Module {module_candidate} found; symbol '{symbol_candidate}' not indexed"
                        )
                    return resolved

        # 4. Classify as external
        top = parts[0]
        if _is_external_module(top, self._module_map):
            resolved.status = ResolutionStatus.EXTERNAL
            resolved.confidence = 1.0
            resolved.resolution_reason = (
                f"Standard library" if _is_stdlib(top) else f"Third-party package: {top}"
            )
        else:
            resolved.status = ResolutionStatus.UNRESOLVED
            resolved.confidence = 0.0
            resolved.resolution_reason = f"Could not resolve: {target}"
        return resolved

    def _resolve_relative(self, dep: Dependency, current_module: str,
                          resolved: ResolvedDependency) -> ResolvedDependency:
        """Resolve a relative import dependency."""
        target = dep.target        # e.g. ".local.helper" or "..services.auth.User"
        level = dep.relative_level

        # Strip leading dots that appear in target (AST parser prefix)
        target_clean = target.lstrip(".")

        absolute_module = self._relative_to_absolute(
            target_clean, current_module, level
        )

        if absolute_module is None:
            resolved.status = ResolutionStatus.RELATIVE_UNRESOLVED
            resolved.confidence = 0.0
            resolved.resolution_reason = (
                f"Relative import (level={level}) could not be resolved "
                f"from module '{current_module}'"
            )
            return resolved

        # Check if resolved module exists in project
        if absolute_module in self._module_map:
            resolved.resolved_module = absolute_module
            resolved.resolved_file = str(self._module_map[absolute_module].file_path)
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 1.0
            resolved.resolution_reason = f"Relative import resolved to {absolute_module}"
            return resolved

        # Try it as a symbol within a module (split last component)
        parts = absolute_module.split(".")
        for split_at in range(len(parts) - 1, 0, -1):
            mod_candidate = ".".join(parts[:split_at])
            sym_candidate = ".".join(parts[split_at:])
            if mod_candidate in self._module_map:
                full_sym_key = f"{mod_candidate}.{sym_candidate}"
                if full_sym_key in self._symbol_table:
                    sym = self._symbol_table[full_sym_key]
                    resolved.resolved_module = mod_candidate
                    resolved.resolved_file = sym.file_path
                    resolved.resolved_symbol = sym.qualified_name
                    resolved.status = ResolutionStatus.RESOLVED
                    resolved.confidence = 1.0
                    resolved.resolution_reason = (
                        f"Relative import resolved to symbol {sym.qualified_name}"
                    )
                else:
                    resolved.resolved_module = mod_candidate
                    resolved.resolved_file = str(self._module_map[mod_candidate].file_path)
                    resolved.status = ResolutionStatus.RESOLVED
                    resolved.confidence = 0.9
                    resolved.resolution_reason = (
                        f"Relative import: module {mod_candidate} found; "
                        f"symbol '{sym_candidate}' not indexed"
                    )
                return resolved

        resolved.status = ResolutionStatus.RELATIVE_UNRESOLVED
        resolved.confidence = 0.0
        resolved.resolution_reason = (
            f"Relative import resolved to '{absolute_module}' but module not in project"
        )
        return resolved

    def _relative_to_absolute(self, target_clean: str, current_module: str,
                               level: int) -> Optional[str]:
        """
        Convert a relative import to an absolute module name.

        Args:
            target_clean: The name part without leading dots (may be empty for
                          ``from . import something``).
            current_module: Dotted name of the importing module.
            level: Number of leading dots (1 = same package, 2 = parent, etc.)

        Returns:
            Absolute dotted module name, or None if impossible.
        """
        if not current_module or current_module in ("<string>", "<module>"):
            return None

        current_parts = current_module.split(".")

        # level=1 → same package (go up 1 to strip the module leaf)
        # level=2 → parent package (go up 2), etc.
        if level > len(current_parts):
            return None  # Can't go higher than root

        base_parts = current_parts[:-level] if level > 0 else current_parts[:]

        if target_clean:
            base_parts.extend(target_clean.split("."))

        return ".".join(base_parts) if base_parts else None

    # ------------------------------------------------------------------
    # Inheritance resolution
    # ------------------------------------------------------------------

    def _resolve_inheritance(self, dep: Dependency, current_module: str,
                             resolved: ResolvedDependency,
                             import_lookup: Dict[str, str]) -> ResolvedDependency:
        """Resolve class inheritance dependency."""
        base_name = dep.target  # e.g. "Base" or "models.Base"

        # 1. Same module?
        same_mod_key = f"{current_module}.{base_name}"
        if same_mod_key in self._symbol_table:
            sym = self._symbol_table[same_mod_key]
            resolved.resolved_module = sym.module_name
            resolved.resolved_file = sym.file_path
            resolved.resolved_symbol = sym.qualified_name
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 1.0
            resolved.resolution_reason = f"Base class in same module: {sym.qualified_name}"
            return resolved

        # 2. Check import lookup  (e.g. "from models import Base" → "models.Base")
        if base_name in import_lookup:
            qualified_target = import_lookup[base_name]
            # Recurse into symbol table
            if qualified_target in self._symbol_table:
                sym = self._symbol_table[qualified_target]
                resolved.resolved_module = sym.module_name
                resolved.resolved_file = sym.file_path
                resolved.resolved_symbol = sym.qualified_name
                resolved.status = ResolutionStatus.RESOLVED
                resolved.confidence = 1.0
                resolved.resolution_reason = f"Base class via import: {sym.qualified_name}"
                return resolved
            # Module may be internal but symbol not indexed
            top = qualified_target.split(".")[0]
            if not _is_external_module(top, self._module_map):
                module_part = ".".join(qualified_target.split(".")[:-1])
                if module_part in self._module_map:
                    resolved.resolved_module = module_part
                    resolved.resolved_file = str(self._module_map[module_part].file_path)
                    resolved.status = ResolutionStatus.RESOLVED
                    resolved.confidence = 0.9
                    resolved.resolution_reason = (
                        f"Base class module found: {module_part}; symbol not indexed"
                    )
                    return resolved

        # 3. Search all modules for a class with this name
        candidates = [
            sym for sym in self._symbol_table.values()
            if sym.name == base_name and sym.symbol_type == "class"
        ]
        if len(candidates) == 1:
            sym = candidates[0]
            resolved.resolved_module = sym.module_name
            resolved.resolved_file = sym.file_path
            resolved.resolved_symbol = sym.qualified_name
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 0.9
            resolved.resolution_reason = (
                f"Unique class match across project: {sym.qualified_name}"
            )
            return resolved
        if len(candidates) > 1:
            # Ambiguous - pick first but lower confidence
            sym = candidates[0]
            resolved.resolved_module = sym.module_name
            resolved.resolved_file = sym.file_path
            resolved.resolved_symbol = sym.qualified_name
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 0.5
            resolved.resolution_reason = (
                f"Ambiguous base class ({len(candidates)} candidates): "
                f"using {sym.qualified_name}"
            )
            return resolved

        # 4. Likely external (stdlib, third-party)
        resolved.status = ResolutionStatus.EXTERNAL
        resolved.confidence = 1.0
        resolved.resolution_reason = f"Base class likely external: {base_name}"
        return resolved

    # ------------------------------------------------------------------
    # Call resolution
    # ------------------------------------------------------------------

    def _resolve_call(self, dep: Dependency, current_module: str,
                      resolved: ResolvedDependency,
                      import_lookup: Dict[str, str]) -> ResolvedDependency:
        """
        Resolve a function/method call dependency.

        Strategy (in order):
        1. Exact local symbol in same module.
        2. Import lookup → symbol table.
        3. Import lookup → module-level partial match.
        4. Global symbol table search (unique name → 0.5 confidence).
        5. Mark unresolved.
        """
        call_name = dep.target  # e.g. "helper" or "process_payment"

        # 1. Local symbol?
        local_key = f"{current_module}.{call_name}"
        if local_key in self._symbol_table:
            sym = self._symbol_table[local_key]
            resolved.resolved_module = sym.module_name
            resolved.resolved_file = sym.file_path
            resolved.resolved_symbol = sym.qualified_name
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 1.0
            resolved.resolution_reason = f"Local call: {sym.qualified_name}"
            return resolved

        # 2. Via import lookup
        if call_name in import_lookup:
            qualified_target = import_lookup[call_name]
            # Direct symbol hit
            if qualified_target in self._symbol_table:
                sym = self._symbol_table[qualified_target]
                resolved.resolved_module = sym.module_name
                resolved.resolved_file = sym.file_path
                resolved.resolved_symbol = sym.qualified_name
                resolved.status = ResolutionStatus.RESOLVED
                resolved.confidence = 1.0
                resolved.resolution_reason = f"Call via import: {sym.qualified_name}"
                return resolved

            # Module hit without symbol
            top = qualified_target.split(".")[0]
            if _is_external_module(top, self._module_map):
                resolved.status = ResolutionStatus.EXTERNAL
                resolved.confidence = 1.0
                resolved.resolution_reason = f"Call to external symbol: {qualified_target}"
                return resolved

            # Internal module exists
            parts = qualified_target.split(".")
            for split_at in range(len(parts), 0, -1):
                mod_candidate = ".".join(parts[:split_at])
                if mod_candidate in self._module_map:
                    resolved.resolved_module = mod_candidate
                    resolved.resolved_file = str(self._module_map[mod_candidate].file_path)
                    resolved.status = ResolutionStatus.RESOLVED
                    resolved.confidence = 0.9
                    resolved.resolution_reason = (
                        f"Call via import to module {mod_candidate} "
                        f"(symbol '{call_name}' not indexed)"
                    )
                    return resolved

        # 3. Search whole project for unique function/method with this name
        candidates = [
            sym for sym in self._symbol_table.values()
            if sym.name == call_name and sym.symbol_type in ("function", "method")
        ]
        if len(candidates) == 1:
            sym = candidates[0]
            resolved.resolved_module = sym.module_name
            resolved.resolved_file = sym.file_path
            resolved.resolved_symbol = sym.qualified_name
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 0.5
            resolved.resolution_reason = (
                f"Unique function match across project: {sym.qualified_name}"
            )
            return resolved

        # 4. Unresolved
        resolved.status = ResolutionStatus.UNRESOLVED
        resolved.confidence = 0.0
        resolved.resolution_reason = f"Call target not resolved: {call_name}"
        return resolved

    # ------------------------------------------------------------------
    # Decorator resolution
    # ------------------------------------------------------------------

    def _resolve_decorator(self, dep: Dependency, current_module: str,
                           resolved: ResolvedDependency,
                           import_lookup: Dict[str, str]) -> ResolvedDependency:
        """Resolve decorator dependency."""
        decorator_name = dep.target

        # Built-in decorators
        _BUILTINS = frozenset({
            "property", "staticmethod", "classmethod", "abstractmethod",
            "overload", "final", "dataclass", "cached_property",
        })
        if decorator_name in _BUILTINS:
            resolved.status = ResolutionStatus.EXTERNAL
            resolved.confidence = 1.0
            resolved.resolution_reason = f"Built-in decorator: {decorator_name}"
            return resolved

        # Local decorator?
        local_key = f"{current_module}.{decorator_name}"
        if local_key in self._symbol_table:
            sym = self._symbol_table[local_key]
            resolved.resolved_module = sym.module_name
            resolved.resolved_file = sym.file_path
            resolved.resolved_symbol = sym.qualified_name
            resolved.status = ResolutionStatus.RESOLVED
            resolved.confidence = 1.0
            resolved.resolution_reason = f"Local decorator: {sym.qualified_name}"
            return resolved

        # Via import lookup
        if decorator_name in import_lookup:
            qualified_target = import_lookup[decorator_name]
            if qualified_target in self._symbol_table:
                sym = self._symbol_table[qualified_target]
                resolved.resolved_module = sym.module_name
                resolved.resolved_file = sym.file_path
                resolved.resolved_symbol = sym.qualified_name
                resolved.status = ResolutionStatus.RESOLVED
                resolved.confidence = 1.0
                resolved.resolution_reason = f"Decorator via import: {sym.qualified_name}"
                return resolved
            top = qualified_target.split(".")[0]
            if _is_external_module(top, self._module_map):
                resolved.status = ResolutionStatus.EXTERNAL
                resolved.confidence = 1.0
                resolved.resolution_reason = f"External decorator: {qualified_target}"
                return resolved

        resolved.status = ResolutionStatus.UNRESOLVED
        resolved.confidence = 0.0
        resolved.resolution_reason = f"Decorator not resolved: {decorator_name}"
        return resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _module_to_file(self, module_name: str) -> Optional[str]:
        """Return absolute file path for a module name, or None."""
        info = self._module_map.get(module_name)
        return str(info.file_path) if info else None

    def _find_parent_module(self, module_name: str) -> Optional[str]:
        """Find the nearest ancestor module that exists in the project."""
        parts = module_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent = ".".join(parts[:i])
            if parent in self._module_map:
                return parent
        return None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def resolve_dependencies(project_root: Path,
                         parsed_modules: List[ParsedModule],
                         exclude_dirs: Optional[List[str]] = None
                         ) -> ProjectResolutionResult:
    """Resolve all dependencies in a project. Convenience wrapper."""
    resolver = NameResolver(project_root, exclude_dirs)
    return resolver.resolve_project(parsed_modules)


def resolve_single_module(project_root: Path,
                          module_deps: ModuleDependencies,
                          parsed_modules: List[ParsedModule],
                          parsed_module: Optional[ParsedModule] = None,
                          exclude_dirs: Optional[List[str]] = None
                          ) -> ResolutionResult:
    """Resolve dependencies for a single module. Convenience wrapper."""
    resolver = NameResolver(project_root, exclude_dirs)
    resolver.prepare(parsed_modules)
    return resolver.resolve_module_dependencies(module_deps, parsed_module=parsed_module)
