"""
Custom Dependency Extractor - Extracts dependencies from parsed AST modules.

This module identifies relationships between code elements from ParsedModule
objects produced by the AST parser. It does NOT use NetworkX, Neo4j, GAT,
or any external graph libraries.

Dependencies are extracted but NOT resolved - resolution happens in the
Name/Module Resolver phase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Set, Dict, Any
from pathlib import Path

from analyzer.ast_parser import (
    ParsedModule,
    ImportInfo,
    ClassInfo,
    FunctionInfo,
    ImportType,
    SourceLocation,
)


class DependencyType(Enum):
    """Types of dependencies that can be extracted."""
    IMPORT = "import"                 # import module
    FROM_IMPORT = "from_import"       # from module import name
    FUNCTION_CALL = "function_call"   # func()
    METHOD_CALL = "method_call"       # obj.method()
    INHERITANCE = "inheritance"       # class Child(Base)
    MODULE_REFERENCE = "module_reference"  # reference to module attribute
    DECORATOR = "decorator"           # @decorator usage


@dataclass
class Dependency:
    """A single dependency relationship between code elements."""
    source: str                       # Source element (module, class, function)
    target: str                       # Target element being depended on
    dep_type: DependencyType
    source_file: str                  # File where dependency originates
    source_line: int                  # Line number of dependency
    source_symbol: Optional[str] = None  # Specific symbol (function/method) if applicable
    target_symbol: Optional[str] = None  # Specific target symbol if applicable
    is_relative: bool = False         # True for relative imports
    relative_level: int = 0           # Relative import level (0 = absolute)
    confidence: float = 1.0           # Confidence in this dependency (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        rel = f" (rel={self.relative_level})" if self.is_relative else ""
        return f"{self.source} --[{self.dep_type.value}]--> {self.target}{rel} @ {self.source_file}:{self.source_line}"


@dataclass
class ModuleDependencies:
    """All dependencies extracted from a single module."""
    module_name: str
    file_path: str
    dependencies: List[Dependency] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    local_symbols: Set[str] = field(default_factory=set)  # Defined in this module
    external_symbols: Set[str] = field(default_factory=set)  # Imported from outside

    def add_dependency(self, dep: Dependency) -> None:
        self.dependencies.append(dep)

    def get_by_type(self, dep_type: DependencyType) -> List[Dependency]:
        return [d for d in self.dependencies if d.dep_type == dep_type]

    def get_imports(self) -> List[Dependency]:
        return self.get_by_type(DependencyType.IMPORT) + self.get_by_type(DependencyType.FROM_IMPORT)


class DependencyExtractor:
    """
    Extracts dependencies from ParsedModule objects.

    Produces ModuleDependencies containing all identified relationships.
    Does NOT resolve imports to actual files - that's the resolver's job.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the dependency extractor.

        Args:
            project_root: Root path of the project (for relative import resolution hints)
        """
        self.project_root = project_root or Path.cwd()

    def extract(self, parsed_module: ParsedModule) -> ModuleDependencies:
        """
        Extract all dependencies from a parsed module.

        Args:
            parsed_module: ParsedModule from AST parser

        Returns:
            ModuleDependencies with all extracted dependencies
        """
        result = ModuleDependencies(
            module_name=parsed_module.module_name,
            file_path=parsed_module.file_path,
            imports=list(parsed_module.imports),
        )

        # Collect local symbols (classes, functions defined in this module)
        for cls in parsed_module.classes:
            result.local_symbols.add(cls.name)
            for method in cls.methods:
                result.local_symbols.add(f"{cls.name}.{method.name}")

        for func in parsed_module.functions:
            result.local_symbols.add(func.name)

        # Collect external symbols from imports
        for imp in parsed_module.imports:
            if imp.import_type == ImportType.IMPORT:
                result.external_symbols.add(imp.module)
            elif imp.import_type == ImportType.FROM_IMPORT:
                for name in imp.names:
                    result.external_symbols.add(f"{imp.module}.{name}")

        # Extract import dependencies
        self._extract_import_dependencies(parsed_module, result)

        # Extract inheritance dependencies
        self._extract_inheritance_dependencies(parsed_module, result)

        # Extract function/method call dependencies
        self._extract_call_dependencies(parsed_module, result)

        # Extract decorator dependencies
        self._extract_decorator_dependencies(parsed_module, result)

        return result

    def extract_multiple(self, parsed_modules: List[ParsedModule]) -> List[ModuleDependencies]:
        """
        Extract dependencies from multiple parsed modules.

        Args:
            parsed_modules: List of ParsedModule objects

        Returns:
            List of ModuleDependencies
        """
        return [self.extract(pm) for pm in parsed_modules]

    def _extract_import_dependencies(self, parsed: ParsedModule, result: ModuleDependencies) -> None:
        """Extract import-related dependencies."""
        for imp in parsed.imports:
            location = imp.location
            line = location.start_line if location else 0

            if imp.import_type == ImportType.IMPORT:
                # import module [as alias]
                target = imp.alias if imp.alias else imp.module
                dep = Dependency(
                    source=parsed.module_name,
                    target=imp.module,
                    dep_type=DependencyType.IMPORT,
                    source_file=parsed.file_path,
                    source_line=line,
                    source_symbol=target,
                    is_relative=imp.is_relative,
                    relative_level=imp.level,
                    confidence=1.0,
                    metadata={"original_names": imp.names}
                )
                result.add_dependency(dep)

            elif imp.import_type == ImportType.FROM_IMPORT:
                # from module import name1, name2
                for name in imp.names:
                    target = f"{imp.module}.{name}"
                    dep = Dependency(
                        source=parsed.module_name,
                        target=target,
                        dep_type=DependencyType.FROM_IMPORT,
                        source_file=parsed.file_path,
                        source_line=line,
                        source_symbol=name,
                        target_symbol=name,
                        is_relative=imp.is_relative,
                        relative_level=imp.level,
                        confidence=1.0,
                        metadata={"module": imp.module, "name": name}
                    )
                    result.add_dependency(dep)

    def _extract_inheritance_dependencies(self, parsed: ParsedModule, result: ModuleDependencies) -> None:
        """Extract class inheritance dependencies."""
        for cls in parsed.classes:
            for base in cls.bases:
                location = cls.location
                line = location.start_line if location else 0

                dep = Dependency(
                    source=cls.name,
                    target=base,
                    dep_type=DependencyType.INHERITANCE,
                    source_file=parsed.file_path,
                    source_line=line,
                    source_symbol=cls.name,
                    target_symbol=base,
                    confidence=1.0,
                    metadata={"class": cls.name, "base": base}
                )
                result.add_dependency(dep)

    def _extract_call_dependencies(self, parsed: ParsedModule, result: ModuleDependencies) -> None:
        """Extract function and method call dependencies."""
        # Process standalone functions
        for func in parsed.functions:
            self._extract_calls_from_function(func, parsed, result)

        # Process class methods
        for cls in parsed.classes:
            for method in cls.methods:
                self._extract_calls_from_function(method, parsed, result)

    def _extract_calls_from_function(self, func: FunctionInfo, parsed: ParsedModule, result: ModuleDependencies) -> None:
        """Extract call dependencies from a single function/method."""
        location = func.location
        base_line = location.start_line if location else 0

        for call_name in func.calls:
            # Determine if this is a method call (has object) or function call
            # Since we only have the name from the AST, we infer from context
            dep_type = DependencyType.FUNCTION_CALL
            target = call_name

            # Check if call is to a local function
            # result.local_symbols contains all locally defined symbols
            is_local = call_name in result.local_symbols

            # Check if it looks like a method call (e.g., self.method, obj.method)
            # The AST visitor only captures the method name, so we can't distinguish
            # perfectly without more context. We'll mark as function_call by default.
            # The resolver can refine this later.

            dep = Dependency(
                source=func.qualified_name,
                target=call_name,
                dep_type=dep_type,
                source_file=parsed.file_path,
                source_line=base_line,
                source_symbol=func.qualified_name,
                target_symbol=call_name,
                confidence=0.8 if not is_local else 1.0,
                metadata={
                    "caller": func.qualified_name,
                    "is_local": is_local,
                    "caller_is_method": func.is_method,
                    "caller_class": func.parent_class
                }
            )
            result.add_dependency(dep)

    def _extract_decorator_dependencies(self, parsed: ParsedModule, result: ModuleDependencies) -> None:
        """Extract decorator dependencies."""
        # Decorators on functions
        for func in parsed.functions:
            for decorator in func.decorators:
                location = func.location
                line = location.start_line if location else 0

                dep = Dependency(
                    source=func.qualified_name,
                    target=decorator,
                    dep_type=DependencyType.DECORATOR,
                    source_file=parsed.file_path,
                    source_line=line,
                    source_symbol=func.qualified_name,
                    target_symbol=decorator,
                    confidence=0.9,
                    metadata={"function": func.qualified_name, "decorator": decorator}
                )
                result.add_dependency(dep)

        # Decorators on classes
        for cls in parsed.classes:
            for decorator in cls.decorators:
                location = cls.location
                line = location.start_line if location else 0

                dep = Dependency(
                    source=cls.name,
                    target=decorator,
                    dep_type=DependencyType.DECORATOR,
                    source_file=parsed.file_path,
                    source_line=line,
                    source_symbol=cls.name,
                    target_symbol=decorator,
                    confidence=0.9,
                    metadata={"class": cls.name, "decorator": decorator}
                )
                result.add_dependency(dep)

        # Decorators on methods
        for cls in parsed.classes:
            for method in cls.methods:
                for decorator in method.decorators:
                    location = method.location
                    line = location.start_line if location else 0

                    dep = Dependency(
                        source=method.qualified_name,
                        target=decorator,
                        dep_type=DependencyType.DECORATOR,
                        source_file=parsed.file_path,
                        source_line=line,
                        source_symbol=method.qualified_name,
                        target_symbol=decorator,
                        confidence=0.9,
                        metadata={"method": method.qualified_name, "decorator": decorator}
                    )
                    result.add_dependency(dep)


def extract_dependencies(parsed_module: ParsedModule,
                         project_root: Optional[Path] = None) -> ModuleDependencies:
    """Convenience function to extract dependencies from a single module."""
    extractor = DependencyExtractor(project_root=project_root)
    return extractor.extract(parsed_module)


def extract_dependencies_multiple(parsed_modules: List[ParsedModule],
                                  project_root: Optional[Path] = None) -> List[ModuleDependencies]:
    """Convenience function to extract dependencies from multiple modules."""
    extractor = DependencyExtractor(project_root=project_root)
    return extractor.extract_multiple(parsed_modules)