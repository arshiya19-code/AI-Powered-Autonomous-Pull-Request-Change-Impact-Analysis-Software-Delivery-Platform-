"""
AST Parser - Parses Python source files using the built-in `ast` module
and extracts structured information for dependency analysis.

This module provides the foundation for the custom dependency extraction pipeline.
It does NOT use NetworkX, Neo4j, GAT, or any external graph libraries.
"""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Set, Any
from enum import Enum


class ImportType(Enum):
    """Type of import statement."""
    IMPORT = "import"           # import module
    FROM_IMPORT = "from_import"  # from module import name


class NodeType(Enum):
    """Type of AST node for classification."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    CALL = "call"
    ASSIGNMENT = "assignment"
    DECORATOR = "decorator"


@dataclass
class SourceLocation:
    """Source code location information."""
    file_path: str
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class ImportInfo:
    """Information about an import statement."""
    import_type: ImportType
    module: str                    # The module being imported from
    names: List[str]               # The names being imported
    alias: Optional[str] = None    # Alias if `as` is used (for single name imports)
    location: Optional[SourceLocation] = None
    is_relative: bool = False      # True for relative imports (from . import)
    level: int = 0                 # Relative import level (0 = absolute)

    @property
    def imported_names(self) -> List[str]:
        """Get the actual names being imported (handles `import x as y`)."""
        return self.names


@dataclass
class ClassInfo:
    """Information about a class definition."""
    name: str
    bases: List[str]               # Base class names (as strings)
    decorators: List[str]          # Decorator names
    methods: List['FunctionInfo'] = field(default_factory=list)
    location: Optional[SourceLocation] = None
    parent_class: Optional[str] = None  # For nested classes
    docstring: Optional[str] = None

    def is_subclass_of(self, class_name: str) -> bool:
        """Check if this class inherits from the given class."""
        return class_name in self.bases


@dataclass
class FunctionInfo:
    """Information about a function or method definition."""
    name: str
    args: List[str]                # Parameter names
    defaults: List[Any] = field(default_factory=list)  # Default values
    returns: Optional[str] = None  # Return type annotation (as string)
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)  # Function calls made
    location: Optional[SourceLocation] = None
    parent_class: Optional[str] = None  # If method, the class name
    is_async: bool = False
    is_method: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    docstring: Optional[str] = None

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name including parent class if applicable."""
        if self.parent_class:
            return f"{self.parent_class}.{self.name}"
        return self.name


@dataclass
class CallInfo:
    """Information about a function/method call."""
    func_name: str                 # Name of function being called
    args_count: int = 0
    keywords: List[str] = field(default_factory=list)
    location: Optional[SourceLocation] = None
    caller: Optional[str] = None   # Containing function/method name
    is_method_call: bool = False   # True if obj.method()
    object_name: Optional[str] = None  # Object on which method is called


@dataclass
class AssignmentInfo:
    """Information about an assignment."""
    targets: List[str]             # Variable names being assigned
    value_type: Optional[str] = None  # Type of assigned value (inferred)
    location: Optional[SourceLocation] = None
    is_annotation: bool = False    # True for type annotations


@dataclass
class DecoratorInfo:
    """Information about a decorator."""
    name: str
    args: List[str] = field(default_factory=list)
    location: Optional[SourceLocation] = None
    target: Optional[str] = None   # What is being decorated


@dataclass
class ParsedModule:
    """Complete parsed representation of a Python module/file."""
    file_path: str
    module_name: str
    imports: List[ImportInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    assignments: List[AssignmentInfo] = field(default_factory=list)
    decorators: List[DecoratorInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    source_lines: int = 0
    syntax_errors: List[str] = field(default_factory=list)
    parse_time_ms: float = 0.0

    def get_all_functions(self) -> List[FunctionInfo]:
        """Get all functions including methods from classes."""
        all_funcs = list(self.functions)
        for cls in self.classes:
            all_funcs.extend(cls.methods)
        return all_funcs

    def get_all_calls(self) -> List[CallInfo]:
        """Get all function calls from functions and methods."""
        calls = []
        for func in self.get_all_functions():
            calls.extend([CallInfo(func_name=c, caller=func.name, location=func.location)
                          for c in func.calls])
        return calls

    def get_imported_modules(self) -> Set[str]:
        """Get set of all imported module names."""
        modules = set()
        for imp in self.imports:
            modules.add(imp.module)
        return modules


class ASTParser:
    """
    Parses Python source files and extracts structured information.

    Uses Python's built-in `ast` module for parsing.
    Designed to be consumed by the dependency extractor.
    """

    def __init__(self, include_docstrings: bool = True, include_calls: bool = True):
        """
        Initialize the AST parser.

        Args:
            include_docstrings: Whether to extract docstrings
            include_calls: Whether to extract function calls
        """
        self.include_docstrings = include_docstrings
        self.include_calls = include_calls

    def parse_file(self, file_path: Path) -> ParsedModule:
        """
        Parse a Python file from disk.

        Args:
            file_path: Path to the Python file

        Returns:
            ParsedModule with extracted information
        """
        import time
        start_time = time.perf_counter()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    source = f.read()
            except Exception as e:
                return self._create_error_module(str(file_path), f"Failed to read file: {e}")
        except Exception as e:
            return self._create_error_module(str(file_path), f"Failed to read file: {e}")

        return self.parse_source(source, str(file_path), start_time)

    def parse_source(self, source: str, file_path: str = "<string>",
                     start_time: Optional[float] = None) -> ParsedModule:
        """
        Parse Python source code string.

        Args:
            source: Python source code
            file_path: Path for error reporting (default: "<string>")
            start_time: Optional start time for parse_time_ms calculation

        Returns:
            ParsedModule with extracted information
        """
        import time
        if start_time is None:
            start_time = time.perf_counter()

        module = ParsedModule(
            file_path=file_path,
            module_name=self._get_module_name(file_path),
            source_lines=len(source.splitlines())
        )

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            module.syntax_errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            module.parse_time_ms = (time.perf_counter() - start_time) * 1000
            return module
        except Exception as e:
            module.syntax_errors.append(f"Parse error: {type(e).__name__}: {e}")
            module.parse_time_ms = (time.perf_counter() - start_time) * 1000
            return module

        # Extract module docstring
        if self.include_docstrings:
            module.docstring = ast.get_docstring(tree)

        # Create visitor and traverse
        visitor = _ASTVisitor(
            file_path=file_path,
            include_docstrings=self.include_docstrings,
            include_calls=self.include_calls
        )
        visitor.visit(tree)

        # Transfer results
        module.imports = visitor.imports
        module.classes = visitor.classes
        module.functions = visitor.functions
        module.assignments = visitor.assignments
        module.decorators = visitor.decorators
        module.parse_time_ms = (time.perf_counter() - start_time) * 1000

        return module

    def _create_error_module(self, file_path: str, error: str) -> ParsedModule:
        """Create a module with error information."""
        module = ParsedModule(file_path=file_path, module_name=self._get_module_name(file_path))
        module.syntax_errors.append(error)
        return module

    def _get_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        path = Path(file_path)
        if path.suffix == '.py':
            return path.stem
        return path.name


class _ASTVisitor(ast.NodeVisitor):
    """Internal AST visitor that extracts structured information."""

    def __init__(self, file_path: str, include_docstrings: bool = True,
                 include_calls: bool = True):
        self.file_path = file_path
        self.include_docstrings = include_docstrings
        self.include_calls = include_calls

        # Results
        self.imports: List[ImportInfo] = []
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        self.assignments: List[AssignmentInfo] = []
        self.decorators: List[DecoratorInfo] = []

        # State tracking
        self._current_class: Optional[str] = None
        self._current_class_info: Optional[ClassInfo] = None
        self._current_function: Optional[str] = None
        self._parent_stack: List[str] = []  # For nested structures

    def _make_location(self, node: ast.AST) -> SourceLocation:
        """Create SourceLocation from AST node."""
        return SourceLocation(
            file_path=self.file_path,
            start_line=getattr(node, 'lineno', 0),
            end_line=getattr(node, 'end_lineno', 0),
            start_col=getattr(node, 'col_offset', 0),
            end_col=getattr(node, 'end_col_offset', 0)
        )

    def _get_decorator_names(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> List[str]:
        """Extract decorator names from a function or class."""
        names = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                names.append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                names.append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                names.append(decorator.attr)
        return names

    def _get_base_names(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names from a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        return bases

    def _get_arg_names(self, node: ast.arguments) -> List[str]:
        """Extract argument names from function arguments."""
        args = []
        for arg in node.args:
            args.append(arg.arg)
        # Handle *args and **kwargs
        if node.vararg:
            args.append(f"*{node.vararg.arg}")
        if node.kwarg:
            args.append(f"**{node.kwarg.arg}")
        return args

    def _get_return_annotation(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
        """Get return type annotation as string."""
        if node.returns:
            return ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        return None

    def visit_Import(self, node: ast.Import) -> None:
        """Handle `import module` statements."""
        for alias in node.names:
            self.imports.append(ImportInfo(
                import_type=ImportType.IMPORT,
                module=alias.name,
                names=[alias.name],
                alias=alias.asname,
                location=self._make_location(node)
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle `from module import name` statements."""
        module_name = node.module or ""
        if node.level > 0:
            module_name = "." * node.level + (module_name if module_name else "")

        names = [alias.name for alias in node.names]
        for alias in node.names:
            self.imports.append(ImportInfo(
                import_type=ImportType.FROM_IMPORT,
                module=module_name,
                names=[alias.name],
                alias=alias.asname,
                location=self._make_location(node),
                is_relative=node.level > 0,
                level=node.level
            ))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions."""
        class_info = ClassInfo(
            name=node.name,
            bases=self._get_base_names(node),
            decorators=self._get_decorator_names(node),
            location=self._make_location(node),
            parent_class=self._current_class,
            docstring=ast.get_docstring(node) if self.include_docstrings else None
        )

        # Add class to list BEFORE visiting body so methods can find it
        self.classes.append(class_info)

        # Track nested classes
        old_class = self._current_class
        old_class_info = getattr(self, '_current_class_info', None)
        self._current_class = node.name
        self._current_class_info = class_info
        self._parent_stack.append(node.name)

        # Visit class body to extract methods
        for item in node.body:
            self.visit(item)

        # Restore state
        self._parent_stack.pop()
        self._current_class = old_class
        self._current_class_info = old_class_info

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function definitions (sync)."""
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        """Common logic for visiting function definitions."""
        decorators = self._get_decorator_names(node)
        is_static = 'staticmethod' in decorators
        is_classmethod = 'classmethod' in decorators
        # classmethod IS a method (just a class method), only staticmethod is not a method
        is_method = self._current_class is not None and not is_static

        func_info = FunctionInfo(
            name=node.name,
            args=self._get_arg_names(node.args),
            decorators=decorators,
            location=self._make_location(node),
            parent_class=self._current_class if is_method else None,
            is_async=is_async,
            is_method=is_method,
            is_static=is_static,
            is_classmethod=is_classmethod,
            docstring=ast.get_docstring(node) if self.include_docstrings else None
        )

        # Extract return annotation
        func_info.returns = self._get_return_annotation(node)

        # Track current function for call extraction
        old_function = self._current_function
        self._current_function = func_info.qualified_name

        # Visit function body
        if self.include_calls:
            self._extract_calls(node, func_info)

        for item in node.body:
            if not isinstance(item, (ast.Expr, ast.Call)):  # Already handled calls
                self.visit(item)

        self._current_function = old_function

        # Add to appropriate collection
        if is_method and self._current_class_info:
            self._current_class_info.methods.append(func_info)
        else:
            self.functions.append(func_info)

    def _extract_calls(self, node: ast.AST, func_info: FunctionInfo) -> None:
        """Extract function calls from a function body."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name:
                    func_info.calls.append(call_name)

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the name of a function being called."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle assignments."""
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, ast.Attribute):
                targets.append(target.attr)
            elif isinstance(target, ast.Subscript):
                # e.g., obj[key] = value - skip for now
                pass

        if targets:
            self.assignments.append(AssignmentInfo(
                targets=targets,
                location=self._make_location(node)
            ))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments (type hints)."""
        target = ""
        if isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node.target, ast.Attribute):
            target = node.target.attr

        if target:
            self.assignments.append(AssignmentInfo(
                targets=[target],
                location=self._make_location(node),
                is_annotation=True
            ))
        self.generic_visit(node)


def parse_file(file_path: Path) -> ParsedModule:
    """Convenience function to parse a single file."""
    parser = ASTParser()
    return parser.parse_file(file_path)


def parse_source(source: str, file_path: str = "<string>") -> ParsedModule:
    """Convenience function to parse source code."""
    parser = ASTParser()
    return parser.parse_source(source, file_path)