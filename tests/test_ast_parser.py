"""
Tests for the AST Parser module.
"""

import pytest
from analyzer.ast_parser import (
    ASTParser,
    parse_source,
    parse_file,
    ParsedModule,
    ImportInfo,
    ClassInfo,
    FunctionInfo,
    ImportType,
    SourceLocation,
)


class TestASTParser:
    """Test cases for ASTParser."""

    def test_simple_imports(self):
        """Test detection of simple imports."""
        source = "import os\nimport sys as system\n"
        result = parse_source(source, "test.py")

        assert len(result.imports) == 2
        assert result.imports[0].import_type == ImportType.IMPORT
        assert result.imports[0].module == "os"
        assert result.imports[0].names == ["os"]
        assert result.imports[0].alias is None

        assert result.imports[1].import_type == ImportType.IMPORT
        assert result.imports[1].module == "sys"
        assert result.imports[1].alias == "system"

    def test_from_imports(self):
        """Test detection of from-imports."""
        source = "from typing import List, Optional\nfrom .local_module import helper\n"
        result = parse_source(source, "test.py")

        assert len(result.imports) == 3  # Two from first, one from second

        # First from-import creates two ImportInfo entries
        assert result.imports[0].import_type == ImportType.FROM_IMPORT
        assert result.imports[0].module == "typing"
        assert result.imports[0].names == ["List"]
        assert result.imports[0].is_relative is False

        assert result.imports[1].import_type == ImportType.FROM_IMPORT
        assert result.imports[1].module == "typing"
        assert result.imports[1].names == ["Optional"]

        # Second from-import
        assert result.imports[2].import_type == ImportType.FROM_IMPORT
        assert result.imports[2].module == ".local_module"
        assert result.imports[2].names == ["helper"]
        assert result.imports[2].is_relative is True
        assert result.imports[2].level == 1

    def test_function_detection(self):
        """Test detection of standalone functions."""
        source = '''
def hello(x):
    print(x)
    return x + 1

def add(a: int, b: int = 10) -> int:
    return a + b
'''
        result = parse_source(source, "test.py")

        assert len(result.functions) == 2
        func1 = result.functions[0]
        assert func1.name == "hello"
        assert func1.args == ["x"]
        assert func1.returns is None  # No return annotation
        assert "print" in func1.calls

        func2 = result.functions[1]
        assert func2.name == "add"
        assert func2.args == ["a", "b"]
        assert func2.returns == "int"

    def test_class_and_inheritance(self):
        """Test detection of classes and inheritance."""
        source = '''
class BaseClass:
    def base_method(self):
        pass

class ChildClass(BaseClass):
    def child_method(self):
        self.base_method()
'''
        result = parse_source(source, "test.py")

        assert len(result.classes) == 2

        base = result.classes[0]
        assert base.name == "BaseClass"
        assert base.bases == []
        assert len(base.methods) == 1
        assert base.methods[0].name == "base_method"

        child = result.classes[1]
        assert child.name == "ChildClass"
        assert child.bases == ["BaseClass"]
        assert len(child.methods) == 1
        assert child.methods[0].name == "child_method"
        assert "base_method" in child.methods[0].calls

    def test_function_calls_detected(self):
        """Test that function calls are extracted."""
        source = '''
def process(data):
    cleaned = clean(data)
    validated = validate(cleaned)
    return validated

def clean(x):
    return x.strip()

def validate(y):
    return bool(y)
'''
        result = parse_source(source, "test.py")

        process_func = next(f for f in result.functions if f.name == "process")
        assert "clean" in process_func.calls
        assert "validate" in process_func.calls

    def test_assignments_detected(self):
        """Test detection of assignments."""
        source = '''
CONSTANT = 42
x = 10
y: int = 20
'''
        result = parse_source(source, "test.py")

        assert len(result.assignments) == 3
        # CONSTANT = 42
        assert result.assignments[0].targets == ["CONSTANT"]
        assert result.assignments[0].is_annotation is False
        # x = 10
        assert result.assignments[1].targets == ["x"]
        assert result.assignments[1].is_annotation is False
        # y: int = 20
        assert result.assignments[2].targets == ["y"]
        assert result.assignments[2].is_annotation is True

    def test_no_syntax_errors_for_valid_code(self):
        """Test that valid code produces no syntax errors."""
        source = "x = 1\ny = 2\n"
        result = parse_source(source, "test.py")

        assert result.syntax_errors == []

    def test_syntax_error_handled_gracefully(self):
        """Test that syntax errors are recorded but don't crash."""
        source = "def broken(\n    pass\n"
        result = parse_source(source, "test.py")

        assert len(result.syntax_errors) > 0
        assert "Syntax error" in result.syntax_errors[0]

    def test_decorators_detected(self):
        """Test detection of decorators on functions and classes."""
        source = '''
@property
def computed(self):
    return self.value * 2

@staticmethod
def utility():
    return "static"

class MyClass:
    @classmethod
    def from_string(cls, s):
        return cls(s)
'''
        result = parse_source(source, "test.py")

        # Check property decorator
        computed = next(f for f in result.functions if f.name == "computed")
        assert "property" in computed.decorators

        # Check staticmethod
        utility = next(f for f in result.functions if f.name == "utility")
        assert "staticmethod" in utility.decorators
        assert utility.is_static is True

        # Check classmethod on method
        cls_method = result.classes[0].methods[0]
        assert cls_method.name == "from_string"
        assert "classmethod" in cls_method.decorators
        assert cls_method.is_classmethod is True

    def test_source_locations_populated(self):
        """Test that source locations are populated."""
        source = "def foo():\n    pass\n"
        result = parse_source(source, "test.py")

        func = result.functions[0]
        assert func.location is not None
        assert func.location.start_line == 1
        assert func.location.end_line == 2
        assert func.location.file_path == "test.py"

    def test_method_attached_to_class(self):
        """Test that methods are correctly stored inside ClassInfo.methods."""
        source = '''
class MyClass:
    def method_one(self):
        return 1

    def method_two(self):
        return self.method_one()
'''
        result = parse_source(source, "test.py")

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "MyClass"
        assert len(cls.methods) == 2

        method_names = [m.name for m in cls.methods]
        assert "method_one" in method_names
        assert "method_two" in method_names

        # Verify they are NOT in standalone functions
        standalone_names = [f.name for f in result.functions]
        assert "method_one" not in standalone_names
        assert "method_two" not in standalone_names

    def test_qualified_name(self):
        """Test FunctionInfo.qualified_name property."""
        source = '''
class MyClass:
    def my_method(self):
        pass

def standalone():
    pass
'''
        result = parse_source(source, "test.py")

        method = result.classes[0].methods[0]
        assert method.qualified_name == "MyClass.my_method"

        func = result.functions[0]
        assert func.qualified_name == "standalone"

    def test_get_all_functions(self):
        """Test ParsedModule.get_all_functions()."""
        source = '''
def standalone():
    pass

class MyClass:
    def method(self):
        pass
'''
        result = parse_source(source, "test.py")

        all_funcs = result.get_all_functions()
        assert len(all_funcs) == 2
        names = {f.name for f in all_funcs}
        assert names == {"standalone", "method"}

    def test_get_imported_modules(self):
        """Test ParsedModule.get_imported_modules()."""
        source = "import os\nfrom sys import path\n"
        result = parse_source(source, "test.py")

        modules = result.get_imported_modules()
        assert "os" in modules
        assert "sys" in modules

    def test_async_functions(self):
        """Test detection of async functions."""
        source = '''
async def async_func():
    await something()

def sync_func():
    pass
'''
        result = parse_source(source, "test.py")

        async_func = next(f for f in result.functions if f.name == "async_func")
        assert async_func.is_async is True

        sync_func = next(f for f in result.functions if f.name == "sync_func")
        assert sync_func.is_async is False


class TestParseFile:
    """Test file parsing functionality."""

    def test_parse_file_creates_module(self, tmp_path):
        """Test parsing a real file."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        result = parse_file(test_file)

        assert isinstance(result, ParsedModule)
        assert result.module_name == "test_module"
        assert len(result.functions) == 1
        assert result.functions[0].name == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])