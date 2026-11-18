"""
Tests for the Dependency Extractor module.
"""

import pytest
from analyzer.ast_parser import parse_source
from analyzer.dependency_extractor import (
    DependencyExtractor,
    extract_dependencies,
    extract_dependencies_multiple,
    ModuleDependencies,
    Dependency,
    DependencyType,
)


class TestDependencyExtractor:
    """Test cases for DependencyExtractor."""

    def test_simple_imports(self):
        """Test extraction of simple import dependencies."""
        source = "import os\nimport sys as system\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        imports = result.get_imports()
        assert len(imports) == 2

        # First import
        imp1 = imports[0]
        assert imp1.dep_type == DependencyType.IMPORT
        assert imp1.source == "test"
        assert imp1.target == "os"
        assert imp1.source_file == "test.py"
        assert imp1.source_line == 1
        assert imp1.is_relative is False

        # Second import with alias
        imp2 = imports[1]
        assert imp2.dep_type == DependencyType.IMPORT
        assert imp2.target == "sys"
        assert imp2.source_symbol == "system"  # alias

    def test_from_imports(self):
        """Test extraction of from-import dependencies."""
        source = "from typing import List, Optional\nfrom .local import helper\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        imports = result.get_imports()
        assert len(imports) == 3  # List, Optional, helper

        # typing imports
        typing_imports = [i for i in imports if i.target.startswith("typing.")]
        assert len(typing_imports) == 2
        assert {i.target_symbol for i in typing_imports} == {"List", "Optional"}

        # Relative import
        rel_import = next(i for i in imports if i.target == ".local.helper")
        assert rel_import.dep_type == DependencyType.FROM_IMPORT
        assert rel_import.is_relative is True
        assert rel_import.relative_level == 1

    def test_relative_imports(self):
        """Test that relative imports preserve level information."""
        source = "from .. import parent_module\nfrom ...pkg import deep\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        imports = result.get_imports()
        assert len(imports) == 2

        assert imports[0].relative_level == 2
        assert imports[1].relative_level == 3

    def test_inheritance(self):
        """Test extraction of class inheritance dependencies."""
        source = '''
class BaseClass:
    pass

class ChildClass(BaseClass):
    pass

class GrandChild(ChildClass):
    pass
'''
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        inheritances = result.get_by_type(DependencyType.INHERITANCE)
        assert len(inheritances) == 2

        child_to_base = next(i for i in inheritances if i.source == "ChildClass")
        assert child_to_base.target == "BaseClass"
        assert child_to_base.dep_type == DependencyType.INHERITANCE

        grand_to_child = next(i for i in inheritances if i.source == "GrandChild")
        assert grand_to_child.target == "ChildClass"

    def test_function_calls(self):
        """Test extraction of function call dependencies."""
        source = '''
def helper():
    pass

def process(data):
    cleaned = clean(data)
    return helper()

def clean(x):
    return x.strip()
'''
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        calls = result.get_by_type(DependencyType.FUNCTION_CALL)
        call_names = {c.target for c in calls}

        assert "clean" in call_names
        assert "helper" in call_names
        assert "strip" in call_names

        # Verify source function is recorded
        process_calls = [c for c in calls if c.source == "process"]
        assert len(process_calls) == 2

    def test_method_calls(self):
        """Test extraction of method calls within classes."""
        source = '''
class MyClass:
    def method_one(self):
        return 1

    def method_two(self):
        return self.method_one()

    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass
'''
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        calls = result.get_by_type(DependencyType.FUNCTION_CALL)
        call_names = {c.target for c in calls}

        assert "method_one" in call_names

        # Verify source is qualified name
        method_calls = [c for c in calls if c.source == "MyClass.method_two"]
        assert len(method_calls) >= 1

    def test_decorators(self):
        """Test extraction of decorator dependencies."""
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
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        decorators = result.get_by_type(DependencyType.DECORATOR)
        decorator_names = {d.target for d in decorators}

        assert "property" in decorator_names
        assert "staticmethod" in decorator_names
        assert "classmethod" in decorator_names

    def test_multiple_dependency_types(self):
        """Test that all dependency types are extracted from a complex module."""
        source = '''
import os
from typing import List

class Base:
    pass

class Derived(Base):
    @property
    def processed(self):
        return True

    def process(self, items: List[int]) -> int:
        total = sum(items)
        return self.validate(total)

    def validate(self, value: int) -> bool:
        return value > 0

@staticmethod
def utility():
    return "static"

def standalone():
    d = Derived()
    return d.process([1, 2, 3])
'''
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        # Check all types present
        types = {d.dep_type for d in result.dependencies}
        assert DependencyType.IMPORT in types
        assert DependencyType.FROM_IMPORT in types
        assert DependencyType.INHERITANCE in types
        assert DependencyType.FUNCTION_CALL in types
        assert DependencyType.DECORATOR in types

        # Verify counts
        assert len(result.get_by_type(DependencyType.INHERITANCE)) == 1
        assert len(result.get_by_type(DependencyType.IMPORT)) >= 1
        assert len(result.get_by_type(DependencyType.FROM_IMPORT)) >= 1

    def test_source_locations(self):
        """Test that source locations are populated correctly."""
        source = "import os\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        imp = result.get_imports()[0]
        assert imp.source_file == "test.py"
        assert imp.source_line == 1

    def test_empty_module(self):
        """Test extraction from empty module."""
        source = ""
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        assert len(result.dependencies) == 0
        assert len(result.local_symbols) == 0
        assert len(result.external_symbols) == 0

    def test_syntax_error_module(self):
        """Test extraction from module with syntax errors."""
        source = "def broken(\n    pass\n"
        parsed = parse_source(source, "test.py")

        assert len(parsed.syntax_errors) > 0

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        # Should not crash, just return empty dependencies
        assert isinstance(result, ModuleDependencies)

    def test_local_symbols_collected(self):
        """Test that local symbols are correctly identified."""
        source = '''
import external

class MyClass:
    def method(self):
        pass

def my_function():
    pass

CONSTANT = 42
'''
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        assert "MyClass" in result.local_symbols
        assert "MyClass.method" in result.local_symbols
        assert "my_function" in result.local_symbols

    def test_external_symbols_collected(self):
        """Test that external symbols from imports are collected."""
        source = "import os\nfrom typing import List, Optional\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        assert "os" in result.external_symbols
        assert "typing.List" in result.external_symbols
        assert "typing.Optional" in result.external_symbols

    def test_module_dependencies_structure(self):
        """Test ModuleDependencies structure."""
        source = "import os\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        assert result.module_name == "test"
        assert result.file_path == "test.py"
        assert len(result.dependencies) == 1
        assert len(result.imports) == 1

    def test_get_by_type(self):
        """Test ModuleDependencies.get_by_type filter."""
        source = '''
import os
from sys import path

class A:
    pass

class B(A):
    pass

def foo():
    bar()

def bar():
    pass
'''
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        imports = result.get_by_type(DependencyType.IMPORT)
        from_imports = result.get_by_type(DependencyType.FROM_IMPORT)
        inheritances = result.get_by_type(DependencyType.INHERITANCE)
        calls = result.get_by_type(DependencyType.FUNCTION_CALL)

        assert len(imports) == 1
        assert len(from_imports) == 1
        assert len(inheritances) == 1
        assert len(calls) >= 1

    def test_multiple_modules(self):
        """Test extracting from multiple modules."""
        source1 = "import os\ndef func1():\n    pass\n"
        source2 = "import sys\nfrom typing import List\n"

        parsed1 = parse_source(source1, "mod1.py")
        parsed2 = parse_source(source2, "mod2.py")

        extractor = DependencyExtractor()
        results = extractor.extract_multiple([parsed1, parsed2])

        assert len(results) == 2
        assert results[0].module_name == "mod1"
        assert results[1].module_name == "mod2"

    def test_dependency_confidence(self):
        """Test that confidence values are set."""
        source = '''
import os

def func():
    os.path.join()
    unknown_func()
'''
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        for dep in result.dependencies:
            assert 0.0 <= dep.confidence <= 1.0

    def test_dependency_metadata(self):
        """Test that metadata is populated."""
        source = "from utils import helper\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        imp = result.get_by_type(DependencyType.FROM_IMPORT)[0]
        assert "module" in imp.metadata
        assert "name" in imp.metadata
        assert imp.metadata["module"] == "utils"
        assert imp.metadata["name"] == "helper"

    def test_inheritance_metadata(self):
        """Test inheritance dependency metadata."""
        source = "class Child(Base):\n    pass\n"
        parsed = parse_source(source, "test.py")

        extractor = DependencyExtractor()
        result = extractor.extract(parsed)

        inh = result.get_by_type(DependencyType.INHERITANCE)[0]
        assert inh.metadata["class"] == "Child"
        assert inh.metadata["base"] == "Base"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_extract_dependencies(self):
        """Test extract_dependencies convenience function."""
        source = "import os\n"
        parsed = parse_source(source, "test.py")

        result = extract_dependencies(parsed)

        assert isinstance(result, ModuleDependencies)
        assert len(result.dependencies) == 1

    def test_extract_dependencies_multiple(self):
        """Test extract_dependencies_multiple convenience function."""
        source1 = "import os\n"
        source2 = "import sys\n"

        parsed1 = parse_source(source1, "mod1.py")
        parsed2 = parse_source(source2, "mod2.py")

        results = extract_dependencies_multiple([parsed1, parsed2])

        assert len(results) == 2
        assert all(isinstance(r, ModuleDependencies) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])