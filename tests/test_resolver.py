"""
Tests for the Custom Name/Module Resolver (Phase 3).

Covers:
  1.  Simple module resolution
  2.  Absolute import resolution
  3.  From-import resolution
  4.  Aliased import resolution
  5.  Relative imports (single level)
  6.  Multi-level relative imports
  7.  Internal vs external dependency classification
  8.  Function symbol resolution
  9.  Class symbol resolution
  10. Method symbol resolution
  11. Local function call resolution
  12. Cross-module function call resolution
  13. Unknown / unresolved symbols
  14. Package / __init__.py module resolution
  15. Multiple modules resolved together
  16. Confidence values
  17. Resolution metadata (reason strings)
  18. No crashes on syntax-error modules

All tests create temporary files/directories so they are fully isolated
from the real project structure.
"""

import pytest
from pathlib import Path

from analyzer.ast_parser import ASTParser, parse_source
from analyzer.dependency_extractor import DependencyExtractor, DependencyType
from analyzer.resolver import (
    NameResolver,
    ModuleDiscovery,
    SymbolTableBuilder,
    ResolvedDependency,
    ResolutionResult,
    ProjectResolutionResult,
    ResolutionStatus,
    Symbol,
    ModuleInfo,
    resolve_dependencies,
    resolve_single_module,
    _build_import_lookup,
    _is_stdlib,
    _is_external_module,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_file(tmp_path: Path, rel: str, content: str) -> Path:
    """Write a file and return its Path."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def parse_and_extract(source: str, file_path: str):
    """Parse source and extract deps; return (ParsedModule, ModuleDependencies)."""
    parser = ASTParser()
    pm = parser.parse_source(source, file_path)
    extractor = DependencyExtractor()
    md = extractor.extract(pm)
    return pm, md


def make_project(tmp_path: Path, files: dict) -> dict:
    """
    Write multiple files and return {rel_path: absolute_Path}.

    ``files`` is a dict mapping relative path strings to source strings.
    """
    result = {}
    for rel, content in files.items():
        result[rel] = write_file(tmp_path, rel, content)
    return result


# ===========================================================================
# 1. Simple module resolution
# ===========================================================================

class TestSimpleModuleResolution:

    def test_discovers_top_level_modules(self, tmp_path):
        """ModuleDiscovery finds top-level .py files."""
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        write_file(tmp_path, "app.py", "import utils\n")

        discovery = ModuleDiscovery(tmp_path)
        modules = discovery.discover()

        assert "utils" in modules
        assert "app" in modules

    def test_module_file_path_is_correct(self, tmp_path):
        write_file(tmp_path, "utils.py", "")
        discovery = ModuleDiscovery(tmp_path)
        modules = discovery.discover()

        assert modules["utils"].file_path == tmp_path / "utils.py"

    def test_excludes_venv_and_cache(self, tmp_path):
        write_file(tmp_path, ".venv/lib/some.py", "")
        write_file(tmp_path, "__pycache__/module.py", "")
        write_file(tmp_path, "real_module.py", "")

        discovery = ModuleDiscovery(tmp_path)
        modules = discovery.discover()

        assert "real_module" in modules
        keys = list(modules.keys())
        assert not any(".venv" in k or "__pycache__" in k for k in keys)


# ===========================================================================
# 2. Absolute import resolution
# ===========================================================================

class TestAbsoluteImportResolution:

    def test_import_stdlib_is_external(self, tmp_path):
        write_file(tmp_path, "app.py", "import os\n")
        pm, md = parse_and_extract("import os\n", str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        dep = result.resolved_dependencies[0]
        assert dep.status == ResolutionStatus.EXTERNAL

    def test_import_internal_module(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "import utils\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        import_deps = [d for d in result.resolved_dependencies
                       if d.dep_type == DependencyType.IMPORT]
        assert len(import_deps) == 1
        dep = import_deps[0]
        assert dep.status == ResolutionStatus.RESOLVED
        assert dep.resolved_module == "utils"

    def test_import_third_party_is_external(self, tmp_path):
        source = "import requests\n"
        write_file(tmp_path, "app.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        dep = result.resolved_dependencies[0]
        assert dep.status == ResolutionStatus.EXTERNAL


# ===========================================================================
# 3. From-import resolution
# ===========================================================================

class TestFromImportResolution:

    def test_from_import_internal_function(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "from utils import helper\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        from_imports = [d for d in result.resolved_dependencies
                        if d.dep_type == DependencyType.FROM_IMPORT]
        assert len(from_imports) == 1
        dep = from_imports[0]
        assert dep.status == ResolutionStatus.RESOLVED
        assert dep.resolved_module == "utils"
        assert dep.resolved_symbol == "utils.helper"

    def test_from_import_resolves_class(self, tmp_path):
        write_file(tmp_path, "models.py", "class User:\n    pass\n")
        source = "from models import User\n"
        write_file(tmp_path, "app.py", source)

        models_pm, _ = parse_and_extract("class User:\n    pass\n", str(tmp_path / "models.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([models_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        dep = [d for d in result.resolved_dependencies
               if d.dep_type == DependencyType.FROM_IMPORT][0]
        assert dep.status == ResolutionStatus.RESOLVED
        assert dep.resolved_symbol == "models.User"

    def test_from_import_stdlib_is_external(self, tmp_path):
        source = "from typing import List, Optional\n"
        write_file(tmp_path, "app.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        from_imports = [d for d in result.resolved_dependencies
                        if d.dep_type == DependencyType.FROM_IMPORT]
        assert all(d.status == ResolutionStatus.EXTERNAL for d in from_imports)


# ===========================================================================
# 4. Aliased import resolution
# ===========================================================================

class TestAliasedImportResolution:

    def test_import_alias_resolves_to_original(self, tmp_path):
        write_file(tmp_path, "services/payment.py", "def process(): pass\n")
        write_file(tmp_path, "services/__init__.py", "")
        source = "import services.payment as payment\n"
        write_file(tmp_path, "app.py", source)

        svc_pm, _ = parse_and_extract("def process(): pass\n",
                                       str(tmp_path / "services" / "payment.py"))
        init_pm, _ = parse_and_extract("", str(tmp_path / "services" / "__init__.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([svc_pm, init_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        import_deps = [d for d in result.resolved_dependencies
                       if d.dep_type == DependencyType.IMPORT]
        assert import_deps
        dep = import_deps[0]
        assert dep.status == ResolutionStatus.RESOLVED
        assert dep.resolved_module == "services.payment"

    def test_from_import_alias(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "from utils import helper as h\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        deps = [d for d in result.resolved_dependencies
                if d.dep_type == DependencyType.FROM_IMPORT]
        assert deps[0].status == ResolutionStatus.RESOLVED

    def test_import_lookup_alias_maps_to_module(self, tmp_path):
        """_build_import_lookup maps alias to original module."""
        source = "import services.payment as pay\n"
        pm, _ = parse_and_extract(source, "app.py")
        lookup = _build_import_lookup(pm.imports)
        assert "pay" in lookup
        assert lookup["pay"] == "services.payment"


# ===========================================================================
# 5. Relative imports (single level)
# ===========================================================================

class TestRelativeImports:

    def test_from_dot_import_sibling(self, tmp_path):
        """from .utils import helper  (within same package)"""
        write_file(tmp_path, "pkg/__init__.py", "")
        write_file(tmp_path, "pkg/utils.py", "def helper(): pass\n")
        write_file(tmp_path, "pkg/app.py", "from .utils import helper\n")

        utils_pm, _ = parse_and_extract("def helper(): pass\n",
                                         str(tmp_path / "pkg" / "utils.py"))
        init_pm, _ = parse_and_extract("", str(tmp_path / "pkg" / "__init__.py"))
        app_pm, app_md = parse_and_extract("from .utils import helper\n",
                                            str(tmp_path / "pkg" / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, init_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        rel_deps = [d for d in result.resolved_dependencies
                    if d.is_relative]
        assert rel_deps
        dep = rel_deps[0]
        assert dep.status == ResolutionStatus.RESOLVED
        assert dep.resolved_module == "pkg.utils"

    def test_relative_import_level_1(self, tmp_path):
        """Relative level preserved after resolution."""
        write_file(tmp_path, "pkg/__init__.py", "")
        write_file(tmp_path, "pkg/helper.py", "def util(): pass\n")
        write_file(tmp_path, "pkg/main.py", "from . import helper\n")

        helper_pm, _ = parse_and_extract("def util(): pass\n",
                                          str(tmp_path / "pkg" / "helper.py"))
        init_pm, _ = parse_and_extract("", str(tmp_path / "pkg" / "__init__.py"))
        main_pm, main_md = parse_and_extract("from . import helper\n",
                                              str(tmp_path / "pkg" / "main.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([helper_pm, init_pm, main_pm])
        result = resolver.resolve_module_dependencies(main_md, main_pm)

        rel_deps = [d for d in result.resolved_dependencies if d.is_relative]
        assert rel_deps
        assert rel_deps[0].relative_level == 1


# ===========================================================================
# 6. Multi-level relative imports
# ===========================================================================

class TestMultiLevelRelativeImports:

    def test_from_dotdot_import(self, tmp_path):
        """from ..services import auth  (go up to parent, then down)"""
        write_file(tmp_path, "project/__init__.py", "")
        write_file(tmp_path, "project/services/__init__.py", "")
        write_file(tmp_path, "project/services/auth.py", "def login(): pass\n")
        write_file(tmp_path, "project/api/__init__.py", "")
        write_file(tmp_path, "project/api/views.py", "from ..services import auth\n")

        auth_pm, _ = parse_and_extract("def login(): pass\n",
                                        str(tmp_path / "project" / "services" / "auth.py"))
        other_pms = [
            parse_and_extract("", str(tmp_path / "project" / "__init__.py"))[0],
            parse_and_extract("", str(tmp_path / "project" / "services" / "__init__.py"))[0],
            parse_and_extract("", str(tmp_path / "project" / "api" / "__init__.py"))[0],
        ]
        views_pm, views_md = parse_and_extract(
            "from ..services import auth\n",
            str(tmp_path / "project" / "api" / "views.py")
        )

        all_pms = [auth_pm, *other_pms, views_pm]

        resolver = NameResolver(tmp_path)
        resolver.prepare(all_pms)
        result = resolver.resolve_module_dependencies(views_md, views_pm)

        rel_deps = [d for d in result.resolved_dependencies if d.is_relative]
        assert rel_deps
        dep = rel_deps[0]
        assert dep.relative_level == 2
        assert dep.status == ResolutionStatus.RESOLVED
        assert "services" in (dep.resolved_module or "")

    def test_triple_dot_relative_import(self, tmp_path):
        """Level-3 relative import; no crash even if unresolvable."""
        write_file(tmp_path, "a/__init__.py", "")
        write_file(tmp_path, "a/b/__init__.py", "")
        write_file(tmp_path, "a/b/c.py", "from ...common import cfg\n")

        pm, md = parse_and_extract(
            "from ...common import cfg\n",
            str(tmp_path / "a" / "b" / "c.py")
        )
        other_pms = [
            parse_and_extract("", str(tmp_path / "a" / "__init__.py"))[0],
            parse_and_extract("", str(tmp_path / "a" / "b" / "__init__.py"))[0],
        ]

        resolver = NameResolver(tmp_path)
        resolver.prepare([*other_pms, pm])
        # Should not raise
        result = resolver.resolve_module_dependencies(md, pm)
        rel_deps = [d for d in result.resolved_dependencies if d.is_relative]
        assert rel_deps
        assert rel_deps[0].relative_level == 3


# ===========================================================================
# 7. Internal vs external dependency classification
# ===========================================================================

class TestInternalExternalClassification:

    def test_os_is_stdlib(self):
        assert _is_stdlib("os") is True

    def test_sys_is_stdlib(self):
        assert _is_stdlib("sys") is True

    def test_typing_is_stdlib(self):
        assert _is_stdlib("typing") is True

    def test_requests_is_not_stdlib(self):
        assert _is_stdlib("requests") is False

    def test_numpy_is_external(self, tmp_path):
        """numpy is not in the project, so classified external."""
        assert _is_external_module("numpy", {}) is True

    def test_internal_module_not_external(self, tmp_path):
        write_file(tmp_path, "utils.py", "")
        discovery = ModuleDiscovery(tmp_path)
        modules = discovery.discover()
        assert _is_external_module("utils", modules) is False

    def test_mixed_resolution_stats(self, tmp_path):
        """ResolutionResult tracks resolved/external/unresolved counts."""
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "import os\nimport utils\nimport nonexistent_package\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        assert result.external_count >= 1   # os (stdlib)
        assert result.resolved_count >= 1   # utils (internal)


# ===========================================================================
# 8. Function symbol resolution
# ===========================================================================

class TestFunctionSymbolResolution:

    def test_symbol_table_contains_function(self, tmp_path):
        utils_pm, _ = parse_and_extract("def helper(): pass\n",
                                         str(tmp_path / "utils.py"))
        write_file(tmp_path, "utils.py", "def helper(): pass\n")

        builder = SymbolTableBuilder()
        # Provide override so builder uses real dotted name
        symbols = builder.build_from_parsed_modules(
            [utils_pm],
            module_name_override={str(tmp_path / "utils.py"): "utils"}
        )
        assert "utils.helper" in symbols
        assert symbols["utils.helper"].symbol_type == "function"

    def test_from_import_resolves_function_symbol(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        write_file(tmp_path, "app.py", "from utils import helper\n")

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract("from utils import helper\n",
                                            str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        from_imports = [d for d in result.resolved_dependencies
                        if d.dep_type == DependencyType.FROM_IMPORT]
        dep = from_imports[0]
        assert dep.resolved_symbol == "utils.helper"


# ===========================================================================
# 9. Class symbol resolution
# ===========================================================================

class TestClassSymbolResolution:

    def test_symbol_table_contains_class(self, tmp_path):
        write_file(tmp_path, "models.py", "class User:\n    pass\n")
        models_pm, _ = parse_and_extract("class User:\n    pass\n",
                                          str(tmp_path / "models.py"))
        builder = SymbolTableBuilder()
        symbols = builder.build_from_parsed_modules(
            [models_pm],
            module_name_override={str(tmp_path / "models.py"): "models"}
        )
        assert "models.User" in symbols
        assert symbols["models.User"].symbol_type == "class"

    def test_from_import_resolves_class_symbol(self, tmp_path):
        write_file(tmp_path, "models.py", "class User:\n    pass\n")
        write_file(tmp_path, "app.py", "from models import User\n")

        models_pm, _ = parse_and_extract("class User:\n    pass\n",
                                          str(tmp_path / "models.py"))
        app_pm, app_md = parse_and_extract("from models import User\n",
                                            str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([models_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        dep = [d for d in result.resolved_dependencies
               if d.dep_type == DependencyType.FROM_IMPORT][0]
        assert dep.resolved_symbol == "models.User"

    def test_inheritance_base_class_resolved(self, tmp_path):
        write_file(tmp_path, "base.py", "class Base:\n    pass\n")
        write_file(tmp_path, "child.py",
                   "from base import Base\nclass Child(Base):\n    pass\n")

        base_pm, _ = parse_and_extract("class Base:\n    pass\n",
                                        str(tmp_path / "base.py"))
        child_pm, child_md = parse_and_extract(
            "from base import Base\nclass Child(Base):\n    pass\n",
            str(tmp_path / "child.py")
        )

        resolver = NameResolver(tmp_path)
        resolver.prepare([base_pm, child_pm])
        result = resolver.resolve_module_dependencies(child_md, child_pm)

        inh_deps = [d for d in result.resolved_dependencies
                    if d.dep_type == DependencyType.INHERITANCE]
        assert inh_deps
        dep = inh_deps[0]
        assert dep.status == ResolutionStatus.RESOLVED
        assert dep.resolved_symbol == "base.Base"


# ===========================================================================
# 10. Method symbol resolution
# ===========================================================================

class TestMethodSymbolResolution:

    def test_symbol_table_contains_method(self, tmp_path):
        source = "class MyClass:\n    def my_method(self): pass\n"
        write_file(tmp_path, "mymod.py", source)
        pm, _ = parse_and_extract(source, str(tmp_path / "mymod.py"))

        builder = SymbolTableBuilder()
        symbols = builder.build_from_parsed_modules(
            [pm],
            module_name_override={str(tmp_path / "mymod.py"): "mymod"}
        )
        assert "mymod.MyClass.my_method" in symbols
        assert symbols["mymod.MyClass.my_method"].symbol_type == "method"
        assert symbols["mymod.MyClass.my_method"].parent_class == "MyClass"

    def test_method_is_public(self, tmp_path):
        source = "class MyClass:\n    def public_method(self): pass\n    def _private(self): pass\n"
        write_file(tmp_path, "mymod.py", source)
        pm, _ = parse_and_extract(source, str(tmp_path / "mymod.py"))

        builder = SymbolTableBuilder()
        symbols = builder.build_from_parsed_modules(
            [pm],
            module_name_override={str(tmp_path / "mymod.py"): "mymod"}
        )
        assert symbols["mymod.MyClass.public_method"].is_public is True
        assert symbols["mymod.MyClass._private"].is_public is False


# ===========================================================================
# 11. Local function call resolution
# ===========================================================================

class TestLocalCallResolution:

    def test_local_call_resolved_with_full_confidence(self, tmp_path):
        source = """\
def helper():
    pass

def process():
    helper()
"""
        write_file(tmp_path, "mymod.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "mymod.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        call_deps = [d for d in result.resolved_dependencies
                     if d.dep_type == DependencyType.FUNCTION_CALL]
        helper_call = next(
            (d for d in call_deps if d.original_target == "helper"), None
        )
        assert helper_call is not None
        assert helper_call.status == ResolutionStatus.RESOLVED
        assert helper_call.confidence == 1.0
        assert "helper" in (helper_call.resolved_symbol or "")

    def test_local_method_resolved(self, tmp_path):
        source = """\
class MyClass:
    def method_a(self):
        pass
    def method_b(self):
        self.method_a()
"""
        write_file(tmp_path, "mymod.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "mymod.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        call_deps = [d for d in result.resolved_dependencies
                     if d.dep_type == DependencyType.FUNCTION_CALL]
        a_call = next(
            (d for d in call_deps if d.original_target == "method_a"), None
        )
        # method_a is in the module symbol table → should resolve
        assert a_call is not None
        assert a_call.status == ResolutionStatus.RESOLVED


# ===========================================================================
# 12. Cross-module function call resolution
# ===========================================================================

class TestCrossModuleCallResolution:

    def test_cross_module_call_resolved_via_import_lookup(self, tmp_path):
        """
        utils.py defines helper().
        app.py does 'from utils import helper' then calls helper().
        The call to helper should resolve to utils.helper.
        """
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = """\
from utils import helper

def process():
    helper()
"""
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        call_deps = [d for d in result.resolved_dependencies
                     if d.dep_type == DependencyType.FUNCTION_CALL]
        helper_call = next(
            (d for d in call_deps if d.original_target == "helper"), None
        )
        assert helper_call is not None
        assert helper_call.status == ResolutionStatus.RESOLVED
        assert helper_call.resolved_module == "utils"
        assert helper_call.resolved_symbol == "utils.helper"
        assert helper_call.confidence == 1.0

    def test_cross_module_call_via_aliased_import(self, tmp_path):
        """from utils import helper as h; then calling h() resolves via alias."""
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "from utils import helper as h\ndef run():\n    h()\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        call_deps = [d for d in result.resolved_dependencies
                     if d.dep_type == DependencyType.FUNCTION_CALL]
        h_call = next((d for d in call_deps if d.original_target == "h"), None)
        assert h_call is not None
        assert h_call.status == ResolutionStatus.RESOLVED
        assert h_call.resolved_symbol == "utils.helper"


# ===========================================================================
# 13. Unknown / unresolved symbols
# ===========================================================================

class TestUnresolvedSymbols:

    def test_call_to_truly_unknown_function(self, tmp_path):
        source = "def run():\n    some_mystery_func()\n"
        write_file(tmp_path, "app.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        call_deps = [d for d in result.resolved_dependencies
                     if d.dep_type == DependencyType.FUNCTION_CALL]
        mystery = next(
            (d for d in call_deps if d.original_target == "some_mystery_func"), None
        )
        assert mystery is not None
        assert mystery.status == ResolutionStatus.UNRESOLVED
        assert mystery.confidence == 0.0

    def test_relative_import_from_top_level_is_unresolvable(self, tmp_path):
        """Relative import from a module with no package should be RELATIVE_UNRESOLVED."""
        source = "from . import sibling\n"
        write_file(tmp_path, "standalone.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "standalone.py"))

        # Module name "standalone" has no package; level=1 would need parent
        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        rel_deps = [d for d in result.resolved_dependencies if d.is_relative]
        if rel_deps:
            # It may be unresolved or relative_unresolved
            for d in rel_deps:
                assert d.status in (
                    ResolutionStatus.RELATIVE_UNRESOLVED,
                    ResolutionStatus.UNRESOLVED,
                )


# ===========================================================================
# 14. Package / __init__.py module resolution
# ===========================================================================

class TestPackageResolution:

    def test_package_discovered_via_init(self, tmp_path):
        write_file(tmp_path, "services/__init__.py", "")
        write_file(tmp_path, "services/payment.py", "def pay(): pass\n")

        discovery = ModuleDiscovery(tmp_path)
        modules = discovery.discover()

        assert "services" in modules
        assert "services.payment" in modules
        assert modules["services"].is_package is True
        assert modules["services.payment"].is_package is False

    def test_nested_package_module_name(self, tmp_path):
        write_file(tmp_path, "a/__init__.py", "")
        write_file(tmp_path, "a/b/__init__.py", "")
        write_file(tmp_path, "a/b/c.py", "")

        discovery = ModuleDiscovery(tmp_path)
        modules = discovery.discover()

        assert "a" in modules
        assert "a.b" in modules
        assert "a.b.c" in modules

    def test_import_submodule_resolves(self, tmp_path):
        write_file(tmp_path, "services/__init__.py", "")
        write_file(tmp_path, "services/payment.py", "def pay(): pass\n")
        source = "from services.payment import pay\n"
        write_file(tmp_path, "app.py", source)

        svc_pm, _ = parse_and_extract("def pay(): pass\n",
                                       str(tmp_path / "services" / "payment.py"))
        init_pm, _ = parse_and_extract("", str(tmp_path / "services" / "__init__.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([svc_pm, init_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        from_imports = [d for d in result.resolved_dependencies
                        if d.dep_type == DependencyType.FROM_IMPORT]
        assert from_imports
        dep = from_imports[0]
        assert dep.status == ResolutionStatus.RESOLVED
        assert dep.resolved_module == "services.payment"


# ===========================================================================
# 15. Multiple modules resolved together
# ===========================================================================

class TestMultipleModules:

    def test_project_resolution_returns_all_modules(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "models.py": "class User:\n    pass\n",
            "app.py": "from utils import helper\nfrom models import User\n",
        }
        make_project(tmp_path, files)

        parser = ASTParser()
        parsed = [
            parser.parse_file(tmp_path / name)
            for name in ("utils.py", "models.py", "app.py")
        ]

        result = resolve_dependencies(tmp_path, parsed)

        assert result.total_modules == 3
        assert "utils" in result.module_results or "app" in result.module_results

    def test_global_symbol_table_aggregated(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "models.py": "class User:\n    pass\n",
        }
        make_project(tmp_path, files)

        parser = ASTParser()
        parsed = [parser.parse_file(tmp_path / name) for name in files]

        resolver = NameResolver(tmp_path)
        resolver.prepare(parsed)

        assert "utils.helper" in resolver._symbol_table
        assert "models.User" in resolver._symbol_table

    def test_cross_module_dependencies_linked(self, tmp_path):
        files = {
            "utils.py": "def helper(): pass\n",
            "app.py": "from utils import helper\ndef run():\n    helper()\n",
        }
        make_project(tmp_path, files)

        parser = ASTParser()
        parsed = [parser.parse_file(tmp_path / name) for name in files]

        result = resolve_dependencies(tmp_path, parsed)

        all_deps = result.all_resolved_dependencies
        helper_call = next(
            (d for d in all_deps
             if d.dep_type == DependencyType.FUNCTION_CALL and d.original_target == "helper"),
            None,
        )
        assert helper_call is not None
        assert helper_call.status == ResolutionStatus.RESOLVED
        assert helper_call.resolved_module == "utils"


# ===========================================================================
# 16. Confidence values
# ===========================================================================

class TestConfidenceValues:

    def test_exact_symbol_match_confidence_1(self, tmp_path):
        """Exact internal symbol → confidence 1.0."""
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "from utils import helper\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        dep = [d for d in result.resolved_dependencies
               if d.dep_type == DependencyType.FROM_IMPORT][0]
        assert dep.confidence == 1.0

    def test_stdlib_confidence_1(self, tmp_path):
        """Standard library module → confidence 1.0."""
        source = "import os\n"
        write_file(tmp_path, "app.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        dep = result.resolved_dependencies[0]
        assert dep.status == ResolutionStatus.EXTERNAL
        assert dep.confidence == 1.0

    def test_unresolved_confidence_0(self, tmp_path):
        """Unresolved dependency → confidence 0.0."""
        source = "def run():\n    totally_unknown()\n"
        write_file(tmp_path, "app.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        call_deps = [d for d in result.resolved_dependencies
                     if d.dep_type == DependencyType.FUNCTION_CALL]
        unknown = next(
            (d for d in call_deps if d.original_target == "totally_unknown"), None
        )
        assert unknown is not None
        assert unknown.confidence == 0.0

    def test_ambiguous_base_class_confidence_05(self, tmp_path):
        """Two modules define a class with the same name → confidence 0.5."""
        write_file(tmp_path, "mod_a.py", "class Base:\n    pass\n")
        write_file(tmp_path, "mod_b.py", "class Base:\n    pass\n")
        source = "class Child(Base):\n    pass\n"
        write_file(tmp_path, "child.py", source)

        pms = [
            parse_and_extract("class Base:\n    pass\n", str(tmp_path / "mod_a.py"))[0],
            parse_and_extract("class Base:\n    pass\n", str(tmp_path / "mod_b.py"))[0],
        ]
        child_pm, child_md = parse_and_extract(source, str(tmp_path / "child.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([*pms, child_pm])
        result = resolver.resolve_module_dependencies(child_md, child_pm)

        inh_deps = [d for d in result.resolved_dependencies
                    if d.dep_type == DependencyType.INHERITANCE]
        assert inh_deps
        assert inh_deps[0].confidence == 0.5

    def test_all_confidence_values_in_range(self, tmp_path):
        """Every resolved dependency must have confidence in [0.0, 1.0]."""
        files = {
            "utils.py": "def helper(): pass\n",
            "models.py": "class User:\n    pass\n",
            "app.py": (
                "import os\n"
                "from utils import helper\n"
                "from models import User\n"
                "def run():\n    helper()\n    unknown_func()\n"
            ),
        }
        make_project(tmp_path, files)

        parser = ASTParser()
        parsed = [parser.parse_file(tmp_path / name) for name in files]
        result = resolve_dependencies(tmp_path, parsed)

        for dep in result.all_resolved_dependencies:
            assert 0.0 <= dep.confidence <= 1.0, (
                f"confidence out of range for {dep}: {dep.confidence}"
            )


# ===========================================================================
# 17. Resolution metadata (reason strings)
# ===========================================================================

class TestResolutionMetadata:

    def test_resolved_dep_has_reason(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "from utils import helper\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        for dep in result.resolved_dependencies:
            assert isinstance(dep.resolution_reason, str)
            assert len(dep.resolution_reason) > 0

    def test_external_reason_mentions_stdlib_or_third_party(self, tmp_path):
        source = "import os\nimport requests\n"
        write_file(tmp_path, "app.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        for dep in result.resolved_dependencies:
            if dep.status == ResolutionStatus.EXTERNAL:
                reason_lower = dep.resolution_reason.lower()
                assert ("standard" in reason_lower or "third" in reason_lower
                        or "external" in reason_lower or "built-in" in reason_lower)

    def test_resolved_dep_fields_populated(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "from utils import helper\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([utils_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        dep = [d for d in result.resolved_dependencies
               if d.dep_type == DependencyType.FROM_IMPORT][0]
        assert dep.source_file != ""
        assert dep.source_line >= 0
        assert dep.resolved_module is not None
        assert dep.resolved_file is not None

    def test_resolution_percentage_property(self, tmp_path):
        result = ResolutionResult(module_name="test", file_path="test.py")
        assert result.resolution_percentage == 100.0  # 0 deps → 100%


# ===========================================================================
# 18. No crashes on syntax-error modules
# ===========================================================================

class TestSyntaxErrorModules:

    def test_syntax_error_module_does_not_crash(self, tmp_path):
        broken = "def broken(\n    pass\n"
        write_file(tmp_path, "broken.py", broken)

        parser = ASTParser()
        pm = parser.parse_source(broken, str(tmp_path / "broken.py"))
        assert len(pm.syntax_errors) > 0

        extractor = DependencyExtractor()
        md = extractor.extract(pm)

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        # Must not raise
        result = resolver.resolve_module_dependencies(md, pm)
        assert isinstance(result, ResolutionResult)

    def test_mixed_valid_and_invalid_modules(self, tmp_path):
        files = {
            "good.py": "def func(): pass\n",
            "bad.py": "def broken(\n    pass\n",
        }
        make_project(tmp_path, files)

        parser = ASTParser()
        parsed = [parser.parse_file(tmp_path / name) for name in files]

        # resolve_dependencies should not raise even with a broken module
        result = resolve_dependencies(tmp_path, parsed)
        assert isinstance(result, ProjectResolutionResult)

    def test_empty_module_no_crash(self, tmp_path):
        write_file(tmp_path, "empty.py", "")
        pm, md = parse_and_extract("", str(tmp_path / "empty.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)
        assert result.total_dependencies == 0


# ===========================================================================
# Additional integration tests
# ===========================================================================

class TestImportLookupHelper:
    """Unit tests for the _build_import_lookup helper."""

    def test_simple_import(self):
        pm, _ = parse_and_extract("import os\n", "app.py")
        lookup = _build_import_lookup(pm.imports)
        assert "os" in lookup
        assert lookup["os"] == "os"

    def test_from_import(self):
        pm, _ = parse_and_extract("from utils import helper\n", "app.py")
        lookup = _build_import_lookup(pm.imports)
        assert "helper" in lookup
        assert lookup["helper"] == "utils.helper"

    def test_aliased_import(self):
        pm, _ = parse_and_extract("import numpy as np\n", "app.py")
        lookup = _build_import_lookup(pm.imports)
        assert "np" in lookup
        assert lookup["np"] == "numpy"

    def test_aliased_from_import(self):
        pm, _ = parse_and_extract("from utils import helper as h\n", "app.py")
        lookup = _build_import_lookup(pm.imports)
        assert "h" in lookup
        assert lookup["h"] == "utils.helper"

    def test_multiple_from_imports(self):
        pm, _ = parse_and_extract("from typing import List, Optional\n", "app.py")
        lookup = _build_import_lookup(pm.imports)
        assert "List" in lookup
        assert "Optional" in lookup


class TestBuiltinDecorators:

    def test_property_decorator_is_external(self, tmp_path):
        source = "@property\ndef val(self):\n    return self._val\n"
        write_file(tmp_path, "app.py", source)
        pm, md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([pm])
        result = resolver.resolve_module_dependencies(md, pm)

        decorators = [d for d in result.resolved_dependencies
                      if d.dep_type == DependencyType.DECORATOR]
        prop_dec = next((d for d in decorators if d.original_target == "property"), None)
        assert prop_dec is not None
        assert prop_dec.status == ResolutionStatus.EXTERNAL

    def test_custom_decorator_from_import_resolved(self, tmp_path):
        write_file(tmp_path, "decorators.py", "def my_dec(f):\n    return f\n")
        source = "from decorators import my_dec\n@my_dec\ndef func(): pass\n"
        write_file(tmp_path, "app.py", source)

        dec_pm, _ = parse_and_extract("def my_dec(f):\n    return f\n",
                                       str(tmp_path / "decorators.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        resolver = NameResolver(tmp_path)
        resolver.prepare([dec_pm, app_pm])
        result = resolver.resolve_module_dependencies(app_md, app_pm)

        dec_deps = [d for d in result.resolved_dependencies
                    if d.dep_type == DependencyType.DECORATOR
                    and d.original_target == "my_dec"]
        assert dec_deps
        assert dec_deps[0].status == ResolutionStatus.RESOLVED


class TestConvenienceFunctions:

    def test_resolve_dependencies_returns_project_result(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        write_file(tmp_path, "app.py", "from utils import helper\n")

        parser = ASTParser()
        parsed = [parser.parse_file(tmp_path / name) for name in ("utils.py", "app.py")]
        result = resolve_dependencies(tmp_path, parsed)
        assert isinstance(result, ProjectResolutionResult)

    def test_resolve_single_module_convenience(self, tmp_path):
        write_file(tmp_path, "utils.py", "def helper(): pass\n")
        source = "from utils import helper\n"
        write_file(tmp_path, "app.py", source)

        utils_pm, _ = parse_and_extract("def helper(): pass\n", str(tmp_path / "utils.py"))
        app_pm, app_md = parse_and_extract(source, str(tmp_path / "app.py"))

        result = resolve_single_module(
            tmp_path, app_md, [utils_pm, app_pm], parsed_module=app_pm
        )
        assert isinstance(result, ResolutionResult)
        assert result.total_dependencies > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
