from __future__ import annotations
from pathlib import Path
import ast
import importlib.util
import re
import subprocess
import sys
from typing import Any, Mapping
from backend.app.schemas.models import TestGenerationResult, RepairResult
from .repair import _extract_json_object, ProviderResponseError, ProviderTransportError
from .repair import LLMProvider, RepairService
from .validation import detect_project

class TestGenerationService:
    __test__ = False
    def __init__(self, provider: LLMProvider | RepairService | None = None): self.provider_source = provider
    @property
    def provider(self):
        return self.provider_source.provider if isinstance(self.provider_source, RepairService) else self.provider_source
    MAX_ATTEMPTS = 3
    BUILTIN_FIXTURES = {
        "cache", "capfd", "capfdbinary", "caplog", "capsys", "capsysbinary",
        "doctest_namespace", "monkeypatch", "pytestconfig", "record_property",
        "record_testsuite_property", "record_xml_attribute", "recwarn",
        "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory",
    }

    @staticmethod
    def _validate_file(item: Any) -> tuple[str, str, str, str]:
        if not isinstance(item, dict):
            raise ValueError("Each generated test must be an object.")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Generated test path is required.")
        normalized = path.replace("\\", "/")
        candidate = Path(normalized)
        if candidate.is_absolute() or re.match(r"^[A-Za-z]:", normalized):
            raise ValueError(f"Generated test path must be relative: {path}")
        if "\\" in path or ".." in candidate.parts or any(part in {"", "."} for part in candidate.parts):
            raise ValueError(f"Generated test path traversal is not allowed: {path}")
        if not normalized.startswith("tests/ai_generated/") or not normalized.lower().endswith(".py"):
            raise ValueError(f"generated tests cannot target production files; path must be a Python file under tests/ai_generated/: {path}")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Generated test content is empty: {path}")
        if re.search(r"(?i)(sk-or-v1-|bearer\s+[a-z0-9._~+/=-]+|api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=])", content):
            raise ValueError(f"Generated test contains a possible secret or token: {path}")
        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError as exc:
            raise ValueError(f"Generated test has invalid Python syntax at line {exc.lineno}: {exc.msg}") from exc
        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_") for child in node.body)]
        if not functions and not classes:
            raise ValueError(f"Generated test must contain pytest-compatible test definitions: {path}")
        if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            raise ValueError(f"Generated test must contain a real assertion: {path}")
        kind = item.get("type", "regression")
        if kind == "targeted_integration": kind = "integration"
        if kind not in {"regression", "integration"}:
            raise ValueError(f"Generated test type must be regression or integration: {path}")
        purpose = item.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"Generated test purpose is required: {path}")
        return normalized, content, kind, purpose

    @staticmethod
    def _parse_response(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, RepairResult):
            items = value.proposed_changes
        elif isinstance(value, dict):
            items = value.get("tests")
        elif isinstance(value, str):
            try:
                items = _extract_json_object(value).get("tests")
            except ProviderResponseError as exc:
                raise ValueError(str(exc)) from exc
        else:
            items = None
        if not isinstance(items, list):
            raise ValueError("Provider response must contain a tests array; unified test patches are no longer accepted.")
        validated=[]
        for item in items:
            path, content, kind, purpose = TestGenerationService._validate_file(item)
            validated.append({"path":path,"content":content,"type":kind,"purpose":purpose, **item})
            validated[-1]["test_file"] = path
            validated[-1]["generated_code"] = content
            validated[-1]["test_kind"] = kind
        return validated

    @staticmethod
    def _ruff_preflight(root: str | Path, details: list[dict[str, Any]]) -> None:
        """Reject invalid model output without changing the generated source."""
        if not importlib.util.find_spec("ruff"):
            return
        for item in details:
            check = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--stdin-filename", item["path"], "-"],
                input=item["content"].encode("utf-8"), cwd=Path(root).resolve(),
                capture_output=True, shell=False, check=False, timeout=30,
            )
            if check.returncode:
                output = (check.stdout + check.stderr).decode("utf-8", errors="replace")[-2000:]
                raise ValueError(f"Generated test failed Ruff preflight: {output}")

    def from_repair(self, repair: RepairResult, *, root: str | Path = ".", changed_symbols=None, impacted_symbols=None) -> TestGenerationResult:
        """Validate and materialize test artifacts returned by the repair call."""
        raw = []
        if repair.regression_test:
            raw.append({
                "path": repair.regression_test.get("path"),
                "type": "regression",
                "purpose": repair.regression_test.get("reason"),
                "content": repair.regression_test.get("source"),
                "target_symbol": repair.regression_test.get("target"),
                "reason": repair.regression_test.get("reason"),
            })
        if repair.integration_test:
            raw.append({
                "path": repair.integration_test.get("path"),
                "type": "integration",
                "purpose": repair.integration_test.get("reason"),
                "content": repair.integration_test.get("source"),
                "target_symbol": repair.integration_test.get("target"),
                "dependency_path": repair.integration_test.get("dependency_path"),
                "reason": repair.integration_test.get("reason"),
            })
        if not repair.regression_test:
            raise ValueError("The repair response must contain a regression test artifact.")
        details = self._parse_response({"tests": raw})
        targets = self._repair_target_symbols(list(changed_symbols or []), repair.model_dump())
        self._repair_relevance_preflight(details, targets)
        self._fixture_preflight(root, details)
        self._integration_preflight(root, details)
        self._ruff_preflight(root, details)
        for index, item in enumerate(details):
            item.update({
                "test_id": f"generated-1-{index + 1}",
                "test_name": Path(item["path"]).stem,
                "execution_status": "NOT_RUN",
                "execution_output": "Candidate validation has not run.",
                "duration_seconds": 0.0,
                "stdout_stderr_summary": "Candidate validation has not run.",
                "attempt": 1,
            })
        regression = [item for item in details if item["test_kind"] == "regression"]
        integration = [item for item in details if item["test_kind"] == "integration"]
        if len(regression) != 1 or len(integration) > 1:
            raise ValueError("Exactly one regression and at most one integration test are required.")
        return TestGenerationResult(
            status="available",
            target_symbols=targets,
            targets=[item if isinstance(item, dict) else {"symbol": str(item)} for item in [*(changed_symbols or []), *(impacted_symbols or [])]],
            framework_detected=detect_project(root).framework,
            generated_test_files={item["path"]: item["content"].replace("\r\n", "\n").replace("\r", "\n") for item in details},
            generated_tests=regression,
            generated_integration_tests=integration,
            test_rationale=repair.explanation or "Provider-generated regression and integration evidence.",
        )

    @classmethod
    def _fixture_preflight(cls, root: str | Path, details: list[dict[str, Any]]) -> None:
        """Reject fixture references that cannot be resolved from the generated path."""
        root_path = Path(root).resolve()
        for item in details:
            available = set(cls.BUILTIN_FIXTURES)
            generated_path = root_path / item["path"]
            parents = []
            parent = generated_path.parent
            while parent == root_path or root_path in parent.parents:
                parents.append(parent)
                if parent == root_path:
                    break
                parent = parent.parent
            # Some established repositories use singular `test/` while AI
            # artifacts are isolated under `tests/ai_generated/`. Validation
            # loads that repository conftest explicitly as a pytest plugin.
            parents.extend([root_path / "test", root_path / "tests"])
            for parent in dict.fromkeys(parents):
                conftest = parent / "conftest.py"
                if conftest.is_file():
                    try:
                        tree = ast.parse(conftest.read_text(encoding="utf-8"), filename=str(conftest))
                    except (OSError, SyntaxError):
                        tree = None
                    if tree:
                        for node in tree.body:
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                                isinstance(decorator, ast.Attribute) and decorator.attr == "fixture"
                                or isinstance(decorator, ast.Name) and decorator.id == "fixture"
                                for decorator in node.decorator_list
                            ):
                                available.add(node.name)
            tree = ast.parse(item["content"], filename=item["path"])
            unknown_names = set()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                    continue
                parametrized = set()
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call) or not decorator.args:
                        continue
                    func = decorator.func
                    if isinstance(func, ast.Attribute) and func.attr == "parametrize" and isinstance(decorator.args[0], ast.Constant) and isinstance(decorator.args[0].value, str):
                        parametrized.update(name.strip() for name in decorator.args[0].value.split(","))
                unknown_names.update(
                    arg.arg for arg in node.args.args
                    if arg.arg not in available and arg.arg not in parametrized and arg.arg not in {"self", "cls"}
                )
            unknown = sorted(unknown_names)
            if unknown:
                names = ", ".join(unknown)
                raise ValueError(
                    f"Generated test references unavailable fixture(s): {names}. "
                    "Generate a dependency-free source/AST regression test instead; do not use unavailable app/client fixtures."
                )

    @staticmethod
    def _integration_preflight(root: str | Path, details: list[dict[str, Any]]) -> None:
        """A label cannot turn a direct callee test into caller integration."""
        for item in details:
            if item.get("test_kind") != "integration":
                continue
            path = str(item.get("dependency_path") or "")
            if "->" not in path:
                raise ValueError("Integration test must identify an actual caller -> callee path.")
            caller = path.split("->", 1)[0].strip().replace(":", ".")
            tree = ast.parse(item["content"])
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]

            def qualified(node):
                if isinstance(node, ast.Name):
                    return aliases.get(node.id, node.id)
                if isinstance(node, ast.Attribute):
                    return qualified(node.value) + "." + node.attr
                if isinstance(node, ast.Call):
                    return qualified(node.func)
                return ""

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = qualified(node.value.func)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            if any(qualified(node.func) == caller or qualified(node.func).endswith('.' + caller) for node in calls):
                continue
            # HTTP integration exercises a decorated route via the real client.
            routes = []
            components = caller.split('.')
            callee_name = path.split('->', 1)[1].strip().replace(':', '.').split('.')[-1]
            for end in range(len(components), 0, -1):
                source = Path(root).joinpath(*components[:end]).with_suffix('.py')
                if not source.is_file():
                    suffix = '/'.join(components[:end]) + '.py'
                    matches = [candidate for candidate in Path(root).rglob(components[end - 1] + '.py')
                               if candidate.relative_to(root).as_posix().endswith('/' + suffix)
                               and not any(part.startswith('.') or part in {'build', 'node_modules', '__pycache__'}
                                           for part in candidate.relative_to(root).parts)]
                    if len(matches) != 1:
                        continue
                    source = matches[0]
                for function in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
                    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    module_caller = end == len(components)
                    calls_callee = any(
                        isinstance(call, ast.Call) and (
                            isinstance(call.func, ast.Attribute) and call.func.attr == callee_name
                            or isinstance(call.func, ast.Name) and call.func.id == callee_name
                        ) for call in ast.walk(function)
                    )
                    if (module_caller and calls_callee) or (not module_caller and function.name == components[-1]):
                        for decorator in function.decorator_list:
                            if isinstance(decorator, ast.Call) and decorator.args and isinstance(decorator.args[0], ast.Constant):
                                route = decorator.args[0].value
                                if isinstance(route, str) and route.startswith('/'):
                                    routes.append(route)
            route_patterns = [re.sub(r"\\\{[^}]+\\\}|<[^>]+>", "[^/]+", re.escape(route)) for route in routes]
            if routes and any(isinstance(node.func, ast.Attribute) and node.func.attr in {'get', 'post', 'put', 'delete', 'patch', 'request'}
                              and any(isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                                      and any(re.fullmatch(pattern, arg.value) for pattern in route_patterns)
                                      for arg in node.args) for node in calls):
                continue
            raise ValueError(f"Integration test does not execute its stated caller {caller}. Call the actual caller or its real HTTP route; calling only the repaired callee is regression coverage.")

    @staticmethod
    def _repair_target_symbols(changed_symbols: list[Any], repair: Mapping[str, Any] | None) -> list[str]:
        affected = {str(path).replace("\\", "/") for path in ((repair or {}).get("affected_files") or [])}
        targets = []
        for item in changed_symbols:
            if not isinstance(item, Mapping):
                continue
            item_file = str(item.get("file") or "").replace("\\", "/")
            if affected and item_file not in affected:
                continue
            name = str(item.get("qualified_name") or item.get("name") or "").rsplit(".", 1)[-1]
            if len(name) >= 4 and name not in targets:
                targets.append(name)
        return targets

    @staticmethod
    def _repair_relevance_preflight(details: list[dict[str, Any]], required_targets: list[str]) -> None:
        """Require an executed regression to exercise the production repair target."""
        if not required_targets:
            return
        regressions = [item for item in details if item.get("test_kind") == "regression"]
        evidence = "\n".join(
            f"{item.get('path', '')}\n{item.get('purpose', '')}\n{item.get('content', '')}" for item in regressions
        )
        if not any(re.search(rf"\b{re.escape(target)}\b", evidence) for target in required_targets):
            names = ", ".join(required_targets)
            raise ValueError(
                f"Generated regression test does not exercise the repaired target(s): {names}. "
                "Generate a dependency-free source/AST test for the exact repaired expression."
            )
    def generate(self, *, root: str | Path = ".", changed_symbols=None, impacted_symbols=None, language=None, existing_tests=None, repair=None, analysis_context=None) -> TestGenerationResult:
        profile=detect_project(root); raw_targets=list(changed_symbols or []) + list(impacted_symbols or [])
        target_details=[item if isinstance(item,dict) else {"symbol":str(item)} for item in raw_targets]
        targets=[str(item.get("qualified_name") or item.get("name") or item.get("symbol")) if isinstance(item,dict) else str(item) for item in raw_targets]
        if self.provider is None: return TestGenerationResult(status="unavailable",target_symbols=targets,framework_detected=profile.framework,warnings=["No LLM provider is configured; tests were not fabricated."])
        required_targets=self._repair_target_symbols(list(changed_symbols or []), repair)
        context={"purpose":"regression_test_generation","requested_test_kinds":["regression","integration"],"requirements":{"executable_code":True,"test_files_only":True,"structured_complete_files":True,"separate_test_metadata":True},"targets":target_details,"required_repair_targets":required_targets,"language":language or profile.language,"framework":profile.framework,"existing_tests":existing_tests,"repair":repair,"analysis_context":analysis_context}
        last_error=None
        generation_attempts = []
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
          details = []
          record = {"attempt": attempt, "provider": getattr(self.provider, 'name', None), "model": getattr(self.provider, 'model', None)}
          try:
            if last_error: context["validation_feedback"] = last_error
            result=self.provider.propose_repair({**context,"attempt":attempt,"max_attempts":self.MAX_ATTEMPTS})
            if isinstance(result, RepairResult):
                record.update(provider=result.provider or record['provider'], model=result.model or record['model'],
                              response_metadata=result.response_metadata)
            if isinstance(result, TestGenerationResult):
                return result.model_copy(update={"target_symbols": targets, "targets":target_details, "framework_detected": result.framework_detected or profile.framework})
            details=self._parse_response(result)
            self._repair_relevance_preflight(details, required_targets)
            self._fixture_preflight(root, details)
            self._integration_preflight(root, details)
            self._ruff_preflight(root, details)
            for index,item in enumerate(details):
                item.update({"test_id":item.get("test_id") or f"generated-{attempt}-{index+1}","test_name":item.get("test_name") or f"Generated test {index+1}","target_symbol":item.get("target_symbol") or (targets[index] if index<len(targets) else None),"execution_status":"NOT_RUN","execution_output":"Candidate validation has not run.","duration_seconds":0.0,"stdout_stderr_summary":"Candidate validation has not run.","attempt":attempt,
                             "provider":result.provider, "model":result.model, "response_metadata":result.response_metadata})
            files={item["path"]:item["content"] for item in details}
            regression=[item for item in details if item["test_kind"] == "regression"]
            integration=[item for item in details if item["test_kind"] == "integration"]
            if len(regression) != 1 or len(integration) > 1:
                raise ValueError("Exactly one focused regression file and at most one real integration file are required.")
            generation_attempts.append({**record, "status": "PASSED"})
            return TestGenerationResult(status="available",target_symbols=targets,targets=target_details,framework_detected=profile.framework,generated_test_files=files,generated_tests=regression,generated_integration_tests=integration,test_rationale=result.explanation or "Provider-generated executable regression and integration files.",warnings=result.warnings,generation_attempts=generation_attempts)
          except ProviderTransportError as exc:
            # The provider already exhausted bounded transport retries. Let the
            # workflow record PROVIDER_ERROR without consuming semantic attempts.
            exc.test_generation_attempts = [*generation_attempts, {**record, "status": "PROVIDER_ERROR",
                "provider": getattr(exc, 'provider_name', record['provider']),
                "model": getattr(exc, 'provider_model', record['model']),
                "provider_attempts": getattr(exc, 'provider_attempts', []), "error_type": exc.error_type}]
            raise
          except Exception as exc:
            last_error=str(exc)
            generation_attempts.append({**record, "status": "REJECTED", "reason": last_error,
                "rejected_tests": details,
                "provider": getattr(exc, 'provider_name', record['provider']),
                "model": getattr(exc, 'provider_model', record['model']),
                "provider_attempts": getattr(exc, 'provider_attempts', [])})
        return TestGenerationResult(status="error",target_symbols=targets,targets=target_details,framework_detected=profile.framework,warnings=[f"TEST_GENERATION_FAILED after {self.MAX_ATTEMPTS} attempts: {last_error}"],generation_attempts=generation_attempts)
