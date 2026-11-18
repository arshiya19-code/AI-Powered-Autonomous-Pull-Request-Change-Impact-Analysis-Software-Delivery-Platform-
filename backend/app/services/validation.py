from __future__ import annotations

import json
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from backend.app.schemas.models import ValidationCheck, ValidationSummary, normalize_validation_variant
from .ci_tools import ScannerService, command_env, redact, executable


@dataclass
class ProjectProfile:
    language: str
    framework: str | None
    commands: dict[str, list[str]]


def detect_project(root: str | Path) -> ProjectProfile:
    root = Path(root)
    names = {p.name for p in root.iterdir()} if root.exists() else set()
    if {"pyproject.toml", "requirements.txt", "pytest.ini", "setup.cfg"} & names or any(root.rglob("*.py")):
        text = (root / "pyproject.toml").read_text(errors="ignore") if (root / "pyproject.toml").exists() else ""
        framework = "pytest"
        tests = [sys.executable, "-m", "pytest"] if framework == "pytest" else [sys.executable, "-m", "unittest"]
        return ProjectProfile("Python", framework, {
            "build": [sys.executable, "-m", "compileall", "-q", "."],
            "unit": tests, "regression": [], "targeted_integration": [], "integration": [],
        })
    if "package.json" in names:
        try:
            data = json.loads((root / "package.json").read_text(errors="ignore"))
        except (ValueError, OSError):
            data = {}
        scripts = data.get("scripts", {})
        blob = str(data)
        framework = "vitest" if "vitest" in blob else "jest" if "jest" in blob else "mocha" if "mocha" in blob else None
        tests = ["npm", "test"] if "test" in scripts else ([framework] if framework else [])
        return ProjectProfile("TypeScript/JavaScript", framework, {
            "build": ["npm", "run", "build"] if "build" in scripts else [],
            "unit": tests, "regression": [], "targeted_integration": [], "integration": [],
        })
    if any(n.endswith((".sln", ".csproj")) for n in names):
        return ProjectProfile("C#", "xUnit/NUnit/MSTest", {"build": ["dotnet", "build"], "unit": ["dotnet", "test"]})
    if "Cargo.toml" in names:
        return ProjectProfile("Rust", "cargo test", {"build": ["cargo", "build"], "unit": ["cargo", "test"]})
    if "CMakeLists.txt" in names:
        return ProjectProfile("C++", "CTest", {"build": ["cmake", "--build", "build"], "unit": ["ctest", "--test-dir", "build"]})
    return ProjectProfile("unknown", None, {})


class SafeCommandRunner:
    allowed = {"python", "python.exe", "python3", "pytest", "npm", "npx", "dotnet", "cargo", "cmake", "ctest", "bandit", "pip-audit"}

    def __init__(self, timeout=60, output_limit=4000):
        self.timeout, self.output_limit = timeout, output_limit

    def run(self, command: Sequence[str], cwd: str | Path, *, variant="ORIGINAL", verifies="") -> ValidationCheck:
        variant = normalize_validation_variant(variant)
        workspace = str(Path(cwd).resolve())
        display = subprocess.list2cmdline(list(command))
        if not command or Path(command[0]).name not in self.allowed:
            return ValidationCheck(check_type="command", status="ERROR", command=list(command), workspace=workspace,
                                   variant=variant, verifies=verifies, command_display=display, summary="Command is not allowlisted.")
        start = time.monotonic()
        try:
            env = command_env()
            env.setdefault("CI", "1")
            # Use this checkout's package roots, never the platform's import path.
            roots = [workspace]
            if (Path(workspace) / "src").is_dir():
                roots.append(str(Path(workspace) / "src"))
            env["PYTHONPATH"] = os.pathsep.join(roots)
            process = subprocess.run(list(command), cwd=workspace, env=env, capture_output=True, text=True,
                                     timeout=self.timeout, check=False, shell=False)
            duration = time.monotonic() - start
            output = (process.stdout or "") + "\n" + (process.stderr or "")
            status = "PASSED" if process.returncode == 0 else "FAILED"
            evidence = {}
            summary = "Completed successfully (exit code 0)." if process.returncode == 0 else f"Command failed with exit code {process.returncode}."
            is_pytest = "pytest" in command or any(Path(str(part)).name.lower() in {"pytest", "pytest.exe"} for part in command)
            if is_pytest and process.returncode == 5:
                status = "UNAVAILABLE"
                summary = "PyTest did not collect any tests; repository test execution is unavailable, not failed."
                evidence = {"failure_category": "ENVIRONMENT_TOOLING", "reason_code": "NO_TESTS_COLLECTED", "state": "NOT_DEFINED"}
            elif is_pytest and process.returncode in {2, 3, 4} and re.search(
                r"(?:error collecting|modulenotfounderror|importerror|unrecognized arguments|internalerror)",
                output, re.IGNORECASE,
            ):
                status = "UNAVAILABLE"
                summary = "PyTest could not execute because collection or environment setup failed."
                evidence = {"failure_category": "ENVIRONMENT_TOOLING", "reason_code": "PYTEST_ENVIRONMENT_FAILURE"}
            elif is_pytest and process.returncode != 0:
                evidence = {"failure_category": "TEST_FAILURE", "reason_code": "PYTEST_TEST_FAILURE"}
            return ValidationCheck(check_type="command", status=status, evidence=evidence,
                                   command=list(command), command_display=display, tool=command[0], exit_code=process.returncode,
                                   duration=duration, duration_seconds=duration, workspace=workspace, variant=variant,
                                   verifies=verifies, summary=summary,
                                   stdout_excerpt=redact(process.stdout)[-self.output_limit:], stderr_excerpt=redact(process.stderr)[-self.output_limit:])
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return ValidationCheck(check_type="command", status="ERROR", command=list(command), command_display=display,
                                   tool=command[0], duration=duration, duration_seconds=duration, workspace=workspace,
                                   variant=variant, verifies=verifies, summary="Command timed out.",
                                   warnings=["Timeout enforced by safe command runner."])
        except OSError as exc:
            duration = time.monotonic() - start
            return ValidationCheck(check_type="command", status="UNAVAILABLE", command=list(command), command_display=display,
                                   tool=command[0], duration=duration, duration_seconds=duration, workspace=workspace,
                                   variant=variant, verifies=verifies, summary="Tool is unavailable.", warnings=[redact(str(exc))])


class ValidationService:
    CHECKS = ("build", "unit", "regression", "targeted_integration", "lint", "security", "dependency", "snyk")
    BLOCKING = {"build", "unit", "regression", "targeted_integration", "lint", "security", "dependency", "dependency_vulnerability", "sonarqube", "snyk", "jenkins"}
    VERIFY = {
        "sonarqube": "Static Analysis / Code Quality: configured quality gate must pass.",
        "snyk": "Dependency/Security Scan: introduced Critical/High block; Medium/Low advisory.",
        "build": "Compiles Python source to bytecode to detect syntax errors; it is not a package build.",
        "unit": "Runs the repository-defined automated test command.",
        "regression": "Runs executable AI-generated regression tests against the candidate.",
        "targeted_integration": "Runs AI-generated cross-component tests against the candidate when generated.",
        "integration": "Runs a distinct repository-defined integration suite when one exists.",
        "lint": "Runs Ruff against the complete isolated candidate workspace.",
        "security": "Runs Bandit and blocks deployment only when High-severity findings exist.",
        "dependency": "Parses dependency manifests for readability; it does not scan vulnerabilities.",
        "dependency_vulnerability": "Runs pip-audit when installed; otherwise the scan is not configured.",
    }

    def __init__(self, runner=None):
        self.runner = runner or SafeCommandRunner()

    @staticmethod
    def normalize_root(root: str | Path) -> Path:
        if root is None:
            raise ValueError("Repository root is required")
        candidate = Path(root).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"Repository root does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"Repository root is not a directory: {candidate}")
        return candidate

    @staticmethod
    def _generated_command(profile: ProjectProfile, paths: list[str], root: Path | None = None) -> list[str]:
        if not paths:
            return []
        if profile.framework == "pytest":
            plugin = ["-p", "test.conftest"] if root and (root / "test" / "conftest.py").is_file() else []
            return [sys.executable, "-m", "pytest", *plugin, *paths, "-q"]
        if profile.framework in {"vitest", "jest", "mocha"}:
            return ["npm", "test", "--", *paths]
        return []

    def _manifest_check(self, root: Path, variant: str) -> ValidationCheck:
        started = time.monotonic()
        manifests = [root / name for name in ("package.json", "pyproject.toml", "requirements.txt", "Cargo.toml")]
        present = [path for path in manifests if path.exists()]
        errors = []
        for path in present:
            try:
                if path.name == "package.json":
                    json.loads(path.read_text(encoding="utf-8"))
                elif path.suffix == ".toml":
                    tomllib.loads(path.read_text(encoding="utf-8"))
                else:
                    path.read_text(encoding="utf-8")
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f"{path.name}: {type(exc).__name__}")
        duration = time.monotonic() - started
        status = "UNAVAILABLE" if not present else "FAILED" if errors else "PASSED"
        summary = "No dependency manifest detected." if not present else "Dependency manifest parsing failed." if errors else f"Parsed {len(present)} dependency manifest(s)."
        return ValidationCheck(check_type="dependency", status=status, duration=duration, duration_seconds=duration,
                               workspace=str(root), variant=variant, verifies=self.VERIFY["dependency"],
                               command_display="In-process dependency manifest parser", summary=summary, warnings=errors)

    def _security_check(self, root: Path, variant: str) -> ValidationCheck:
        if not executable("bandit") and not importlib.util.find_spec("bandit"):
            return ValidationCheck(check_type="security", status="UNAVAILABLE", workspace=str(root), variant=variant,
                                   verifies=self.VERIFY["security"], command_display="bandit -r . -f txt",
                                   summary="Bandit security analysis is not configured.")
        raw = self.runner.run([sys.executable, "-m", "bandit", "-r", ".", "-f", "txt"], root, variant=variant, verifies=self.VERIFY["security"])
        output = (raw.stdout_excerpt or "") + "\n" + (raw.stderr_excerpt or "")
        import re
        counts = {level: int(match.group(1)) if (match := re.search(rf"{level}:\s*(\d+)", output, re.I)) else 0
                  for level in ("High", "Medium", "Low")}
        if not any(re.search(rf"{level}:\s*\d+", output, re.I) for level in counts):
            return raw.model_copy(update={"check_type": "security", "status": "ERROR", "summary": "Bandit returned unreadable severity evidence."})
        gate = "FAILED" if counts["High"] else "PASSED"
        return raw.model_copy(update={"check_type": "security", "status": gate, "severity_counts": counts,
                                      "summary": "Security deployment gate blocks High-severity findings; Medium and Low remain advisories."})

    def validate(self, root, requested=None, generated_test_files=None, generated_integration_files=None, *, variant="ORIGINAL", ci_context=None, baseline_root=None):
        variant = normalize_validation_variant(variant)
        root = self.normalize_root(root)
        if variant == "AI_CANDIDATE" and os.getenv("CI_EXECUTOR", "local").lower() == "jenkins":
            from .jenkins import JenkinsService
            return JenkinsService().validate(root, baseline_root, ci_context or {}, generated_test_files or [], generated_integration_files or [])
        profile = detect_project(root)
        checks = {}
        blocked_by = None
        regression_paths = [str(path) for path in (generated_test_files or []) if str(path) != "proposal.patch"]
        integration_paths = [str(path) for path in (generated_integration_files or []) if str(path) != "proposal.patch"]
        configured_checks = (*self.CHECKS, "sonarqube") if os.getenv("SONAR_HOST_URL") or os.getenv("SONAR_TOKEN") else self.CHECKS
        for name in requested or configured_checks:
            if blocked_by:
                checks[name] = ValidationCheck(check_type=name, status="SKIPPED", workspace=str(root), variant=variant,
                                               verifies=self.VERIFY.get(name, ""), summary=f"SKIPPED — blocked by previous failure: {blocked_by}.")
                continue
            if name == "sonarqube":
                check = ScannerService().sonar(root, variant, ci_context)
            elif name == "snyk":
                check = ScannerService().snyk(root, baseline_root, variant)
            elif name == "dependency":
                check = self._manifest_check(root, variant)
            elif name == "dependency_vulnerability":
                command = ["pip-audit", "--format=json"] if profile.language == "Python" and shutil.which("pip-audit") else []
                check = (self.runner.run(command, root, variant=variant, verifies=self.VERIFY[name]) if command else
                         ValidationCheck(check_type=name, status="UNAVAILABLE", workspace=str(root), variant=variant,
                                         verifies=self.VERIFY[name], command_display="pip-audit --format=json",
                                         summary="Dependency vulnerability scan: Not configured"))
            elif name == "security":
                check = self._security_check(root, variant)
            elif name == "lint":
                command = [sys.executable, "-m", "ruff", "check", "."] if variant == "AI_CANDIDATE" and profile.language == "Python" and importlib.util.find_spec("ruff") else []
                check = (self.runner.run(command, root, variant=variant, verifies=self.VERIFY[name]).model_copy(update={"check_type": name}) if command else
                         ValidationCheck(check_type=name, status="UNAVAILABLE", workspace=str(root), variant=variant,
                                         verifies=self.VERIFY[name], command_display="python -m ruff check .",
                                         summary="Ruff is enforced for isolated AI candidates only." if variant != "AI_CANDIDATE" else "Ruff static analysis is not configured."))
            else:
                command = profile.commands.get(name, [])
                if name == "unit" and profile.language == "Python":
                    command = [sys.executable, "-m", "pytest", "-q", *["--ignore=" + path for path in set(regression_paths + integration_paths)]]
                    if any((root / folder / "test.py").is_file() for folder in ("test", "tests")):
                        command.extend(["-o", "python_files=test_*.py *_test.py test.py"])
                if name == "regression":
                    command = self._generated_command(profile, regression_paths, root)
                elif name == "targeted_integration":
                    command = self._generated_command(profile, integration_paths, root)
                if command:
                    check = self.runner.run(command, root, variant=variant, verifies=self.VERIFY.get(name, ""))
                    check = check.model_copy(update={"check_type": name})
                else:
                    reason = ("No AI-generated targeted integration tests were supplied; this check is not applicable."
                              if name == "targeted_integration" else
                              "This repository does not define a distinct command for this check."
                              if name in {"regression", "integration"} else
                              "No trusted local command is configured for this check.")
                    check = ValidationCheck(check_type=name, status="UNAVAILABLE", workspace=str(root), variant=variant,
                                            verifies=self.VERIFY.get(name, ""), summary=reason)
            if name == "build" and check.status == "PASSED" and (root / "setup.py").is_file():
                # Repository-declared native modules must exist before importing tests.
                setup_source = (root / "setup.py").read_text(encoding="utf-8", errors="replace")
                if "ext_modules" in setup_source:
                    native_runner = (SafeCommandRunner(timeout=max(300, self.runner.timeout), output_limit=self.runner.output_limit)
                                     if type(self.runner) is SafeCommandRunner else self.runner)
                    native = native_runner.run([sys.executable, str(Path(__file__).with_name("native_build.py"))], root,
                                             variant=variant, verifies="Build repository-declared native extensions before tests.")
                    native.evidence["python_compilation"] = check.model_dump()
                    check = native.model_copy(update={"check_type": "build"})
            if check.status == "UNAVAILABLE" and name in {"integration", "targeted_integration"}:
                check.evidence["state"] = "NOT_DEFINED"
            if variant == "AI_CANDIDATE" and name == "unit" and check.evidence.get("reason_code") == "PYTEST_ENVIRONMENT_FAILURE":
                check = check.model_copy(update={"status": "ERROR"})
            checks[name] = check
            if name in self.BLOCKING and check.status in {"FAILED", "ERROR"}:
                blocked_by = name
        summary = self.aggregate(checks)
        if ci_context:
            summary.ci = {**ci_context, "executor": "local", "status": "PASS" if summary.status == "passed" else "FAIL"}
        return summary

    def aggregate(self, checks):
        values = list(checks.values())
        failed = [key for key, value in checks.items() if value.status == "FAILED"]
        errors = [key for key, value in checks.items() if value.status == "ERROR"]
        blocking = [key for key in failed + errors if key in self.BLOCKING]
        status = ("failed" if blocking else "passed" if any(value.status == "PASSED" for value in values) and not failed and not errors
                  else "unavailable" if values and all(value.status == "UNAVAILABLE" for value in values) else "not_run")
        return ValidationSummary(status=status, overall_status=status.upper(), checks={key: value.model_dump() for key, value in checks.items()},
                                 total_checks=len(values), passed=sum(value.status == "PASSED" for value in values),
                                 failed=len(failed), skipped=sum(value.status == "SKIPPED" for value in values),
                                 unavailable=sum(value.status == "UNAVAILABLE" for value in values), error_count=len(errors),
                                 blocking_failures=blocking, errors=[checks[key].summary for key in blocking])

    def run_command(self, name, command, cwd=None):
        if name not in self.CHECKS:
            raise ValueError(f"Unsupported validation check: {name}")
        safe_cwd = self.normalize_root(cwd or Path.cwd())
        return self.aggregate({name: self.runner.run(command, safe_cwd, verifies=self.VERIFY.get(name, "")).model_copy(update={"check_type": name})})

    def not_run(self):
        return ValidationSummary(status="not_run")
