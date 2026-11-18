from __future__ import annotations

import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .ci_tools import redact
from backend.app.schemas.models import TestGenerationResult


_DIFF_PATH = re.compile(r"^(?:---|\+\+\+) (.*)$")


@dataclass
class CandidateWorkspace:
    original: Path
    path: Path

    def cleanup(self) -> None:
        if not self.path.exists():
            return
        # Windows scanners and recently-finished subprocesses can briefly keep
        # directory handles open. Cleanup must never replace the authoritative
        # workflow result with a transient disposal error.
        for delay in (0.0, 0.2, 0.8):
            if delay:
                time.sleep(delay)
            try:
                shutil.rmtree(self.path)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                continue


class PatchValidationError(ValueError):
    """A safe, structured patch rejection suitable for retry feedback."""

    def __init__(self, message: str, *, diagnostic: dict | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


class CandidateWorkspaceService:
    """Create disposable candidates and apply bounded repair/test artifacts."""

    def create(self, root: str | Path) -> CandidateWorkspace:
        source = Path(root).expanduser().resolve()
        if not source.exists() or not source.is_dir():
            raise ValueError("Candidate source root must be an existing directory")
        target = Path(tempfile.mkdtemp(prefix="change-impact-candidate-", dir=str(source.parent)))
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache",
                "__pycache__", ".venv", ".venv-*", "node_modules",
                "build", "dist", ".scannerwork", ".tox",
            ),
        )
        return CandidateWorkspace(source, target)

    @classmethod
    def production_snapshot(cls, root: Path) -> dict[str, str]:
        import hashlib
        from .github import SOURCE_SUFFIXES
        snapshot = {}
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if any(part.startswith(".") or part in {"build", "dist", "node_modules", "__pycache__"} for part in relative.parts):
                continue
            if not cls._is_test_path(relative.as_posix()):
                snapshot[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return snapshot

    @staticmethod
    def _safe_relative(value: str) -> Path:
        value = value.strip()
        if not value or value == "/dev/null" or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
            raise ValueError("Patch path is outside the repository")
        path = Path(value)
        if ".." in path.parts or "\\" in value or any(part.startswith('.') for part in path.parts):
            raise ValueError("Patch path traversal is not allowed")
        return path

    def apply_patch(self, candidate: CandidateWorkspace, patch: str, *, test_only: bool = False) -> list[str]:
        if not patch:
            raise PatchValidationError("Repair patch is empty", diagnostic={"reason": "empty patch"})
        from .repair import normalize_patch, _validate_unified_diff
        patch = normalize_patch(patch)
        paths: list[str] = []
        for line in patch.splitlines():
            match = _DIFF_PATH.match(line)
            if match:
                raw = match.group(1).split("\t", 1)[0]
                if raw == "/dev/null":
                    continue
                if raw.startswith(("a/", "b/")):
                    raw = raw[2:]
                rel = self._safe_relative(raw)
                normalized = rel.as_posix()
                if test_only and not self._is_test_path(str(rel)):
                    raise ValueError(f"Generated-test patch touched production source: {rel}")
                if normalized not in paths:
                    paths.append(normalized)
                target = candidate.path / rel
                if not target.resolve().is_relative_to(candidate.path.resolve()) or target.is_symlink():
                    raise ValueError('Patch path escapes candidate workspace')
        _validate_unified_diff(patch)
        if re.search(r'^(?:new|old) mode 120000|^new file mode 120000', patch, re.MULTILINE):
            raise ValueError('Symlink patches are not permitted')
        if not paths:
            raise ValueError("Repair patch contains no bounded file paths")
        # Use the fixed git executable and fixed arguments; the provider cannot
        # supply a command.  --check prevents partial application.
        import subprocess
        patch_bytes = patch.encode("utf-8")
        checks = []
        selected_mode = None
        for autocrlf in ("false", "true"):
            command = ["git", "-c", f"core.autocrlf={autocrlf}", "apply", "--whitespace=nowarn"]
            check = subprocess.run([*command, "--check", "-"], input=patch_bytes,
                                   capture_output=True, cwd=candidate.path, shell=False, check=False)
            checks.append((autocrlf, check))
            if check.returncode == 0:
                selected_mode = autocrlf
                break
        if selected_mode is None:
            diagnostics = []
            for autocrlf, check in checks:
                detail = (check.stderr or check.stdout).decode("utf-8", errors="replace")
                if detail.strip(): diagnostics.append(f"core.autocrlf={autocrlf}: {detail.strip()}")
            stderr = redact("\n".join(diagnostics))[-2000:]
            head_sha = None
            try:
                head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=candidate.original,
                                      capture_output=True, text=True, shell=False, check=False)
                head_sha = head.stdout.strip() if head.returncode == 0 else None
            except OSError:
                pass
            affected = paths[0] if paths else None
            raise PatchValidationError("Repair patch failed validation: " + stderr,
                diagnostic={"affected_file": affected, "reason": "git apply --check rejected patch",
                            "git_apply_check_stderr": stderr, "file_exists": bool(affected and (candidate.original / affected).exists()),
                            "expected_head_sha": head_sha})
        git_apply = ["git", "-c", f"core.autocrlf={selected_mode}", "apply", "--whitespace=nowarn"]
        applied = subprocess.run([*git_apply, "-"], input=patch_bytes,
                                 capture_output=True, cwd=candidate.path, shell=False, check=False)
        if applied.returncode:
            raise ValueError("Repair patch could not be applied")
        return paths

    @staticmethod
    def _is_test_path(name: str) -> bool:
        normalized = name.replace("\\", "/").lower()
        leaf = normalized.rsplit("/", 1)[-1]
        return (normalized.startswith(("tests/", "test/", "spec/", "__tests__/"))
                or "/tests/" in normalized or "/__tests__/" in normalized
                or leaf.startswith("test_") or leaf.endswith("_test.py")
                or ".test." in leaf or ".spec." in leaf)

    def comparison(self, candidate: CandidateWorkspace, paths: list[str]) -> list[dict]:
        """Capture review evidence before the disposable workspace is removed."""
        files: list[dict] = []
        for name in paths:
            rel = self._safe_relative(name)
            original_path, candidate_path = candidate.original / rel, candidate.path / rel
            original = original_path.read_text(encoding="utf-8", errors="replace") if original_path.is_file() else ""
            proposed = candidate_path.read_text(encoding="utf-8", errors="replace") if candidate_path.is_file() else ""
            import difflib
            delta = list(difflib.unified_diff(original.splitlines(), proposed.splitlines(), fromfile='a/'+name, tofile='b/'+name))
            files.append({
                "file": str(rel).replace("\\", "/"),
                "original_code": original,
                "candidate_code": proposed,
                "original_lines": len(original.splitlines()),
                "proposed_lines": len(proposed.splitlines()),
                "added_lines": sum(line.startswith('+') and not line.startswith('+++') for line in delta),
                "removed_lines": sum(line.startswith('-') and not line.startswith('---') for line in delta),
                "unified_diff": '\n'.join(delta),
            })
        return files

    def add_generated_tests(self, candidate: CandidateWorkspace, generated: TestGenerationResult) -> list[str]:
        def iter_generated_files():
            yielded: set[str] = set()
            for name, content in generated.generated_test_files.items():
                yielded.add(name.replace("\\", "/"))
                yield name, content
            for item in [*generated.generated_tests, *generated.generated_integration_tests]:
                name = item.get("test_file") or item.get("path")
                if not name or name.replace("\\", "/") in yielded:
                    continue
                content = item.get("generated_code") or item.get("content")
                if content is None:
                    continue
                yielded.add(name.replace("\\", "/"))
                yield name, content

        written: list[str] = []
        for name, content in iter_generated_files():
            rel = self._safe_relative(name)
            if rel.name == "proposal.patch":
                continue
            if rel.suffix != ".py":
                raise ValueError(f'Generated tests cannot target production files; generated-test source touched production source: {rel}')
            if not self._is_test_path(rel.as_posix()):
                raise ValueError(f'Generated tests cannot target production files; generated-test source touched production source: {rel}')
            target = (candidate.path / rel).resolve()
            if candidate.path not in target.parents:
                raise ValueError("Generated test path escapes candidate workspace")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(rel.as_posix())
        return written
