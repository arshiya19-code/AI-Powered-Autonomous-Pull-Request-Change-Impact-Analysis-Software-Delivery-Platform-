from pathlib import Path

import pytest

from backend.app.services.candidate import CandidateWorkspaceService
from backend.app.services.validation import ValidationService, detect_project


def test_validation_accepts_string_and_path_roots(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    service = ValidationService()
    assert service.validate(str(tmp_path), requested=["dependency"]).status == "passed"
    assert service.validate(tmp_path, requested=["dependency"]).status == "passed"
    assert detect_project(tmp_path).language == "Python"


def test_validation_rejects_nonexistent_root(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        ValidationService().validate(tmp_path / "missing")


def test_candidate_patch_and_cleanup_leave_original_unchanged(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("value = 1\n")
    original = source.read_text()
    patch = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
    service = CandidateWorkspaceService()
    candidate = service.create(tmp_path)
    try:
        assert service.apply_patch(candidate, patch) == ["app.py"]
        assert (candidate.path / "app.py").read_text() == "value = 2\n"
        assert source.read_text() == original
    finally:
        candidate.cleanup()
    assert not candidate.path.exists()


def test_candidate_patch_reports_each_file_once(tmp_path):
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    candidate = CandidateWorkspaceService().create(tmp_path)
    patch = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
    try:
        assert CandidateWorkspaceService().apply_patch(candidate, patch) == ["app.py"]
    finally:
        candidate.cleanup()


def test_candidate_rejects_path_traversal(tmp_path):
    service = CandidateWorkspaceService()
    candidate = service.create(tmp_path)
    try:
        with pytest.raises(ValueError, match="outside|traversal"):
            service.apply_patch(candidate, "--- a/../escape.py\n+++ b/../escape.py\n@@ -0,0 +1 @@\n++bad\n")
    finally:
        candidate.cleanup()
