from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from backend.app.schemas.models import RepairResult, RepairSummary

_RESPONSE_PREVIEW_LIMIT = 1200
_FENCED_JSON = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_PATCH_FENCE = re.compile(r"```(?:diff|patch)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_SENSITIVE_RESPONSE_TEXT = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]+", re.IGNORECASE),
    re.compile(r'(?i)(["\']?(?:api[_-]?key|authorization|token|secret)["\']?\s*[:=]\s*)["\']?[^\s,"\'}]+'),
)
_REPAIR_RESPONSE_KEYS = {
    "explanation", "confidence", "affected_files", "patch",
    "regression_test", "integration_test",
}
_UNIFIED_DIFF_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)
_REPAIR_JSON_SCHEMA = {
    "name": "repair_proposal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "affected_files": {"type": "array", "items": {"type": "string"}},
            "patch": {"type": "string"},
            "regression_test": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "reason": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["path", "target", "reason", "source"],
                "additionalProperties": False,
            },
            "integration_test": {
                "type": ["object", "null"],
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "dependency_path": {"type": "string"},
                    "reason": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["path", "target", "dependency_path", "reason", "source"],
                "additionalProperties": False,
            },
        },
        "required": ["explanation", "confidence", "affected_files", "patch", "regression_test", "integration_test"],
        "additionalProperties": False,
    },
}
_PATCH_RESPONSE_SCHEMA = {
    "name": "production_repair", "strict": True,
    "schema": {
        "type": "object",
        "properties": {key: value for key, value in _REPAIR_JSON_SCHEMA["schema"]["properties"].items()
                       if key in {"explanation", "confidence", "affected_files", "patch"}},
        "required": ["explanation", "confidence", "affected_files", "patch"],
        "additionalProperties": False,
    },
}
_TEST_RESPONSE_SCHEMA = {
    "name": "generated_tests",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "tests": {
                "type": "array", "minItems": 1, "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "type": {"type": "string", "enum": ["regression", "integration"]},
                        "purpose": {"type": "string"},
                        "content": {"type": "string"},
                        "dependency_path": {"type": "string"},
                    },
                    "required": ["path", "type", "purpose", "content", "dependency_path"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tests"],
        "additionalProperties": False,
    },
}


class ProviderResponseError(RuntimeError):
    """A safe, user-readable provider failure with bounded diagnostics."""


class ProviderTransportError(ProviderResponseError):
    """A provider request failure that is distinct from patch semantics."""

    def __init__(self, message: str, *, error_type: str, status_code: int | None = None,
                 retryable: bool = False, retry_after_seconds: float | None = None,
                 transport_attempts: int = 1):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.transport_attempts = transport_attempts


class PatchContractError(ProviderResponseError):
    """A structured provider response whose patch is unsafe to apply."""

    def __init__(self, message: str, *, patch: str, data: Mapping[str, Any]):
        super().__init__(message)
        self.patch = patch
        self.data = dict(data)


def _safe_response_preview(raw: bytes) -> str:
    from .ci_tools import redact
    text = redact(raw.decode("utf-8", errors="replace"))
    for pattern in _SENSITIVE_RESPONSE_TEXT:
        text = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    text = text[:_RESPONSE_PREVIEW_LIMIT]
    if len(raw) > _RESPONSE_PREVIEW_LIMIT:
        text += "…[truncated]"
    return text or "<empty>"


def _response_diagnostic(status: int | None, content_type: str | None, raw: bytes) -> str:
    return f"HTTP {status if status is not None else 'unknown'}; content-type {content_type or '<missing>'}; body {_safe_response_preview(raw)}"


def normalize_patch(patch: str) -> str:
    """Remove harmless model transport formatting; never invent diff content."""
    if not isinstance(patch, str):
        raise ProviderResponseError("Provider patch must be a unified-diff string.")
    value = patch.replace("\r\n", "\n").replace("\r", "\n")
    fenced = _PATCH_FENCE.search(value)
    if fenced:
        value = fenced.group(1)
    value = value.replace("\\r\\n", "\n").replace("\\n", "\n")
    lines = value.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(("diff --git ", "--- "))]
    if starts:
        lines = lines[starts[0]:]
    while lines and not lines[0].strip():
        lines.pop(0)
    # A single-space line is valid unified-diff context for a blank source
    # line. Remove transport blank lines, never semantic hunk context.
    while lines and (lines[-1] == "" or lines[-1].strip() == "```"):
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def _validate_test_artifact(value: Any, *, integration: bool) -> None:
    if integration and value is None:
        return
    label = "integration_test" if integration else "regression_test"
    required = {"path", "target", "reason", "source", *( ["dependency_path"] if integration else [])}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ProviderResponseError(f"Provider {label} must contain exactly {', '.join(sorted(required))}.")
    for key in required:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ProviderResponseError(f"Provider {label}.{key} must be a non-empty string.")
    normalized = value["path"].replace("\\", "/")
    if not normalized.startswith("tests/ai_generated/") or not normalized.endswith(".py") or ".." in normalized.split("/"):
        raise ProviderResponseError(f"Provider {label}.path must be a safe Python path under tests/ai_generated/.")


def normalize_confidence(value: Any) -> float | None:
    """Normalize display-only confidence metadata without rejecting repair evidence."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            number = float(text)
        except ValueError:
            return None
        if is_percent:
            number /= 100
    else:
        return None
    return number if 0 <= number <= 1 else None


def _validate_repair_contract(data: Mapping[str, Any], *, require_tests: bool = True) -> None:
    """Reject repair responses that are not the deliberately small JSON contract."""
    legacy = {"explanation", "confidence", "affected_files", "patch"}
    expected = _REPAIR_RESPONSE_KEYS if require_tests or "regression_test" in data or "integration_test" in data else legacy
    if set(data) != expected:
        raise ProviderResponseError("Provider repair JSON must contain exactly " + ", ".join(sorted(expected)) + ".")
    if not isinstance(data["explanation"], str):
        raise ProviderResponseError("Provider repair explanation must be a string.")
    if data["confidence"] is not None and (
        isinstance(data["confidence"], bool)
        or not isinstance(data["confidence"], (int, float))
        or not 0 <= data["confidence"] <= 1
    ):
        raise ProviderResponseError("Normalized provider confidence must be a number from 0.0 to 1.0 or null.")
    if not isinstance(data["affected_files"], list) or not all(isinstance(path, str) for path in data["affected_files"]):
        raise ProviderResponseError("Provider repair affected_files must be an array of paths.")
    if not isinstance(data["patch"], str):
        raise ProviderResponseError("Provider patch must be a unified-diff string.")
    if expected == _REPAIR_RESPONSE_KEYS:
        _validate_test_artifact(data["regression_test"], integration=False)
        _validate_test_artifact(data["integration_test"], integration=True)


def _validate_unified_diff(patch: str) -> None:
    """Require a complete git-compatible production patch before candidate creation."""
    if not patch:
        return
    if not patch.startswith("diff --git a/"):
        raise ProviderResponseError("Provider patch must start with a git-compatible 'diff --git a/... b/...' header.")
    if not re.search(r"^--- a/.+", patch, re.MULTILINE) or not re.search(r"^\+\+\+ b/.+", patch, re.MULTILINE):
        raise ProviderResponseError("Provider patch must include both --- a/<path> and +++ b/<path> headers.")
    if not _UNIFIED_DIFF_HUNK.search(patch):
        raise ProviderResponseError("Provider patch must include at least one complete unified-diff hunk. Bare @@ is invalid: output numeric ranges such as @@ -1,3 +1,3 @@ with the actual source line numbers and exact old/new line counts.")
    lines = patch.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("@@"):
            continue
        match = re.match(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
        if not match:
            raise ProviderResponseError("Invalid unified-diff hunk header: " + line)
        old_count = new_count = 0
        for body in lines[index + 1:]:
            if body.startswith(("@@", "diff --git ")):
                break
            if body.startswith(" "):
                old_count += 1
                new_count += 1
            elif body.startswith("-"):
                old_count += 1
            elif body.startswith("+"):
                new_count += 1
            elif body.startswith("\\ No newline at end of file"):
                continue
            else:
                raise ProviderResponseError("Invalid unified-diff body line; blank context lines must begin with one space.")
        expected = (int(match.group(1) or 1), int(match.group(2) or 1))
        if expected != (old_count, new_count):
            raise ProviderResponseError(
                f"Hunk {line} declares old/new counts {expected}, but its body contains "
                f"{old_count}/{new_count}. Regenerate a complete patch with accurate counts; no automatic recount is performed."
            )


def _compact_repair_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider input repair-specific and bounded; analysis stays local."""
    impact, risk, decision = context.get("impact") or {}, context.get("risk") or {}, context.get("decision") or {}
    validation = context.get("validation") or {}
    code_failures: dict[str, Any] = {}
    environment_evidence: dict[str, Any] = {}
    for name, raw in (validation.get("checks") or {}).items():
        check = raw or {}
        evidence = check.get("evidence") or {}
        item = {
            "status": check.get("status"), "command": check.get("command_display"),
            "summary": check.get("summary"), "stdout": check.get("stdout_excerpt"),
            "stderr": check.get("stderr_excerpt"), "reason_code": evidence.get("reason_code"),
        }
        legacy_no_tests = check.get("exit_code") == 5 and "no tests ran" in str(check.get("stdout_excerpt") or "").lower()
        if legacy_no_tests:
            item["reason_code"] = "NO_TESTS_COLLECTED"
            environment_evidence[name] = item
        elif evidence.get("failure_category") == "TEST_FAILURE":
            code_failures[name] = item
        elif evidence.get("failure_category") == "ENVIRONMENT_TOOLING":
            environment_evidence[name] = item
    base_sources = {}
    budget = 3500
    for name, source in (context.get("base_source_context") or {}).items():
        if len(source) <= budget:
            base_sources[name] = source
            budget -= len(source)
    compact = {
        "repository": context.get("repository"), "base_sha": context.get("base_sha"), "head_sha": context.get("head_sha"),
        "base_source_snippets": base_sources,
        "omitted_base_source_files": sorted(set(context.get("base_source_context") or {}) - set(base_sources)),
        "repository_layout": context.get("repository_layout") or [],
        "shap": {k:v for k,v in (context.get("shap") or {}).items() if k in {"shap_status", "top_risk_factors", "protective_factors"}}, "dependency_changes": context.get("dependency_changes"),
        "pr_title": context.get("pr_title"), "pr_body": context.get("pr_body"), "unified_diff": context.get("diff"),
        "changed_files": context.get("changed_files") or [], "changed_symbols": [{k: v for k, v in item.items() if k in {"name", "file", "type", "change_type", "start_line", "end_line"}} if isinstance(item, dict) else item for item in (context.get("changed_symbols") or [])[:20]],
        "relevant_original_source_snippets": dict(context.get("source_context") or {}),
        "head_file_line_counts": {name: len(source.splitlines()) for name, source in (context.get("source_context") or {}).items()},
        "head_function_line_numbers": {name: {str(i): line for i, line in enumerate(source.splitlines(), 1) if line.lstrip().startswith(("def ", "async def "))} for name, source in (context.get("source_context") or {}).items()},
        "top_impacted_symbols": [({k:v for k,v in item.items() if k in {"name","file","id"}} if isinstance(item, dict) else item) for item in (context.get("impacted_symbols") or [])[:10]], "critical_nodes": [{k:v for k,v in item.items() if k in {"id", "name", "file"}} for item in (context.get("critical_nodes") or [])[:5]],
        "demonstrated_dependency_paths": (context.get("propagation_paths") or [])[:3],
        "structural_impact_summary": {key: impact.get(key) for key in ("impact_score", "blast_radius", "maximum_depth", "impacted_nodes", "critical_nodes") if key in impact},
        "catboost_label_probability": {key: risk.get(key) for key in ("risk_label", "label", "risk_probability", "probability") if key in risk},
        "decision_engine_result": decision,
        "original_code_test_failures": code_failures,
        "original_environment_validation": environment_evidence,
        "previous_retry_failure_evidence": _compact_failure_feedback(context.get("failure_feedback") or {}),
    }
    # Keep diagnostics useful without allowing full test logs or explanation
    # payloads to dominate a small production repair request.
    for evidence_bucket in (code_failures, environment_evidence):
        for evidence in evidence_bucket.values():
            for key in ("stdout", "stderr", "summary"):
                if isinstance(evidence.get(key), str):
                    evidence[key] = evidence[key][-700:]
    if isinstance(compact["decision_engine_result"], dict):
        compact["decision_engine_result"] = {key: value for key, value in compact["decision_engine_result"].items()
                                             if key in {"decision", "risk_level"}}
    for key in ("top_risk_factors", "protective_factors"):
        if isinstance(compact["shap"].get(key), list):
            compact["shap"][key] = compact["shap"][key][:3]
    compact["head_source_window_ranges"] = {}
    for name, source in list(compact["relevant_original_source_snippets"].items()):
        regions = [item for item in context.get("changed_symbols", []) if isinstance(item, dict)
                   and item.get("file") == name and item.get("start_line") and item.get("end_line")]
        if regions and len(source) > 1800:
            lines = source.splitlines()
            first = max(1, min(int(item["start_line"]) for item in regions) - 2)
            last = min(len(lines), max(int(item["end_line"]) for item in regions) + 2)
            compact["relevant_original_source_snippets"][name] = "\n".join(lines[first - 1:last])
            compact["head_source_window_ranges"][name] = {"first_line": first, "last_line": last}
    # Bound the complete envelope, not just each source bucket independently.
    # Whole files are removed in reverse selection order; exact diff/head text
    # is never rewritten. Record omissions so the model cannot assume coverage.
    limit = int(os.getenv("LLM_CONTEXT_MAX_CHARS", "11000"))
    compact["omitted_context"] = []
    def size():
        return len(json.dumps(compact, ensure_ascii=False, default=str))
    for bucket in ("base_source_snippets", "relevant_original_source_snippets"):
        for name in reversed(list(compact[bucket])):
            if size() <= limit:
                break
            if bucket == "relevant_original_source_snippets" and len(compact[bucket]) <= 1:
                break
            compact[bucket].pop(name)
            compact["omitted_context"].append(f"{bucket}:{name}")
    numbered = {}
    for name, source in compact["relevant_original_source_snippets"].items():
        regions = [item for item in context.get("changed_symbols", [])
                   if isinstance(item, dict) and item.get("file") == name]
        start = max(1, min((int(item.get("start_line") or 1) for item in regions), default=1) - 3)
        lines = (context.get("source_context") or {}).get(name, source).splitlines()
        numbered[name] = "\n".join(f"{i} | {lines[i - 1]}" for i in range(start, min(len(lines), start + 24) + 1))
        if sum(len(value) for value in numbered.values()) >= 1800:
            break
    compact["numbered_head_source_windows"] = numbered
    return compact


def _compact_failure_feedback(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain actionable retry facts without resending entire validation objects."""
    if not isinstance(value, Mapping):
        return {}
    compact = {}
    for key in (
        "patch_rejection_reason", "repair_instruction", "candidate_error", "patch_diagnostic",
        "failed_test_names", "failed_commands", "validation_failures",
        "expected", "actual", "assertion_or_traceback_excerpt",
        "prior_candidate_patch", "prior_repair_explanation",
    ):
        if key == "prior_candidate_patch" and (value.get("patch_rejection_reason") or value.get("patch_diagnostic")):
            continue
        item = value.get(key)
        if item in (None, "", [], {}):
            continue
        if isinstance(item, str):
            item = item[-1600:]
        compact[key] = item
    return compact


def _provider_error_reason(status: int | None, raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    lowered = text.lower()
    if status in {401, 403} or any(term in lowered for term in ("invalid api key", "unauthorized", "authentication")):
        return "authentication failed"
    if status == 402 or any(term in lowered for term in ("insufficient credit", "insufficient balance", "more credits")):
        return "insufficient credits"
    if status == 429 or "rate limit" in lowered or "rate_limit" in lowered:
        return "rate limited"
    if status in {408, 504} or "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if status == 404 or any(term in lowered for term in ("invalid model", "model not found", "no endpoints found", "model unavailable")):
        return "model unavailable"
    if status is not None and status >= 500:
        return "model unavailable"
    return "provider request failed"


def _retry_after_seconds(headers: Any, raw: bytes) -> float | None:
    """Read bounded Retry-After evidence from headers or a provider JSON body."""
    value = headers.get("Retry-After") if headers else None
    if value:
        try:
            return max(1.0, float(value) + 0.25)
        except (TypeError, ValueError):
            pass
    try:
        message = str((json.loads(raw.decode("utf-8")).get("error") or {}).get("message") or "")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        message = ""
    match = re.search(r"try again in\s+((?:[0-9.]+(?:ms|h|m|s))+)", message, re.IGNORECASE)
    if not match:
        return None
    factors = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    seconds = sum(float(value) * factors[unit.lower()] for value, unit in re.findall(r"([0-9.]+)(ms|h|m|s)", match.group(1), re.IGNORECASE))
    return max(1.0, seconds + 0.25)


def _extract_json_object(content: str) -> dict[str, Any]:
    candidate = content.strip()
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        objects: list[dict[str, Any]] = []
        index = 0
        while index < len(candidate):
            start = candidate.find("{", index)
            if start < 0:
                break
            depth = 0
            in_string = False
            escaped = False
            end = None
            for cursor in range(start, len(candidate)):
                character = candidate[cursor]
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                    continue
                if character == '"':
                    in_string = True
                elif character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        end = cursor + 1
                        break
            if end is None:
                raise ProviderResponseError("Provider returned malformed or incomplete repair data (unclosed JSON object).")
            try:
                value = json.loads(candidate[start:end])
            except json.JSONDecodeError:
                index = end
                continue
            if isinstance(value, dict):
                objects.append(value)
            index = end
        if len(objects) != 1:
            detail = "no complete JSON object" if not objects else "multiple JSON objects"
            raise ProviderResponseError(f"Provider returned malformed or incomplete repair data ({detail}).")
        parsed = objects[0]
    if not isinstance(parsed, dict):
        raise ProviderResponseError("Provider repair content must be one JSON object.")
    return parsed

class LLMProvider(ABC):
    name = "unknown"; model = "unknown"
    @abstractmethod
    def propose_repair(self, context: Mapping[str, Any]) -> RepairResult: ...

class ConfiguredProvider(LLMProvider):
    def __init__(self, name: str, model: str, base_url: str, api_key: str | None = None):
        self.name, self.model, self.base_url, self.api_key = name, model, base_url.rstrip("/"), api_key
    def propose_repair(self, context: Mapping[str, Any]) -> RepairResult:
        test_only = context.get("purpose") == "regression_test_generation"
        instruction = (
            "Generate executable regression and, where practical, targeted integration TEST FILES ONLY. Never modify production files. "
            "Return exactly one JSON object with a tests array. Each tests item must contain path, type (regression or integration), purpose, content, and dependency_path (empty for regression; actual caller -> callee for integration). "
            "content must be the complete Python test file, not a diff, snippet, or markdown. Paths must be relative and under tests/ai_generated/. "
            "Generate at least one regression test for the repaired behavior. Assert only requirements explicitly supported by the PR intent, source, repair, or retry evidence; do not invent edge-case semantics. "
            "Use type=integration only when the test source actually exercises a real cross-component caller path demonstrated in the supplied source and graph. Otherwise use type=regression. Do not infer a call from similar names or behavior. "
            "For dependency_path use fully qualified FUNCTION names on both sides, not only module names, separated by the literal ASCII ->. For HTTP tests name the decorated route function as caller and invoke its actual URL through TestClient. "
            "Every generated test file must compile and pass Ruff check unchanged, including import ordering and two blank lines before top-level test functions. Import only symbols actually used; do not import pytest unless pytest APIs are required. No unused imports. Prefer plain test functions over pytest parametrization when a few explicit assertions suffice. "
            "Honor original_environment_validation. Do not request app/client or other fixtures unless shown in repository test context. Execute the actual repaired behavior; source-text assertions do not establish a behavioral repair. Use real caller code for integration. "
            "Return one regression and one integration test whenever supplied source demonstrates a real caller path. Omit integration only when no real multi-component path exists. Do not replace execution with inspecting source text. "
            if test_only else
            "Propose one minimal production code repair. Passing existing tests does not establish correctness. "
            "Return STRICT JSON with exactly explanation, confidence, affected_files, patch. Do not generate tests in this request. "
            "Use only requirements supported by PR intent, base/head source, diff and validation evidence. "
            "Use standard git diff output, NOT apply_patch tool format. Bare @@ is apply_patch format and is rejected by git. The patch applies to HEAD source, never base source. Copy unchanged lines exactly, including indentation. "
            "The supplied PR diff describes BASE to HEAD, not your repair. Do not copy its hunk headers: compute new ranges from the numbered HEAD source and the exact body of your own patch. "
            "Every file section MUST have these four headers, replacing angle-bracket fields with real paths/integers:\n"
            "diff --git a/<path> b/<path>\n--- a/<path>\n+++ b/<path>\n@@ -<start>,<count> +<start>,<count> @@\n"
            "Never output a bare @@. Count old hunk lines as context plus deletions; new hunk lines as context plus additions. "
            "Use conventional git diff context: THREE unchanged lines before and THREE after each replacement where available. Use minimal edits without explanatory comments in code. "
            "For a one-line replacement, three context lines on each side produce exactly seven old lines and seven new lines. Include all six context lines if you declare a count of seven; blank context lines count too. "
            "Start coordinates identify the FIRST context line, not the edited line. Blank context lines have one leading space. "
            "Before returning, count every hunk body line and verify the declared old/new counts exactly match it. "
            "Do not include *** End Patch, placeholders, invented code, or validation claims. "
            "If there is no evidence-backed repair, return an empty patch with an explanation. "

        )
        output_rules = (
            "Return only the structured JSON test-file contract; do not return a patch or markdown. Context:\n"
            if test_only else
            "Return the complete production patch object only. Do not invent a patch or test semantics when evidence is insufficient. Context:\n"
        )
        repair_context = _compact_repair_context(context)
        retry_evidence = context.get("failure_feedback") or {}
        if not test_only and retry_evidence.get("patch_rejection_reason"):
            # A retry must not keep anchoring the model on the original PR's
            # hunk header. Supply exact head evidence and the prior proposal's
            # intent; the model must author the entire replacement diff.
            retry_keys = {"repository", "pr_title", "pr_body", "base_source_snippets", "changed_files",
                          "changed_symbols", "relevant_original_source_snippets", "head_source_window_ranges",
                          "numbered_head_source_windows", "head_file_line_counts", "previous_retry_failure_evidence"}
            repair_context = {key: value for key, value in repair_context.items() if key in retry_keys}
            repair_context["previous_proposal_intent"] = retry_evidence.get("prior_repair_explanation")
        if test_only:
            analysis_context = context.get("analysis_context") or {}
            repair_context = _compact_repair_context(analysis_context)
            repair = context.get("repair") or {}
            affected = set(repair.get("affected_files") or [])
            if affected:
                repair_context["relevant_original_source_snippets"] = {
                    name: source for name, source in repair_context["relevant_original_source_snippets"].items()
                    if name in affected
                }
            repair_context.update({
                "purpose": context.get("purpose"),
                "test_targets": context.get("targets") or {},
                "required_repair_targets": context.get("required_repair_targets") or [],
                "language": context.get("language"),
                "framework": context.get("framework"),
                "validated_candidate_repair": {
                    key: repair.get(key) for key in ("explanation", "affected_files", "patch") if repair.get(key)
                },
                "candidate_source_snippets": analysis_context.get("candidate_source") or {},
                "test_generation_retry_feedback": context.get("validation_feedback"),
            })
            # Test requests need the actual candidate and callers. Repeating the
            # same files as base, head and candidate exhausted the provider's
            # request limit after the repair context had already been bounded.
            candidate_sources = repair_context["candidate_source_snippets"]
            for bucket in ("base_source_snippets", "relevant_original_source_snippets"):
                for name in list(repair_context[bucket]):
                    if name in candidate_sources:
                        repair_context[bucket].pop(name)
                        repair_context["omitted_context"].append(f"{bucket}:{name} (candidate supplied)")
            repair_context["test_targets"] = [
                {key: value for key, value in item.items() if key in {"name", "symbol", "file", "type"}}
                for item in (context.get("targets") or [])[:12] if isinstance(item, dict)
            ]
            # The accepted repair already identifies the defect. Do not repeat
            # the entire PR (including unrelated cleanup) in a test request.
            test_keys = {"repository", "pr_title", "pr_body", "purpose", "language", "framework",
                         "required_repair_targets", "validated_candidate_repair", "candidate_source_snippets",
                         "test_generation_retry_feedback", "previous_retry_failure_evidence",
                         "demonstrated_dependency_paths", "original_environment_validation"}
            repair_context = {key: value for key, value in repair_context.items() if key in test_keys}
        system = (
            "You generate evidence-backed pytest files only. Obey the JSON test-file contract exactly, invent no requirements, and emit Ruff-clean Python."
            if test_only else
            "You are a code repair engine. Return a complete git unified diff inside the four-field JSON contract. File headers and numeric hunk coordinates are mandatory. Never invent source or skip patch headers."
        )
        prompt = instruction + output_rules + json.dumps(repair_context, ensure_ascii=False, default=str)
        feedback = (context.get("failure_feedback") or {}) if not test_only else {}
        rejection = feedback.get("patch_rejection_reason") or feedback.get("candidate_error")
        if rejection:
            prompt += ("\n\nPREVIOUS ATTEMPT REJECTED: " + str(rejection)[:1600]
                       + "\nGenerate the entire patch again from scratch. Do not reuse the invalid hunk header. "
                         "Every blank context line also counts as one old line and one new line. "
                         "Count your final body before writing its header. Return the entire four-field JSON object.")
        request_payload = {"model": self.model, "messages": [{"role":"system","content":system}, {"role":"user","content":prompt}],
                           "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
                           "max_tokens": (int(os.getenv("LLM_TEST_MAX_TOKENS", "1800")) if test_only
                                          else int(os.getenv("LLM_MAX_TOKENS", "2400")))}
        if self.name.lower() == "groq" and self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            # Keep the provider reasoning effort configurable independently of
            # the completion budget for the strict JSON document and diff.
            request_payload["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT", "medium")
        # Production repairs have a deliberately closed contract.  Test-file
        # generation has its own validated contract in TestGenerationService.
        request_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": _TEST_RESPONSE_SCHEMA if test_only else _PATCH_RESPONSE_SCHEMA,
        }
        use_code_tool = (os.getenv("LLM_PATCH_CODE_TOOL", "0") == "1" and not test_only
                         and self.name.lower() == "groq"
                         and self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"})
        if use_code_tool:
            # Explicit opt-in: the provider's remote sandbox receives source.
            # The model computes its own patch; the application never fills gaps.
            request_payload.pop("response_format", None)
            request_payload["tools"] = [{"type": "code_interpreter"}]
            request_payload["tool_choice"] = "required"
            request_payload["messages"][0]["content"] = (
                "First use the python code interpreter tool to compute an evidence-backed repair. "
                "JSON is the FINAL ANSWER format, not a tool name. Do not call a tool named JSON or production_repair. "
                "Infer the minimal corrected production source yourself from the evidence. In python, compare the exact "
                "complete head file with your corrected version using difflib.unified_diff with three context lines, "
                "a/ and b/ filenames, and a diff --git prefix. Compute the entire diff in the tool, including numeric "
                "hunk coordinates and counts. After execution, return exactly explanation, confidence, affected_files, "
                "patch as one JSON object containing the computed diff. Never edit tests or invent requirements."
            )
            full_sources = {name: source for name, source in (context.get("source_context") or {}).items()
                            if name in (context.get("changed_files") or [])}
            request_payload["messages"][1]["content"] = "Compute the repair using python, then return the final JSON. Evidence:\n" + json.dumps({
                "repository": context.get("repository"), "pr_title": context.get("pr_title"), "pr_body": context.get("pr_body"),
                "complete_head_source": full_sources, "base_source": context.get("base_source_context") or {},
                "pr_diff": context.get("diff"), "previous_rejection": _compact_failure_feedback(context.get("failure_feedback") or {}),
            }, ensure_ascii=False, default=str)
        payload = json.dumps(request_payload).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ai-change-impact-platform/1.0",
        }
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        max_transport_attempts = max(1, min(int(os.getenv("LLM_TRANSPORT_ATTEMPTS", "3")), 4))
        status = None; content_type = None; raw = b""; transport_attempt = 0
        structured_fallback_used = False
        provider_request_count = 0
        while transport_attempt < max_transport_attempts:
            transport_attempt += 1
            provider_request_count += 1
            request = urllib.request.Request(endpoint, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    status = response.getcode()
                    content_type = response.headers.get("Content-Type")
                    raw = response.read()
                break
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                content_type = exc.headers.get("Content-Type") if exc.headers else None
                structured_failure = (
                    self.name.lower() == "groq" and exc.code == 400
                    and any(code in raw.decode("utf-8", errors="replace").lower()
                            for code in ("json_validate_failed", "tool_use_failed"))
                    and request_payload.get("response_format", {}).get("type") == "json_schema"
                )
                if structured_failure and not structured_fallback_used:
                    # This is the same semantic request. Only provider-side schema
                    # enforcement changes; the returned object is still validated
                    # against the full local contract below.
                    structured_fallback_used = True
                    request_payload["response_format"] = {"type": "json_object"}
                    payload = json.dumps(request_payload).encode()
                    transport_attempt -= 1
                    continue
                if exc.code == 400 and any(code in raw.decode("utf-8", errors="replace").lower()
                                           for code in ("json_validate_failed", "tool_use_failed")):
                    raise ProviderResponseError("Provider rejected malformed structured output. " + _response_diagnostic(exc.code, content_type, raw)) from exc
                reason = _provider_error_reason(exc.code, raw)
                retryable = exc.code == 429 or exc.code >= 500
                retry_after = _retry_after_seconds(exc.headers, raw)
                # Preserve long provider reset intervals as evidence and return
                # control instead of repeatedly retrying an exhausted daily quota.
                if retryable and transport_attempt < max_transport_attempts and (retry_after is None or retry_after <= 60):
                    wait_seconds = retry_after if retry_after is not None else min(2 ** (transport_attempt - 1), 8)
                    if exc.code == 429:
                        wait_seconds = max(wait_seconds, min(60.0, max(0.0, float(os.getenv("LLM_RATE_LIMIT_MIN_WAIT_SECONDS", "0")))))
                    time.sleep(wait_seconds)
                    continue
                raise ProviderTransportError(
                    f"{self.name} {reason}. {_response_diagnostic(exc.code, content_type, raw)}",
                    error_type="RATE_LIMIT" if exc.code == 429 else "HTTP_ERROR",
                    status_code=exc.code, retryable=retryable,
                    retry_after_seconds=retry_after, transport_attempts=transport_attempt,
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if transport_attempt < max_transport_attempts:
                    time.sleep(min(2 ** (transport_attempt - 1), 8)); continue
                raise ProviderTransportError(
                    f"{self.name} timeout.", error_type="TIMEOUT", retryable=True,
                    transport_attempts=transport_attempt,
                ) from exc
            except urllib.error.URLError as exc:
                timed_out = isinstance(exc.reason, (TimeoutError, socket.timeout))
                if transport_attempt < max_transport_attempts:
                    time.sleep(min(2 ** (transport_attempt - 1), 8)); continue
                raise ProviderTransportError(
                    f"{self.name} {'timeout' if timed_out else 'provider request failed'}.",
                    error_type="TIMEOUT" if timed_out else "NETWORK_ERROR", retryable=True,
                    transport_attempts=transport_attempt,
                ) from exc

        diagnostic = _response_diagnostic(status, content_type, raw)
        if not raw.strip():
            raise ProviderResponseError(f"{self.name} returned an empty response body. {diagnostic}")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(f"{self.name} returned a non-JSON response. {diagnostic}") from exc
        if not isinstance(body, dict):
            raise ProviderResponseError(f"{self.name} returned an invalid response envelope. {diagnostic}")
        if body.get("error"):
            reason = _provider_error_reason(status, json.dumps(body.get("error"), default=str).encode())
            raise ProviderResponseError(f"{self.name} {reason}. {diagnostic}")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderResponseError(f"{self.name} response is missing choices. {diagnostic}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError(f"{self.name} response is missing choices/message. {diagnostic}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            refusal = message.get("refusal")
            suffix = f" Refusal: {_safe_response_preview(str(refusal).encode())}" if refusal else ""
            raise ProviderResponseError(f"{self.name} response is missing choices/message/content.{suffix} {diagnostic}")
        try:
            data = _extract_json_object(content)
        except ProviderResponseError as exc:
            raise ProviderResponseError(f"{exc} {diagnostic}") from exc
        data = dict(data)
        if "confidence" in data:
            data["confidence"] = normalize_confidence(data.get("confidence"))
        if not test_only:
            _validate_repair_contract(data, require_tests=False)
        patch = data.get("patch")
        if isinstance(patch, str):
            patch = normalize_patch(patch)
            try:
                _validate_unified_diff(patch)
            except ProviderResponseError as exc:
                raise PatchContractError(str(exc), patch=patch, data=data) from exc
            if not patch.strip(): patch = None
            elif not patch.endswith('\n'): patch += '\n'
        # Models sometimes emit one concise change description instead of a
        # one-item JSON array. Normalize that harmless shape variation here;
        # patch parsing and deterministic validation remain authoritative.
        affected_files = data.get("affected_files") or []
        if isinstance(affected_files, str):
            affected_files = [affected_files]
        proposed_changes = data.get("proposed_changes") or []
        if isinstance(proposed_changes, str):
            proposed_changes = [proposed_changes]
        elif isinstance(proposed_changes, dict):
            proposed_changes = [{"file": key, **(value if isinstance(value, dict) else {"description": value})}
                                for key, value in proposed_changes.items()]
        confidence = normalize_confidence(data.get("confidence"))
        choice = choices[0]
        response_metadata = {
            "http_status": status,
            "content_type": content_type,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage") if isinstance(body.get("usage"), dict) else {},
            "transport_attempts": transport_attempt,
            "provider_request_count": provider_request_count,
            "structured_output_mode": "provider_code_tool" if use_code_tool else "json_object_fallback" if structured_fallback_used else "strict_json_schema",
            "provider_tool_calls": len(message.get("executed_tools") or []),
            "max_completion_tokens": request_payload["max_tokens"],
            "reasoning_effort": request_payload.get("reasoning_effort"),
        }
        response_warning = f"Provider response diagnostic: HTTP {status}; content-type {content_type or '<missing>'}"
        if test_only:
            tests = data.get("tests")
            if not isinstance(tests, list):
                raise ProviderResponseError("Provider test response must contain a tests array of complete files.")
            return RepairResult(status="available", provider=self.name, model=self.model,
                                issue_summary=data.get("issue_summary") or "Provider returned structured generated tests.",
                                explanation=data.get("explanation"), proposed_changes=tests,
                                confidence=confidence, warnings=[response_warning], response_metadata=response_metadata)
        if not patch:
            return RepairResult(status="unavailable", provider=self.name, model=self.model,
                                issue_summary=data.get("issue_summary") or "Provider returned no safe patch.",
                                explanation=data.get("explanation"), affected_files=affected_files,
                                proposed_changes=proposed_changes, confidence=confidence,
                                warnings=["The provider returned analysis but no applicable patch; no candidate workspace was changed.", response_warning],
                                response_metadata=response_metadata)
        return RepairResult(status="available", provider=self.name, model=self.model,
                            issue_summary=data.get("issue_summary"), patch=patch,
                            explanation=data.get("explanation"), affected_files=affected_files,
                            regression_test=dict(data.get("regression_test") or {}),
                            integration_test=dict(data.get("integration_test") or {}),
                            proposed_changes=proposed_changes, confidence=confidence,
                            warnings=[response_warning], response_metadata=response_metadata)

class FallbackProvider(LLMProvider):
    """Try explicitly configured providers on transport failures only."""

    def __init__(self, providers):
        self.providers = providers
        self.name, self.model = providers[0].name, providers[0].model

    def propose_repair(self, context):
        attempts = []
        for provider in self.providers:
            record = {"provider": provider.name, "model": provider.model}
            try:
                result = provider.propose_repair(context)
            except ProviderTransportError as exc:
                attempts.append({**record, "status": "PROVIDER_ERROR", "error_type": exc.error_type,
                                 "status_code": exc.status_code, "transport_attempts": exc.transport_attempts})
                exc.provider_attempts = list(attempts)
                exc.provider_name, exc.provider_model = provider.name, provider.model
                if provider is self.providers[-1]:
                    raise
                continue
            except ProviderResponseError as exc:
                exc.provider_attempts = [*attempts, {**record, "status": "SEMANTIC_ERROR"}]
                exc.provider_name, exc.provider_model = provider.name, provider.model
                raise
            result.provider, result.model = provider.name, provider.model
            result.response_metadata["provider_attempts"] = [*attempts, {**record, "status": result.status}]
            return result


class RepairService:
    def __init__(self, provider: LLMProvider | None = None):
        # Explicit providers are stable test/application dependencies. Environment
        # providers are deliberately resolved live so a worker imported before its
        # process environment was populated cannot remain stale forever.
        self._explicit_provider = provider
    @property
    def provider(self) -> LLMProvider | None:
        if self._explicit_provider is not None:
            return self._explicit_provider
        providers = []
        # Each fallback has its own credentials, endpoint and model. Never send
        # a primary credential to a different provider or infer authorization.
        for prefix in ("LLM", "LLM_FALLBACK", *[f"LLM_FALLBACK_{i}" for i in range(2, 5)]):
            name, url, key = (os.getenv(f"{prefix}_{field}") for field in ("PROVIDER", "BASE_URL", "API_KEY"))
            model = os.getenv(f"{prefix}_MODEL", "qwen2.5-coder" if prefix == "LLM" else "")
            if name and url and key and model and not any(
                item.name.lower() == name.lower() and item.base_url == url.rstrip('/')
                and item.model == model and item.api_key == key for item in providers
            ):
                providers.append(ConfiguredProvider(name, model, url, key))
        return FallbackProvider(providers) if len(providers) > 1 else providers[0] if providers else None
    def readiness(self) -> dict[str, Any]:
        provider = self.provider
        model_labels = {
            "qwen/qwen2.5-coder-7b-instruct": "Qwen2.5-Coder 7B",
            "qwen/qwen-2.5-coder-32b-instruct": "Qwen2.5-Coder 32B",
        }
        display_model = model_labels.get(provider.model, provider.model) if provider else None
        return {
            "status": "READY" if provider else "NOT_CONFIGURED",
            "provider": provider.name if provider else None,
            "model": provider.model if provider else None,
            "provider_display": "OpenRouter" if provider and provider.name.lower() == "openrouter" else (provider.name.title() if provider else None),
            "model_display": display_model,
            "base_url_configured": bool(provider),
            "api_key_configured": bool(provider),
            "fallbacks": [{"provider": item.name, "model": item.model}
                          for item in provider.providers[1:]] if isinstance(provider, FallbackProvider) else [],
        }
    def propose(self, context: Mapping[str, Any] | None = None, **kwargs: Any) -> RepairResult:
        payload = dict(context or {}); payload.update(kwargs)
        provider = self.provider
        if provider is None:
            return RepairResult(status="unavailable", issue_summary="Repair provider unavailable. Configure LLM_PROVIDER and LLM_BASE_URL.", warnings=["Repairs are proposals only; the original repository was not modified."])
        try: return provider.propose_repair(payload)
        except PatchContractError as exc:
            return RepairResult(
                status="error", provider=getattr(exc, "provider_name", provider.name), model=getattr(exc, "provider_model", provider.model),
                issue_summary="Provider returned a rejected patch.", patch=exc.patch,
                response_metadata={"provider_attempts": getattr(exc, "provider_attempts", [])},
                explanation=exc.data.get("explanation"),
                affected_files=exc.data.get("affected_files") or [],
                regression_test=dict(exc.data.get("regression_test") or {}),
                integration_test=dict(exc.data.get("integration_test") or {}),
                confidence=exc.data.get("confidence"), warnings=[str(exc)],
            )
        except ProviderTransportError as exc:
            from .ci_tools import redact
            return RepairResult(
                status="error", provider=getattr(exc, "provider_name", provider.name), model=getattr(exc, "provider_model", provider.model),
                issue_summary="Provider request failed.", warnings=[redact(str(exc))],
                response_metadata={"provider_attempts": getattr(exc, "provider_attempts", [])},
                provider_error={
                    "type": exc.error_type, "status_code": exc.status_code,
                    "retryable": exc.retryable,
                    "retry_after_seconds": exc.retry_after_seconds,
                    "transport_attempts": exc.transport_attempts,
                },
            )
        except ProviderResponseError as exc:
            from .ci_tools import redact
            return RepairResult(
                status="error", provider=getattr(exc, "provider_name", provider.name), model=getattr(exc, "provider_model", provider.model),
                issue_summary="Provider structured output failed local validation.",
                response_metadata={"provider_attempts": getattr(exc, "provider_attempts", [])},
                warnings=[redact(str(exc))],
                provider_error={
                    "type": "STRUCTURED_OUTPUT_ERROR", "status_code": None,
                    "retryable": False, "transport_attempts": 1,
                },
            )
        except Exception as exc:
            from .ci_tools import redact
            return RepairResult(status="error", provider=provider.name, model=provider.model,
                                issue_summary="Repair provider failed.", warnings=[redact(str(exc))])
    def summary(self, context: Mapping[str, Any] | None = None) -> RepairSummary:
        result = self.propose(context)
        return RepairSummary(status=result.status, patch=result.patch, explanation=result.explanation)
