import io
import json
import urllib.error

import pytest

from backend.app.services.repair import (
    ConfiguredProvider,
    RepairService,
    ProviderResponseError,
    ProviderTransportError,
    normalize_confidence,
    _extract_json_object,
)
from backend.app.services.test_generation import TestGenerationService

VALID = {
    "explanation": "Fix arithmetic",
    "confidence": 0.9,
    "patch": "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
    "affected_files": ["app.py"],
    "regression_test": {
        "path": "tests/ai_generated/test_app.py", "target": "app.value",
        "reason": "Protect the repaired value.",
        "source": "from app import value\n\n\ndef test_value():\n    assert value() == 'new'\n",
    },
    "integration_test": {
        "path": "tests/ai_generated/test_app_integration.py", "target": "caller.value",
        "dependency_path": "caller.value -> app.value", "reason": "Protect the caller path.",
        "source": "from caller import value\n\n\ndef test_caller_value():\n    assert value() == 'new'\n",
    },
}


class Response:
    def __init__(self, body, status=200, content_type="application/json"):
        self.body = body
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def getcode(self):
        return self.status

    def read(self):
        return self.body


def provider():
    return ConfiguredProvider("openrouter", "qwen/test", "https://openrouter.ai/api/v1", "test-key")


@pytest.mark.parametrize("content", [
    json.dumps(VALID),
    f"```json\n{json.dumps(VALID)}\n```",
    f"I found one safe repair.\n{json.dumps(VALID)}\nPlease review it.",
])
def test_accepts_supported_json_wrappers(monkeypatch, content):
    envelope = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(envelope))
    result = provider().propose_repair({"diff": "example"})
    assert result.status == "available"
    assert result.patch == VALID["patch"]
    assert result.warnings[0].startswith("Provider response diagnostic: HTTP 200")


@pytest.mark.parametrize(("raw", "expected"), [
    (0.95, 0.95), ("0.95", 0.95), ("95%", 0.95), ("high", None), (True, None), ("120%", None),
])
def test_confidence_is_non_blocking_display_metadata(raw, expected):
    assert normalize_confidence(raw) == expected


def test_missing_file_markers_are_rejected_without_synthesis(monkeypatch):
    value={**VALID,"patch":"diff --git a/app.py b/app.py\n@@\n-old\n+new\n"}
    response=json.dumps({"choices":[{"message":{"content":json.dumps(value)}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen",lambda *args,**kwargs:Response(response))
    with pytest.raises(ProviderResponseError,match="headers"):
        provider().propose_repair({"source_context":{"app.py":"old\n"}})


def test_production_repairs_request_closed_groq_json_schema(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    captured = {}

    def request_handler(request, **kwargs):
        captured.update(json.loads(request.data.decode()))
        return Response(json.dumps({"choices": [{"message": {"content": json.dumps(VALID)}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", request_handler)
    ConfiguredProvider("groq", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1", "test-key").propose_repair({})
    schema = captured["response_format"]
    assert captured["reasoning_effort"] == "medium"
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["schema"]["required"] == [
        "explanation", "confidence", "affected_files", "patch",
    ]
    assert schema["json_schema"]["schema"]["additionalProperties"] is False


def test_groq_schema_failure_falls_back_once_to_json_object(monkeypatch):
    payloads = []

    def request_handler(request, **kwargs):
        payloads.append(json.loads(request.data.decode()))
        if len(payloads) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 400, "schema validation failed",
                {"Content-Type": "application/json"},
                io.BytesIO(b'{"error":{"code":"json_validate_failed"}}'),
            )
        return Response(json.dumps({"choices": [{"message": {"content": json.dumps(VALID)}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", request_handler)
    result = ConfiguredProvider("groq", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1", "test-key").propose_repair({})

    assert [item["response_format"]["type"] for item in payloads] == ["json_schema", "json_object"]
    assert result.response_metadata["structured_output_mode"] == "json_object_fallback"
    assert result.response_metadata["provider_request_count"] == 2


def test_test_generation_uses_a_smaller_completion_budget(monkeypatch):
    monkeypatch.delenv("LLM_TEST_MAX_TOKENS", raising=False)
    captured = {}
    response = {"choices": [{"message": {"content": json.dumps({"tests": []})}}]}

    def request_handler(request, **kwargs):
        captured.update(json.loads(request.data.decode()))
        return Response(json.dumps(response).encode())

    monkeypatch.setenv("LLM_MAX_TOKENS", "2400")
    monkeypatch.setattr("urllib.request.urlopen", request_handler)
    provider().propose_repair({"purpose": "regression_test_generation"})

    assert captured["max_tokens"] == 1800
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"]["properties"]["tests"]["maxItems"] == 2


def test_rejected_provider_patch_is_preserved_for_workflow_retry(monkeypatch):
    invalid = {**VALID, "patch": "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ ... @@\n-old\n+new\n"}
    envelope = json.dumps({"choices": [{"message": {"content": json.dumps(invalid)}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(envelope))

    result = RepairService(provider()).propose({"diff": "example"})

    assert result.status == "error"
    assert result.patch == invalid["patch"]
    assert "complete unified-diff hunk" in result.warnings[0]


def test_bare_hunk_is_rejected_even_with_unique_source(monkeypatch):
    patch="diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@\n-old\n+new\n"
    response=json.dumps({"choices":[{"message":{"content":json.dumps({**VALID,"patch":patch})}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen",lambda *args,**kwargs:Response(response))
    with pytest.raises(ProviderResponseError,match="hunk"):
        provider().propose_repair({"source_context":{"app.py":"old\n"}})


@pytest.mark.parametrize("content", [
    '{"patch": "incomplete',
    'Prefix {"outer":{"patch":"incomplete"}',
    '{} trailing {"patch":"second"}',
    "I cannot provide a patch for this request.",
])
def test_rejects_malformed_incomplete_or_explanatory_content(content):
    with pytest.raises(ProviderResponseError):
        _extract_json_object(content)


@pytest.mark.parametrize("body, expected", [
    (b"", "empty response body"),
    (b"<html>gateway error</html>", "non-JSON response"),
    (b'{"choices":[]}', "missing choices"),
    (b'{"choices":[{"message":{}}]}', "missing choices/message/content"),
    (b'{"error":{"message":"No endpoints found for model"}}', "model unavailable"),
])
def test_classifies_invalid_success_responses(monkeypatch, body, expected):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(body, content_type="text/html"))
    with pytest.raises(ProviderResponseError, match=expected):
        provider().propose_repair({})


@pytest.mark.parametrize("status, body, expected", [
    (401, b'{"error":{"message":"invalid API key"}}', "authentication failed"),
    (402, b'{"error":{"message":"insufficient credits"}}', "insufficient credits"),
    (429, b'{"error":{"message":"rate limit exceeded"}}', "rate limited"),
    (404, b'{"error":{"message":"invalid model"}}', "model unavailable"),
    (504, b"gateway timeout", "timeout"),
])
def test_classifies_http_errors_without_exposing_credentials(monkeypatch, status, body, expected):
    error = urllib.error.HTTPError("https://openrouter.ai", status, "error", {"Content-Type": "application/json"}, io.BytesIO(body))
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(ProviderResponseError, match=expected) as captured:
        provider().propose_repair({})
    assert "test-key" not in str(captured.value)


def test_retryable_transport_failure_is_retried_before_returning_a_repair(monkeypatch):
    success = json.dumps({"choices": [{"message": {"content": json.dumps(VALID)}, "finish_reason": "stop"}]}).encode()
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                "https://api.groq.com/openai/v1/chat/completions", 429, "rate limited",
                {"Content-Type": "application/json", "Retry-After": "0"},
                io.BytesIO(b'{"error":{"message":"rate limit exceeded"}}'),
            )
        return Response(success)

    monkeypatch.setenv("LLM_TRANSPORT_ATTEMPTS", "2")
    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = RepairService(provider()).propose({"diff": "example"})

    assert result.status == "available"
    assert len(calls) == 2
    assert result.response_metadata["transport_attempts"] == 2


def test_test_generation_transport_error_does_not_consume_semantic_attempts(tmp_path):
    class RateLimitedProvider:
        name = "groq"
        model = "openai/gpt-oss-20b"

        def __init__(self):
            self.calls = 0

        def propose_repair(self, context):
            self.calls += 1
            raise ProviderTransportError(
                "groq rate limited", error_type="RATE_LIMIT", status_code=429,
                retryable=True, retry_after_seconds=30, transport_attempts=3,
            )

    rate_limited = RateLimitedProvider()
    with pytest.raises(ProviderTransportError):
        TestGenerationService(rate_limited).generate(root=tmp_path, changed_symbols=["app.primes"])

    assert rate_limited.calls == 1
