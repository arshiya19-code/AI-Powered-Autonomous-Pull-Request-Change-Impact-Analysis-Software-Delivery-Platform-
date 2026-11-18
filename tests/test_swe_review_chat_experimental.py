from dataset.swe_review_chat_experimental import is_github_auth_failure


def test_auth_failures_are_retryable_infrastructure_failures():
    assert is_github_auth_failure({"status": "GITHUB_AUTH_FAILED"})
    assert is_github_auth_failure({"failure_reason": "HTTPError: 401 Client Error: Unauthorized"})
    assert is_github_auth_failure({"failure_reason": "BAD_CREDENTIALS"})
    assert not is_github_auth_failure({"status": "ANALYSIS_FAILED", "failure_reason": "parser failed"})
