import pytest

from backend.app.schemas.models import RepairResult
from backend.app.services.repair import (
    FallbackProvider, LLMProvider, ProviderResponseError, ProviderTransportError, RepairService,
)


class Provider(LLMProvider):
    def __init__(self, name, outcome):
        self.name, self.model, self.outcome = name, name + '-model', outcome
        self.calls = 0

    def propose_repair(self, context):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_rate_limit_fallback_records_actual_provider_and_attempts():
    primary = Provider('primary', ProviderTransportError('quota', error_type='RATE_LIMIT', status_code=429))
    fallback = Provider('fallback', RepairResult(status='available'))
    result = RepairService(FallbackProvider([primary, fallback])).propose({})
    assert (result.provider, result.model) == ('fallback', 'fallback-model')
    attempts = result.response_metadata['provider_attempts']
    assert [item['provider'] for item in attempts] == ['primary', 'fallback']
    assert attempts[0]['status_code'] == 429
    assert primary.calls == fallback.calls == 1


def test_semantic_failure_does_not_switch_provider():
    primary = Provider('primary', ProviderResponseError('invalid JSON'))
    fallback = Provider('fallback', RepairResult(status='available'))
    result = RepairService(FallbackProvider([primary, fallback])).propose({})
    assert result.provider_error['type'] == 'STRUCTURED_OUTPUT_ERROR'
    assert fallback.calls == 0


def test_all_transport_failures_preserve_each_provider_without_semantic_retry():
    providers = [Provider(name, ProviderTransportError('timeout', error_type='TIMEOUT'))
                 for name in ('primary', 'fallback')]
    chain = FallbackProvider(providers)
    with pytest.raises(ProviderTransportError) as caught:
        chain.propose_repair({'purpose': 'regression_test_generation'})
    assert len(caught.value.provider_attempts) == 2
    assert caught.value.provider_name == 'fallback'
    assert [provider.calls for provider in providers] == [1, 1]


def test_incomplete_fallback_is_not_used(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'primary')
    monkeypatch.setenv('LLM_MODEL', 'primary-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://primary.invalid/v1')
    monkeypatch.setenv('LLM_API_KEY', 'test-only')
    monkeypatch.setenv('LLM_FALLBACK_PROVIDER', 'fallback')
    monkeypatch.delenv('LLM_FALLBACK_API_KEY', raising=False)
    assert RepairService().provider.name == 'primary'
